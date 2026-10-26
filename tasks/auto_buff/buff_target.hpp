#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buff_algo/predictor/buff_predictor.hpp"
#include "buff_algo/tracker/buff_tracker.hpp"
#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
/// Target 基类（合并 HW：内部使用 BuffTracker + BuffPredictor）

class Target
{
public:
  Target();
  virtual ~Target() = default;
  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;  // 纯虚函数

  virtual void predict(double dt) = 0;  // 纯虚函数

  // 返回 HW predictor 预测的世界系打击点（aimer 弹道迭代使用）
  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  bool is_unsolve() const;

  // 兼容旧接口：x[4]=yaw, x[5]=roll, x[6]=roll_velocity（调试/aimer 使用）
  Eigen::VectorXd ekf_x() const;

  double spd = 0;  // 调试用

  // 只读调试接口：HW 跟踪/预测内部状态
  buff_algo::BuffState buff_state() const { return state_; }
  buff_algo::StatusType tracker_status() const
  {
    return tracker_ ? tracker_->status() : buff_algo::StatusType::LOST;
  }

  // 运行时切换小符/大符：转发给 tracker，模式变化时 tracker 内部会 reset
  // 观测器与状态机，重新从 CONVERGING 收敛（模式不变时调用无副作用）。
  // SmallTarget/BigTarget 的区别仅在构造时的初始模式，运行期可用本接口互切。
  void set_mode(buff_algo::BuffMode mode)
  {
    if (tracker_) tracker_->set_mode(mode);
  }

  // ===== 调试接口：预测值导出 =====
  // 最近一次 predict() 得到的打击点（世界系）。aimer 弹道迭代结束后，
  // 它就是最终参与弹道解算的"飞行时间时域瞄准点"。
  Eigen::Vector3d aim_point_world() const { return aim_point_world_; }

  // 预测"从状态基准时刻起 dt 秒后"的扇叶 roll（弧度），内部含 state_age 补偿
  double predict_roll(double dt) const
  {
    return predictor_ ? static_cast<double>(predictor_->predict_roll(static_cast<float>(dt)))
                      : 0.0;
  }

protected:
  // HW 三件套（shared_ptr 以便 target 可拷贝，拷贝为浅拷贝共享状态）
  std::shared_ptr<buff_algo::BuffTracker> tracker_;
  std::shared_ptr<buff_algo::BuffPredictor> predictor_;

  buff_algo::BuffState state_;
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  // cam(前x左y上z)->gimbal 修正旋转，由 Solver 从 config 标定外参算出并下发
  Eigen::Matrix3d R_fix_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d aim_point_world_ = Eigen::Vector3d::Zero();

  bool unsolvable_ = true;
  int lost_cn_ = 0;
  bool first_in_ = true;
};

/// SmallTarget（小符：SMALL_BUFF 模式）

class SmallTarget : public Target
{
public:
  SmallTarget();  // 无配置构造（tracker 为空，不工作）
  explicit SmallTarget(const std::string & config_path);  // 读 hw_buff.config_dir

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;
};

/// BigTarget（大符：BIG_BUFF 模式，RANSAC 正弦拟合）

class BigTarget : public Target
{
public:
  BigTarget();
  explicit BigTarget(const std::string & config_path);

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;
};

}  // namespace auto_buff
#endif
