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

protected:
  // HW 三件套（shared_ptr 以便 target 可拷贝，拷贝为浅拷贝共享状态）
  std::shared_ptr<buff_algo::BuffTracker> tracker_;
  std::shared_ptr<buff_algo::BuffPredictor> predictor_;

  buff_algo::BuffState state_;
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d aim_point_world_ = Eigen::Vector3d::Zero();

  // 瞄准点微调：HW predictor 给出的是扇叶中心（实测距符中心约 0.55m），
  // aim_radius_offset_ 表示在"符中心->扇叶中心"方向上再外推多少米：
  //   0    = 扇叶中心
  //   +0.08 ≈ 装甲板中心
  // 之所以用"相对偏移"而不是"距圆心的绝对半径"：绝对半径依赖扇叶中心真实半径，
  // 而这个量以前被硬编码成 0.70m（实测约 0.55m），一旦估计有偏，绝对半径会把
  // 瞄准点整体推出扇叶；相对偏移则始终贴着扇叶。
  double aim_radius_offset_ = 0.08;
  Eigen::Vector3d r_center_world_ = Eigen::Vector3d::Zero();  // 符中心（世界系）
  bool has_r_center_ = false;

  // 把扇叶中心的打击点沿半径方向外推 aim_radius_offset_；无符中心或不需调整时原样返回
  Eigen::Vector3d apply_aim_radius(const Eigen::Vector3d & leaf_world) const;

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
