#ifndef AUTO_BUFF__BIG_RUNE__BIG_RUNE_TRACKER_HPP
#define AUTO_BUFF__BIG_RUNE__BIG_RUNE_TRACKER_HPP

// BigRuneTracker：RP 大符相位运动估计与 Adam 链路的适配门面。
//
// 职责（plan B：只替换大符"运动估计+目标切换"段）：
//   1. 把 solver 输出的 PowerRune（PnP 位姿）转成 big_rune::CandidateTarget
//      —— 相位定义迁移自 RP PhaseMotionEstimator::complete_candidate_target，
//      差异仅两点：up 向量 RP(0,-1,0)[车系y向下] -> Adam(0,0,1)[world z向上]；
//      法向朝向判断 RP 用 normal·armor_center（其车系原点在符轴上），
//      Adam world 原点在云台，等价改用 normal·rune_center。
//   2. 驱动 big_rune::LM_IRLS_BigRunePhaseMotionFilter（连续化+建模+迭代）。
//   3. 对外提供与 auto_buff::Target 一致的输出语义：
//        predict_aim_point(dt)  世界系打击点，dt 从"拍照时刻"起算
//        predict_roll(dt)       预测相位（兼容 HW roll 语义，调试用）
//      预测时域天然覆盖拍照->当前->飞行的全链路延迟（模型以绝对时间戳为参数）。

#include "big_rune/LM_IRLS_BigRunePhaseMotionFilter.hpp"
#include "buff_type.hpp"

#include <chrono>
#include <memory>
#include <vector>

namespace auto_buff
{

class BigRuneTracker
{
public:
  explicit BigRuneTracker(const big_rune::MotionFilterParams & cfg);

  // ===== 单帧更新 =====
  // 主候选来自 PowerRune 的 PnP 观测；副候选（大符两叶同亮的切换期）取自
  // PowerRune::extra_armor_centers_in_world（solver 已完成平面求交推算），
  // 复用主候选的符心/法向构造 CandidateTarget 后一并入队。
  // capture_time 必须是拍照时刻（与 PowerRune 的 PnP/IMU 对齐的那个时间戳）。
  void update(const PowerRune & pr, std::chrono::steady_clock::time_point capture_time);

  // 丢帧：无需推进滤波器（其队列按时间窗自动裁剪、间隔过大自动重置），
  // 这里只维持重置计数。
  void update_lost();

  // ===== 状态查询 =====
  // 运动方程是否有效（唯一可靠来源：calculate_motion 的返回值）。
  // 注意不能用 reference_timestamp != 0 判断——prepare_date 每次拟合都会刷新它。
  bool model_ready() const { return model_valid_; }
  bool has_target() const { return has_target_; }
  bool is_unsolve() const { return unsolvable_; }

  // 最近追踪目标的原始相位（[-pi, pi]）。建模前返回最新候选的观测相位。
  double phase() const;
  // 连续化相位（建模坐标系）。建模前等于 phase()。
  double continuous_phase() const;
  // 运动方程（model_ready() 为 false 时内容无意义）
  const big_rune::MotionModelParams & motion() const { return filter_->get_motion(); }

  // ===== 预测输出（与 Target 的 aimer 消费语义对齐）=====
  // 预测"拍照时刻 + dt"的相位（连续域，建模坐标系）
  double predict_phase(double dt) const;
  // 预测"拍照时刻 + dt"的打击点（世界系）。dt 含全链路延迟与飞行时间。
  Eigen::Vector3d predict_aim_point(double dt) const;
  // 当前帧（dt=0）打击点
  Eigen::Vector3d aim_point() const { return predict_aim_point(0.0); }
  // 预测 roll（(-pi,pi]；兼容 HW x[5] 调试语义）
  double predict_roll(double dt) const;
  // 当前角速度 v(t) = a*cos(ωt+δ) + b（rad/s，t=拍照时刻）
  double speed_now() const;

  // ===== 调试导出（r7 曲线用）=====
  double fitted_phase_now() const { return predict_phase(0.0); }
  double raw_phase_a() const { return raw_phase_a_; }  // 双候选原始相位（仅一候选时 b 等于 a）
  double raw_phase_b() const { return raw_phase_b_; }
  double radius() const { return radius_; }

  void reset();

  // ===== CandidateTarget 构造（相位计算）=====
  // 由 3D 观测构造候选。plane_normal_world 取符面法向（buff 系 x 轴在 world 的方向）。
  static big_rune::CandidateTarget make_candidate(
    const Eigen::Vector3d & rune_center_world, const Eigen::Vector3d & armor_center_world,
    const Eigen::Vector3d & plane_normal_world, big_rune::Time capture_ns);

private:
  static big_rune::Time to_ns(const std::chrono::steady_clock::time_point & t)
  {
    return t.time_since_epoch().count();
  }

  big_rune::MotionFilterParams cfg_;
  std::unique_ptr<big_rune::LM_IRLS_BigRunePhaseMotionFilter> filter_;

  // 最新观测候选的几何缓存：建模前（追踪队列为空）所有查询接口以此回退，
  // 输出"无提前量的当前点"（与 HW 链路 CONVERGING 阶段行为一致）
  big_rune::CandidateTarget latest_candidate_;
  bool has_candidate_ = false;

  double radius_ = 0.7;   // 符心->靶心半径（EMA 平滑）
  double raw_phase_a_ = 0.0, raw_phase_b_ = 0.0;
  bool model_valid_ = false;  // 运动方程有效性（calculate_motion 返回值）
  int lost_cn_ = 0;
  bool has_target_ = false;
  bool unsolvable_ = true;
};

}  // namespace auto_buff
#endif
