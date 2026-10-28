#include "big_rune/big_rune_tracker.hpp"

#include "tools/logger.hpp"

#include <cmath>

namespace auto_buff
{

// static_assert：steady_clock duration 必须是纳秒，to_ns() 的转换才成立
static_assert(
  std::chrono::steady_clock::period::num == 1 && std::chrono::steady_clock::period::den == 1000000000,
  "BigRuneTracker 假设 steady_clock 的 period 为 1 ns");

big_rune::CandidateTarget BigRuneTracker::make_candidate(
  const Eigen::Vector3d & rune_center_world, const Eigen::Vector3d & armor_center_world,
  const Eigen::Vector3d & plane_normal_world, big_rune::Time capture_ns)
{
  big_rune::CandidateTarget c;
  c.rune_center = rune_center_world;
  c.armor_module_center = armor_center_world;
  c.capture_timestamp = capture_ns;

  // 符面法向：归一化并统一朝向
  Eigen::Vector3d n = plane_normal_world.normalized();
  // RP 原版用 normal·armor_center 判断法向朝向（其车系原点位于符轴延长线上，等价于
  // "法向指向车体/观察者"）。Adam world 原点在云台，同样近似位于符轴上，
  // 故等价改用 normal·rune_center。统一朝向是相位帧间连续化的前提。
  if (n.dot(rune_center_world) < 0.0) n = -n;

  // 相位零方向：RP 车系 y 向下故取 up=(0,-1,0)；Adam world z 向上，取 up=(0,0,1)。
  // up 只决定相位常偏（连续化免疫、拟合的 C 自由参数吸收），不影响顺逆时针判断。
  // n×up 在 n 接近竖直时退化为零向量（IMU 异常姿态），换用水平轴兜底防 NaN 污染窗口。
  const Eigen::Vector3d up(0.0, 0.0, 1.0);
  Eigen::Vector3d start = n.cross(up);
  if (start.squaredNorm() < 1e-6) start = n.cross(Eigen::Vector3d::UnitX());
  start.normalize();

  // 靶心相对符心的平面内方向
  const Eigen::Vector3d dir = (armor_center_world - rune_center_world).normalized();

  // 相位（RP PhaseMotionEstimator::complete_candidate_target 逐行对应）
  const double sin_theta = n.dot(start.cross(dir));
  const double cos_theta = start.dot(dir);
  c.phase = std::atan2(sin_theta, cos_theta);
  c.start_vector = start;
  c.rune_plane_world_normal = n;
  return c;
}

BigRuneTracker::BigRuneTracker(const big_rune::MotionFilterParams & cfg)
: cfg_(cfg), filter_(std::make_unique<big_rune::LM_IRLS_BigRunePhaseMotionFilter>(cfg))
{
}

void BigRuneTracker::update(const PowerRune & pr, std::chrono::steady_clock::time_point capture_time)
{
  const big_rune::Time capture_ns = to_ns(capture_time);

  // ---- 构造主候选 ----
  // R 心（符中心）world：solver 存于 blade_xyz_in_world（历史命名，语义为 R 心）
  const Eigen::Vector3d & rune_center_world = pr.blade_xyz_in_world;
  // 靶心（扇叶面板中心）world：solver 存于 xyz_in_world
  const Eigen::Vector3d & armor_center_world = pr.xyz_in_world;
  // 符面法向 world：buff 系 x 轴（HW BUFF_POINTS 全在 buff 系 x=0 平面）
  const Eigen::Matrix3d R_buff2world =
    pr.R_gimbal2world * pr.R_fix * pr.pose_camera.block<3, 3>(0, 0);
  const Eigen::Vector3d plane_normal_world = R_buff2world.col(0);

  big_rune::CandidateTarget main =
    make_candidate(rune_center_world, armor_center_world, plane_normal_world, capture_ns);
  raw_phase_a_ = main.phase;

  // ---- 双候选入队 ----
  std::vector<big_rune::CandidateTarget> candidates;
  candidates.reserve(1 + pr.extra_armor_centers_in_world.size());
  // 先拷贝一份作为"最新候选缓存"（main 本体要移入队列）
  big_rune::CandidateTarget main_storage = main;
  candidates.emplace_back(std::move(main));
  // 副候选：复用主候选的符心/法向（同一符、同一平面），靶心用 solver 求交结果
  for (const auto & c3d : pr.extra_armor_centers_in_world) {
    big_rune::CandidateTarget c =
      make_candidate(rune_center_world, c3d, plane_normal_world, capture_ns);
    raw_phase_b_ = c.phase;
    candidates.emplace_back(std::move(c));
  }
  if (pr.extra_armor_centers_in_world.empty()) raw_phase_b_ = raw_phase_a_;

  // calculate_motion 的返回值即 m_is_motion_model_vaild（唯一可靠的有效性来源：
  // prepare_date 每次拟合都会无条件刷新 reference_timestamp，不能用 != 0 判断）
  model_valid_ = filter_->calculate_motion(std::move(candidates));
  has_target_ = true;
  latest_candidate_ = std::move(main_storage);
  has_candidate_ = true;

  // 半径 EMA：PnP 距离噪声大，标量平滑即可
  const double r_now = (armor_center_world - rune_center_world).norm();
  if (r_now > 0.3 && r_now < 1.2) {
    radius_ = 0.95 * radius_ + 0.05 * r_now;
  }

  lost_cn_ = 0;
  unsolvable_ = false;
}

void BigRuneTracker::update_lost()
{
  lost_cn_++;
  if (lost_cn_ > 15) {
    // 与旧 BigTarget 相同的丢帧重置策略：彻底重建，从建模阶段重新开始
    reset();
  }
}

double BigRuneTracker::phase() const
{
  // 追踪队列在建模前（数据量 < min_data_size）为空，get_target() 不可用
  return filter_->has_tracked_target() ? filter_->get_target().phase : latest_candidate_.phase;
}

double BigRuneTracker::continuous_phase() const
{
  return filter_->has_tracked_target() ? filter_->get_target().continuous_phase
                                       : latest_candidate_.phase;
}

double BigRuneTracker::predict_phase(double dt) const
{
  if (!model_valid_ || !filter_->has_tracked_target()) {
    // 模型未建立：外推不可靠，返回当前观测相位（等效于"无提前量的当前点"，
    // 与 HW 链路 CONVERGING 阶段的行为一致，aimer 至少能瞄住当前扇叶）
    return latest_candidate_.phase;
  }

  const auto & m = filter_->get_motion();
  // t 相对建模参考时刻（秒）。dt 从"拍照时刻"起算，天然包含拍照->当前->飞行全链路延迟
  const auto & tgt = filter_->get_target();
  const double t =
    static_cast<double>(
      tgt.capture_timestamp + static_cast<big_rune::Time>(dt * 1e9) - m.reference_timestamp) *
    1e-9;
  const double omega_t = m.speed_angular_frequency * t;
  return m.phase_cos_coefficient * std::cos(omega_t) + m.phase_sin_coefficient * std::sin(omega_t) +
         m.phase_linear_velocity * t + m.phase_constant_offset_radians;
}

Eigen::Vector3d BigRuneTracker::predict_aim_point(double dt) const
{
  if (!has_candidate_) return Eigen::Vector3d::Zero();

  // 回退路径：建模前（无追踪目标）直接输出最新观测的靶心，无提前量
  if (!filter_->has_tracked_target()) return latest_candidate_.armor_module_center;

  const auto & tgt = filter_->get_target();

  // 模型相位是连续域（相对 root 帧 start 系）。dir 重建必须用与 start 严格配对的
  // 原始相位坐标系：减去追踪帧的连续化累积偏移，换回 back 帧 start 系。
  const double offset = tgt.continuous_phase - tgt.phase;
  const double phase_back = predict_phase(dt) - offset;

  // Rodrigues 平面内简化式（RP RuneDecisionModule::calculate_armor_module_center）：
  //   dir = start·cos(phase) + (n×start)·sin(phase)
  const Eigen::Vector3d ns = tgt.rune_plane_world_normal.cross(tgt.start_vector);
  const Eigen::Vector3d dir =
    tgt.start_vector * std::cos(phase_back) + ns * std::sin(phase_back);

  return tgt.rune_center + dir * radius_;
}

double BigRuneTracker::predict_roll(double dt) const
{
  if (!has_candidate_) return 0.0;
  return big_rune::normalize_phase<big_rune::rad>(predict_phase(dt));
}

double BigRuneTracker::speed_now() const
{
  const auto & m = filter_->get_motion();
  if (!model_valid_ || !filter_->has_tracked_target()) return 0.0;
  const auto & tgt = filter_->get_target();
  const double t = static_cast<double>(tgt.capture_timestamp - m.reference_timestamp) * 1e-9;
  const double omega_t = m.speed_angular_frequency * t;
  return m.speed_amplitude * std::cos(omega_t + m.speed_phase_shift) + m.phase_linear_velocity;
}

void BigRuneTracker::reset()
{
  filter_ = std::make_unique<big_rune::LM_IRLS_BigRunePhaseMotionFilter>(cfg_);
  radius_ = 0.7;
  model_valid_ = false;
  lost_cn_ = 0;
  has_target_ = false;
  has_candidate_ = false;
  unsolvable_ = true;
  raw_phase_a_ = raw_phase_b_ = 0.0;
}

}  // namespace auto_buff
