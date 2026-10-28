#ifndef AUTO_BUFF__BIG_RUNE__TYPES_HPP
#define AUTO_BUFF__BIG_RUNE__TYPES_HPP

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <vector>

// big_rune：迁移自 RP-26Rune 的大符相位运动估计模块。
// 本头文件是模块公共类型层，替代 RP 的三处依赖：
//   1. foxglove::Time (flatbuffers)   -> big_rune::Time (int64 纳秒)
//   2. function::timestampMinus       -> big_rune::minus_ms
//   3. PRF::calculate_delta_phase 等  -> big_rune 命名空间内同构模板（语义逐行保留）

namespace big_rune
{

// ================= 时间类型 =================
// RP 原版 foxglove::Time{sec:uint32, nsec:uint32}（UTC epoch），这里拍平成 int64 纳秒。
// 与 Adam 侧 steady_clock 时间戳的 ns 表示互相转换（见 BigRuneTracker 门面）。
using Time = int64_t;

// a - b，返回毫秒（与 RP function::timestampMinus 语义一致，调用处 *0.001 换算秒）
inline double minus_ms(Time a, Time b) { return static_cast<double>(a - b) * 1e-6; }

inline constexpr double kTwoPi = 6.283185307179586476925286766559;  // 替代 CV_2PI

// ================= 相位工具（迁移自 RP power_rune_function.hpp，语义逐行保留）=================
struct rad
{
  static constexpr double period = kTwoPi;
};
struct deg
{
  static constexpr double period = 360.0;
};

// 差角归位到 (-period/2, period/2]
template <typename Unit>
inline double calculate_delta_phase(double new_phase, double old_phase)
{
  constexpr double period = Unit::period;
  constexpr double half_period = period * 0.5;

  double delta_phase = new_phase - old_phase;
  while (delta_phase > half_period) delta_phase -= period;
  while (delta_phase < -half_period) delta_phase += period;
  return delta_phase;
}

// 归位到 (-period/2, period/2]
template <typename Unit>
inline double normalize_phase(double phase)
{
  constexpr double period = Unit::period;
  constexpr double half_period = period * 0.5;

  phase = std::fmod(phase + half_period, period);
  if (phase < 0) phase += period;
  return phase - half_period;
}

// ================= 数据结构（迁移自 RP BigRuneMotionEstimate.hpp）=================
enum class RotationDirection { clockwise, anticlockwise, Unknown };

// 单帧单候选：solver 输出的一个扇叶观测（世界系）
struct CandidateTarget
{
  Eigen::Vector3d rune_center = Eigen::Vector3d::Zero();            // 符中心（R 字）位置
  Eigen::Vector3d armor_module_center = Eigen::Vector3d::Zero();    // 靶心（装甲板中心）位置
  double phase = 0.0;                                               // 靶心平面内方位角（rad）
  Eigen::Vector3d start_vector = Eigen::Vector3d::Zero();           // 相位零方向单位向量
  Eigen::Vector3d rune_plane_world_normal = Eigen::Vector3d::Zero();  // 符平面法向（单位）
  Time capture_timestamp = 0;                                       // 拍照时刻
};

// 追踪目标 = 候选 + 连续化相位
struct TrackedTarget : public CandidateTarget
{
  TrackedTarget() = default;
  explicit TrackedTarget(const CandidateTarget & candidate) : CandidateTarget(candidate)
  {
    // 初始连续相位：取 (0, 2π] 表示
    continuous_phase = phase > 0 ? phase : phase + kTwoPi;
  }

  int switch_num = 0;                    // 累计目标切换数（每次跳变的扇叶格数）
  double continuous_phase = 0.0;         // 连续化相位（不取模，可无限增长）
  double decentralized_timestamp = 0.0;  // 去中心化时间戳（拟合准备阶段填充，秒）
};

// 运动方程参数：phase(t) = A*cos(ωt) + B*sin(ωt) + b*t + C
// 物理转速 v(t) = a*cos(ωt + δ) + b，其中 a = ω*sqrt(A²+B²)，δ = atan2(A, B)
struct MotionModelParams
{
  Time reference_timestamp = 0;  // 方程参考时刻（建模型数据的中位时间戳）
  double phase_cos_coefficient = 0.0;           // A
  double phase_sin_coefficient = 0.0;           // B
  double phase_linear_velocity = 0.0;           // b（基础角速度）
  double phase_constant_offset_radians = 0.0;   // C
  double speed_angular_frequency = 0.0;         // ω
  double speed_amplitude = 0.0;                 // a
  double speed_phase_shift = 0.0;               // δ
};

// ================= 滤波器参数（迁移自 RP config/power_rune.json 的 big_phase_motion_estimate 块）=================
// 键名与 RP 完全一致，默认值即 RP 的线上值；from_yaml 时可整体覆盖。
struct MotionFilterParams
{
  // 队列 / 窗口
  double max_time_interval = 3.0;                 // 相邻帧最大间隔（秒），超过则重置方程
  double prepare_time = 180.0;                    // 间隔超过此值（秒）连旋转方向一起重置
  double window_settling_time = 8.0;              // 原始候选队列滑窗时长（秒）
  int min_data_size_to_build_motion_model = 100;  // 建模所需最少帧数
  double acceptable_time_interval_to_confirm_rotation = 0.15;  // 判旋转方向的最大帧间隔（秒）
  double target_switch_threshold = 0.77;          // 相位跳变判定阈值（rad）
  double expect_phase_linear_velocity = 1.1775;   // 建模前猜相位用的线速度（rad/s）
  double highest_time_weight = 1.0;               // 时间权重上限
  double lowwest_time_weight = 0.2;               // 时间权重下限
  int filter_window_size = 7;                     // 模型滤波窗口（奇数）

  // 拟合初值与物理域
  double speed_angular_frequency_init = 1.942;        // ω 初值
  double phase_cos_coefficient_init = -0.3322528;     // A 初值
  double phase_sin_coefficient_init = 0.3322528;      // B 初值
  double phase_linear_velocity_init = 1.1775;         // b 初值
  double phase_constant_offset_radians_init = 0.0;    // C 初值
  double omega_lower_bound = 1.884;                   // ω 下界（RM 规则 0.2 rev/s）
  double omega_upper_bound = 2.000;                   // ω 上界

  // LM 外层（迭代 ω）
  int outer_max_iterations = 20;
  double outer_tolerance = 1e-5;
  double lambda_init = 1e-2;
  double lambda_min = 1e-6;
  double lambda_max = 1e6;
  double max_omega_step = 0.05;
  double diff_eps_min = 1e-4;
  double diff_eps_rel_init = 1e-3;

  // IRLS 内层（迭代 A,B,b,C）
  int inner_max_iterations = 40;
  double robust_scale_factor = 2.5;
  double inner_tolerance = 1e-2;

  // 从 yaml 节点读取（节点即 hw_buff.big_phase_motion_estimate，键名与 RP json 一致）
  static MotionFilterParams from_yaml(const YAML::Node & n)
  {
    MotionFilterParams p;
    p.max_time_interval = n["max_time_interval"].as<double>(p.max_time_interval);
    p.prepare_time = n["prepare_time"].as<double>(p.prepare_time);
    p.window_settling_time = n["window_settling_time"].as<double>(p.window_settling_time);
    p.min_data_size_to_build_motion_model =
      n["min_data_size_to_build_motion_model"].as<int>(p.min_data_size_to_build_motion_model);
    p.acceptable_time_interval_to_confirm_rotation =
      n["acceptable_time_interval_to_confirm_rotation"].as<double>(
        p.acceptable_time_interval_to_confirm_rotation);
    p.target_switch_threshold = n["target_switch_threshold"].as<double>(p.target_switch_threshold);
    p.expect_phase_linear_velocity =
      n["expect_phase_linear_velocity"].as<double>(p.expect_phase_linear_velocity);
    p.highest_time_weight = n["highest_time_weight"].as<double>(p.highest_time_weight);
    p.lowwest_time_weight = n["lowwest_time_weight"].as<double>(p.lowwest_time_weight);
    p.filter_window_size = n["filter_window_size"].as<int>(p.filter_window_size);

    p.speed_angular_frequency_init =
      n["speed_angular_frequency_init"].as<double>(p.speed_angular_frequency_init);
    p.phase_cos_coefficient_init =
      n["phase_cos_coefficient_init"].as<double>(p.phase_cos_coefficient_init);
    p.phase_sin_coefficient_init =
      n["phase_sin_coefficient_init"].as<double>(p.phase_sin_coefficient_init);
    p.phase_linear_velocity_init =
      n["phase_linear_velocity_init"].as<double>(p.phase_linear_velocity_init);
    p.phase_constant_offset_radians_init =
      n["phase_constant_offset_radians_init"].as<double>(p.phase_constant_offset_radians_init);
    p.omega_lower_bound = n["omega_lower_bound"].as<double>(p.omega_lower_bound);
    p.omega_upper_bound = n["omega_upper_bound"].as<double>(p.omega_upper_bound);

    p.outer_max_iterations = n["outer_max_iterations"].as<int>(p.outer_max_iterations);
    p.outer_tolerance = n["outer_tolerance"].as<double>(p.outer_tolerance);
    p.lambda_init = n["lambda_init"].as<double>(p.lambda_init);
    p.lambda_min = n["lambda_min"].as<double>(p.lambda_min);
    p.lambda_max = n["lambda_max"].as<double>(p.lambda_max);
    p.max_omega_step = n["max_omega_step"].as<double>(p.max_omega_step);
    p.diff_eps_min = n["diff_eps_min"].as<double>(p.diff_eps_min);
    p.diff_eps_rel_init = n["diff_eps_rel_init"].as<double>(p.diff_eps_rel_init);

    p.inner_max_iterations = n["inner_max_iterations"].as<int>(p.inner_max_iterations);
    p.robust_scale_factor = n["robust_scale_factor"].as<double>(p.robust_scale_factor);
    p.inner_tolerance = n["inner_tolerance"].as<double>(p.inner_tolerance);
    return p;
  }
};

}  // namespace big_rune

#endif
