// LM+IRLS 大符相位运动拟合实现（迁移自 RP-26Rune LM_IRLS_BigRunePhaseMotionFilter.cpp）
// 外层 LM 数值梯度迭代 ω（带边界钳制与步长限制），内层 IRLS（Cauchy 鲁棒核 + ridge 正则）
// 迭代 A,B,b,C。算法与 RP 原版逐行对应，仅替换参数读取与日志。

#include "big_rune/LM_IRLS_BigRunePhaseMotionFilter.hpp"

#include "tools/logger.hpp"

#include <algorithm>
#include <cmath>

namespace big_rune
{

void LM_IRLS_BigRunePhaseMotionFilter::prepare_date()
{
  // 清除旧的数据
  m_t.setZero();
  m_t.resize(m_tracked_target_deque.size());
  m_y.setZero();
  m_y.resize(m_tracked_target_deque.size());
  m_time_weight.setZero();
  m_time_weight.resize(m_tracked_target_deque.size());
  m_residual_weight.resize(m_tracked_target_deque.size());
  m_residual_weight.setOnes();

  // 将内层的拟合结果重置为无效
  m_fit_result_inside.cost = std::numeric_limits<double>::infinity();
  m_fit_result_inside.is_valid = false;

  // 由于可以认为间隔是相对稳定的，所以取中值时间戳代替平均值时间戳
  const Time reference_timestamp =
    m_tracked_target_deque[static_cast<int>(m_tracked_target_deque.size() * 0.5 + 1)]
      .capture_timestamp;

  // 时间权重与去中心化时间
  const double highest_time_weight = cfg_.highest_time_weight;
  const double lowwest_time_weight = cfg_.lowwest_time_weight;
  const double time_weight_gap = highest_time_weight - lowwest_time_weight;
  const double max_time_interval = minus_ms(
    m_tracked_target_deque.back().capture_timestamp,
    m_tracked_target_deque.front().capture_timestamp);
  for (int i = 0; i < static_cast<int>(m_tracked_target_deque.size()); i++) {
    m_tracked_target_deque[i].decentralized_timestamp =
      minus_ms(m_tracked_target_deque[i].capture_timestamp, reference_timestamp) * 0.001;

    m_t(i) = m_tracked_target_deque[i].decentralized_timestamp;
    m_y(i) = m_tracked_target_deque[i].continuous_phase;
    m_time_weight(i) =
      lowwest_time_weight +
      minus_ms(
        m_tracked_target_deque[i].capture_timestamp,
        m_tracked_target_deque.front().capture_timestamp) /
        max_time_interval * time_weight_gap;
  }
  m_big_rune_motion_model.reference_timestamp = reference_timestamp;
}

void LM_IRLS_BigRunePhaseMotionFilter::fit_motion_model()
{
  // 拟合大符的运动方程phase = A*cos(ωt) + B*sin(ωt) + b*t + C或-phase = A*cos(ωt) + B*sin(ωt) + b*t + C
  // 拟合方法是LM + 迭代加权最小二乘算法
  // 外部循环迭代ω
  // 内部迭代A,B,b,C
  // 外层用LM算法迭代ω，内层用迭代加权最小二乘拟合A,B,b,C
  // 1. 初始化ω,A,B,b,C
  // 2. 按照时间为参数分配权重
  // 3. 内层循环第一次迭代。仅有时间加权。迭代结果与测量值作差，得到距离权重。
  // 4. 根据距离权重迭代和时间权重进行迭代，迭代后重新计算距离权重,继续迭代直至达到内层的终止条件
  // 5. 外层采用lm算法进行迭代

  // 如果之前没有拟合，需要初始化参数
  if (!m_is_motion_model_vaild) {
    m_big_rune_motion_model.speed_angular_frequency = cfg_.speed_angular_frequency_init;
    m_big_rune_motion_model.phase_cos_coefficient = cfg_.phase_cos_coefficient_init;
    m_big_rune_motion_model.phase_sin_coefficient = cfg_.phase_sin_coefficient_init;
    m_big_rune_motion_model.phase_linear_velocity = cfg_.phase_linear_velocity_init;
    m_big_rune_motion_model.phase_constant_offset_radians =
      cfg_.phase_constant_offset_radians_init;

    if (m_rotation_direction == RotationDirection::anticlockwise) {
      // 取反
      m_big_rune_motion_model.phase_cos_coefficient =
        -m_big_rune_motion_model.phase_cos_coefficient;
      m_big_rune_motion_model.phase_sin_coefficient =
        -m_big_rune_motion_model.phase_sin_coefficient;
      m_big_rune_motion_model.phase_linear_velocity = -m_big_rune_motion_model.phase_linear_velocity;
      m_big_rune_motion_model.phase_constant_offset_radians =
        -m_big_rune_motion_model.phase_constant_offset_radians;
    }
    m_fit_result_inside.theta << m_big_rune_motion_model.phase_cos_coefficient,
      m_big_rune_motion_model.phase_sin_coefficient, m_big_rune_motion_model.phase_linear_velocity,
      m_big_rune_motion_model.phase_constant_offset_radians;
  }

  // 准备数据
  prepare_date();

  // 用于限制omega的范围
  auto clamp_omega = [&](double omega) -> double {
    if (omega < cfg_.omega_lower_bound) {
      return cfg_.omega_lower_bound;
    }
    if (omega > cfg_.omega_upper_bound) {
      return cfg_.omega_upper_bound;
    }
    return omega;
  };

  // 获取某个 ω 下的最优线性参数与代价并判断内层优化是否成功
  auto eval_omega = [&](double omega_in, Eigen::Vector4d & theta_out, double & cost_out) -> bool {
    fit_linear(omega_in);
    if (!m_fit_result_inside.is_valid) {
      // 内层优化失败
      return false;
    }
    theta_out = m_fit_result_inside.theta;
    cost_out = m_fit_result_inside.cost;
    return std::isfinite(cost_out) && theta_out.allFinite();
  };

  const int outer_max_iters = cfg_.outer_max_iterations;
  const double omega_tolerance = cfg_.outer_tolerance;
  const double lambda_init = cfg_.lambda_init;
  const double lambda_min = cfg_.lambda_min;
  const double lambda_max = cfg_.lambda_max;
  const double max_omega_step = cfg_.max_omega_step;
  const double diff_eps_min = cfg_.diff_eps_min;
  const double diff_eps_rel = cfg_.diff_eps_rel_init;

  double omega = m_big_rune_motion_model.speed_angular_frequency;
  if (!m_is_motion_model_vaild || !std::isfinite(omega)) {
    omega = cfg_.speed_angular_frequency_init;
  }
  omega = clamp_omega(omega);

  Eigen::Vector4d best_theta = Eigen::Vector4d::Zero();
  double best_cost = std::numeric_limits<double>::infinity();
  double best_omega = omega;
  if (!eval_omega(omega, best_theta, best_cost)) {
    // 优化失败
    m_is_motion_model_vaild = false;
    return;
  }

  double lambda = std::max(lambda_min, lambda_init);
  for (int iter = 0; iter < outer_max_iters; iter++) {
    const double eps = std::max(diff_eps_min, diff_eps_rel * std::fabs(omega));
    const double omega_p = clamp_omega(omega + eps);
    const double omega_m = clamp_omega(omega - eps);
    if (omega_p == omega_m) {
      tools::logger()->warn("[LM_IRLS_BigRunePhaseMotionFilter]边界值");
      break;
    }

    Eigen::Vector4d theta_p;
    Eigen::Vector4d theta_m;
    double cost_p = 0.0;
    double cost_m = 0.0;
    if (!eval_omega(omega_p, theta_p, cost_p) || !eval_omega(omega_m, theta_m, cost_m)) {
      lambda = std::min(lambda_max, lambda * 2.0);
      continue;
    }

    const double grad = (cost_p - cost_m) / (omega_p - omega_m);
    const double half_span = 0.5 * (omega_p - omega_m);
    double hess = (cost_p - 2.0 * best_cost + cost_m) / (half_span * half_span);
    if (!std::isfinite(grad) || !std::isfinite(hess)) {
      // 无效值，优化失败
      m_is_motion_model_vaild = false;
      return;
    }

    if (hess < 0.0) {
      hess = -hess;
    }
    const double raw_step = -grad / (hess + lambda);
    const double step = std::clamp(raw_step, -max_omega_step, max_omega_step);
    const double omega_candidate = clamp_omega(omega + step);

    if (std::fabs(omega_candidate - omega) < omega_tolerance) {
      break;
    }

    Eigen::Vector4d theta_c;
    double cost_c = 0.0;
    if (eval_omega(omega_candidate, theta_c, cost_c) && cost_c < best_cost) {
      omega = omega_candidate;
      best_omega = omega_candidate;
      best_theta = theta_c;
      best_cost = cost_c;
      lambda = std::max(lambda_min, lambda * 0.5);

      if (std::fabs(step) < omega_tolerance) {
        break;
      }
    } else {
      lambda = std::min(lambda_max, lambda * 2.0);
    }
  }

  complete_motion(best_theta, best_omega);
}

void LM_IRLS_BigRunePhaseMotionFilter::fit_linear(const double & omega)
{
  // 构建设计矩阵 X=[cos(ωt), sin(ωt), t, 1]
  Eigen::MatrixXd X(m_tracked_target_deque.size(), 4);
  for (int i = 0; i < static_cast<int>(m_tracked_target_deque.size()); ++i) {
    const double wt = omega * m_t(i);
    X(i, 0) = std::cos(wt);
    X(i, 1) = std::sin(wt);
    X(i, 2) = m_t(i);
    X(i, 3) = 1.0;
  }

  // 重置残差权重
  m_residual_weight.setOnes();

  // 是否进行过至少一次拟合
  bool is_solved_at_least_one_time = false;

  // 迭代循环
  for (int iter = 0; iter < cfg_.inner_max_iterations; iter++) {
    // 更新和权重
    m_total_weight = m_time_weight.cwiseProduct(m_residual_weight);

    // 构造正规方程
    // 加权最小二乘: (X^T W X + ridge I)theta = X^T W y
    Eigen::Matrix4d H = X.transpose() * m_total_weight.asDiagonal() * X;
    H += m_ridge;
    Eigen::Vector4d g = X.transpose() * m_total_weight.asDiagonal() * m_y;
    if (!H.allFinite() || !g.allFinite()) {
      return;
    }

    Eigen::LDLT<Eigen::Matrix4d> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
      return;
    }

    // 求解参数
    Eigen::Vector4d theta_new = ldlt.solve(g);
    if (!theta_new.allFinite()) {
      return;
    }

    // 计算残差和总代价
    const Eigen::VectorXd residual = m_y - X * theta_new;
    const double cost = (m_total_weight.array() * residual.array().square()).sum();
    if (!std::isfinite(cost)) {
      return;
    }

    // 更新距离权重
    const double rms = std::sqrt(std::max(
      1e-12,
      (m_time_weight.array() * residual.array().square()).sum() /
        static_cast<double>(m_tracked_target_deque.size())));
    const double cauchy_scale =
      std::max(1e-6, cfg_.robust_scale_factor * rms);
    for (int i = 0; i < static_cast<int>(m_tracked_target_deque.size()); i++) {
      const double ratio = residual(i) / cauchy_scale;
      m_residual_weight(i) = 1.0 / (1.0 + ratio * ratio);
    }

    // 下面开始为收敛性判断
    if (!is_solved_at_least_one_time) {
      // 说明为第一次迭代，必然不允许判断为收敛
      m_fit_result_inside.theta = theta_new;
      m_fit_result_inside.cost = cost;
      is_solved_at_least_one_time = true;
      continue;
    }
    if (
      (m_fit_result_inside.theta - theta_new).norm() < cfg_.inner_tolerance ||
      std::fabs(m_fit_result_inside.cost - cost) < cfg_.inner_tolerance) {
      m_fit_result_inside.theta = theta_new;
      m_fit_result_inside.cost = cost;
      m_fit_result_inside.is_valid = true;
      return;
    }

    // 未收敛时也要把当前迭代结果保存为下一轮比较基准。
    m_fit_result_inside.theta = theta_new;
    m_fit_result_inside.cost = cost;
  }

  // 运行到此处说明达到了迭代上限却没有满足收敛条件
  tools::logger()->warn("[LM_IRLS_BigRunePhaseMotionFilter]内层达到迭代上限,求解失败");
}

void LM_IRLS_BigRunePhaseMotionFilter::complete_motion(
  const Eigen::Vector4d & theta, const double & omega)
{
  const double A = theta(0);  // A
  const double B = theta(1);  // B
  const double b = theta(2);  // b
  const double C = theta(3);  // C
  const double speed_amplitude = omega * std::sqrt(A * A + B * B);  // a = ω * sqrt(A^2 + B^2)
  double speed_phase_shift = std::atan2(B, -A);                     // δ = atan2(B, -A)

  // 安全检查
  if (
    !std::isfinite(A) || !std::isfinite(B) || !std::isfinite(b) || !std::isfinite(C) ||
    !std::isfinite(omega) || !std::isfinite(speed_amplitude) || !std::isfinite(speed_phase_shift)) {
    m_is_motion_model_vaild = false;
    return;
  }
  if (speed_phase_shift < 0.0) {
    speed_phase_shift += kTwoPi;
  }

  // 赋值
  m_big_rune_motion_model.phase_cos_coefficient = A;
  m_big_rune_motion_model.phase_sin_coefficient = B;
  m_big_rune_motion_model.phase_linear_velocity = b;
  m_big_rune_motion_model.phase_constant_offset_radians = C;
  m_big_rune_motion_model.speed_angular_frequency = omega;
  m_big_rune_motion_model.speed_amplitude = speed_amplitude;
  m_big_rune_motion_model.speed_phase_shift = speed_phase_shift;
  m_is_motion_model_vaild = true;
}

}  // namespace big_rune
