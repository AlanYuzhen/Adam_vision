#include "trajectory.hpp"

#include <cmath>

namespace tools
{
Trajectory::Trajectory(
  const double v0, const double d, const double h, const Eigen::Vector3d & offsets)
{
  (void)offsets;
  const double k = k_small;
  points.reserve(static_cast<size_t>(R_K_iter) * static_cast<size_t>(max_iter));

  unsolvable = false;//只要在相机可识别范围内理论全部有解
  const double x_offset = 0.0;//暂未使用，假设轨迹理想
  const double y_offset = 0.0;
  const double z_offset = 0.0;
  const double new_shoot_speed = v0;
  const double high = h;
  double offseted_high = high + z_offset;
  const double distance_x = d;
  const double pitch0 = std::atan2(high, distance_x) * 180.0 / M_PI;
  auto pitch_new = pitch0;
  fly_time = 0.0;

  for (int i = 0; i < max_iter; ++i) {
    points.clear();
    double x = x_offset;
    double y = y_offset;
    const double pitch_rad = pitch_new * M_PI / 180.0;
    double p = std::tan(pitch_rad);
    const double v = new_shoot_speed;
    const double p2 = p * p;
    double u = v / std::sqrt(1.0 + p2);
    const double delta_x = distance_x / static_cast<double>(R_K_iter);
    double iter_fly_time = 0.0;

    for (int j = 0; j < R_K_iter; ++j) {
      const double sqrt_1_p2 = std::sqrt(1.0 + p * p);
      const double u2 = u * u;
      const double half_dx = delta_x * 0.5;
      const double k1_u = -k * u * sqrt_1_p2;
      const double k1_p = -g / u2;
      const double k1_u_sum = u + k1_u * half_dx;
      const double k1_p_sum = p + k1_p * half_dx;

      const double sqrt_1_p2_k2 = std::sqrt(1.0 + k1_p_sum * k1_p_sum);
      const double k2_u = -k * k1_u_sum * sqrt_1_p2_k2;
      const double k2_p = -g / (k1_u_sum * k1_u_sum);
      const double k2_u_sum = u + k2_u * half_dx;
      const double k2_p_sum = p + k2_p * half_dx;

      const double sqrt_1_p2_k3 = std::sqrt(1.0 + k2_p_sum * k2_p_sum);
      const double k3_u = -k * k2_u_sum * sqrt_1_p2_k3;
      const double k3_p = -g / (k2_u_sum * k2_u_sum);
      const double k3_u_sum = u + k3_u * half_dx;
      const double k3_p_sum = p + k3_p * half_dx;

      const double sqrt_1_p2_k4 = std::sqrt(1.0 + k3_p_sum * k3_p_sum);
      const double k4_u = -k * k3_u_sum * sqrt_1_p2_k4;
      const double k4_p = -g / (k3_u_sum * k3_u_sum);

      const double sixth_dx = delta_x / 6.0;
      u += sixth_dx * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u);
      p += sixth_dx * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);

      x += delta_x;
      y += p * delta_x;
      iter_fly_time += delta_x / u;
      points.emplace_back(x, y);
    }

    const double error = high - y;
    if (std::abs(error) <= stop_error) {
      fly_time = iter_fly_time;
      pitch_angle = pitch_new;
      pitch = pitch_new * M_PI / 180.0;
      return;
    }

    offseted_high += error;
    pitch_new = std::atan2(offseted_high, distance_x) * 180.0 / M_PI;
    fly_time = iter_fly_time;
  }

  pitch_angle = pitch_new;
  pitch = pitch_new * M_PI / 180.0;
}

}  // namespace tools
