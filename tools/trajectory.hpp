#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace tools
{
constexpr int max_iter = 10;
constexpr int R_K_iter = 60;
constexpr double k_small = 0.01903;
constexpr double stop_error = 0.001;
constexpr double g = 9.781;

struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正
  double pitch_angle;
  std::vector<std::pair<double, double>> points;

  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  // offsets 预留偏移量
  Trajectory(
    const double v0, const double d, const double h,
    const Eigen::Vector3d & offsets = Eigen::Vector3d::Zero());
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
