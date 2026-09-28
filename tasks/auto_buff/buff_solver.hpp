#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <optional>

#include "buff_algo/locate/buff_pnp_solver.hpp"
#include "buff_type.hpp"
#include "tools/math_tools.hpp"
namespace auto_buff
{
// 旋转角度
const double THETA = 2.0 * CV_PI / 5.0;  // 2/5π

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  void solve(std::optional<PowerRune> & ps) const;

  // 调试用
  cv::Point2f point_buff2pixel(cv::Point3f x);

  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  cv::Vec3d rvec_, tvec_;

  // HW BuffPnPSolver：IPPE 平面四点 PnP，模型尺寸 BUFF_WIDTH=0.114m
  // mutable：solve() 为 const，而 solve_pnp 内部会更新状态
  mutable buff_algo::BuffPnPSolver pnp_solver_;

  // 调试重投影用的 4 角点物体坐标（前x左y上z，上/左/下/右）
  const std::vector<cv::Point3f> OBJECT_POINTS = {
    cv::Point3f(0, 0, 114e-3), cv::Point3f(0, 114e-3, 0),
    cv::Point3f(0, 0, -114e-3), cv::Point3f(0, -114e-3, 0)};  // 单位：米

  // 函数：生成绕x轴旋转的旋转矩阵
  cv::Matx33f rotation_matrix(double angle) const;

  // 函数：旋转点并填充到 OBJECT_POINTS 中
  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);

  // 用模型直接输出的 R 字中心像素点，与扇叶所在平面求交，得到相机系 3D 符中心。
  // 失败（视线与平面近平行 / 交点明显不合理）返回 false，调用方回退到几何法。
  bool r_center_by_ray(
    const cv::Point2f & r_center_px, const Eigen::Matrix3d & R_buff2cam,
    const Eigen::Vector3d & t_buff2cam, Eigen::Vector3d & r_center_cam) const;
};
}  // namespace auto_buff
#endif  // AUTO_AIM__SOLVER_HPP
