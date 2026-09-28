#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

class FanBlade
{
public:
  cv::Point2f center;               // 扇页中心
  std::vector<cv::Point2f> points;  // 四个点从左上角开始逆时针
  double angle, width, height;
  FanBlade_type type;  // 类型

  explicit FanBlade() = default;

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;  // 按target开始顺时针

  int light_num;

  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();  // 单位：rad
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系, 单位: m

  // ---- HW 合并新增：相机系位姿（前x左y上z）与外参，solver 填充、target 读取 ----
  Eigen::Matrix4d pose_camera = Eigen::Matrix4d::Identity();   // 相机系 4x4 位姿
  Eigen::Matrix3d R_gimbal2world = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();

  // 符中心(blade_xyz_in_world)的来源：true=用模型输出的 R 点射线求交，
  // false=射线求交失败后回退到"扇叶中心 + 0.7m"的几何法（调试用）
  bool r_center_is_measured = false;

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF__TYPE_HPP
