#include "buff_solver.hpp"

#include <cmath>

#include "tools/logger.hpp"
namespace auto_buff
{
cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // HW BuffPnPSolver 内部硬编码的 cv_to_tf 就是这个理想置换。
  // 打符链路不再假设"相机轴对齐安装"：cam->gimbal 直接读取 config 标定外参，
  // 修正旋转 R_fix = R_camera2gimbal · Tᵀ。这里仍计算并报告偏差，
  // 用于判断标定/安装是否可靠（偏差越大，越应复核标定）。
  {
    Eigen::Matrix3d R_cv2tf;
    R_cv2tf << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    const double dev_rad = Eigen::AngleAxisd(R_camera2gimbal_ * R_cv2tf.transpose()).angle();
    const double dev_deg = dev_rad * 57.29577951308232;
    if (dev_deg > 1.0) {
      tools::logger()->warn(
        "[BuffSolver] 本车 R_camera2gimbal 与理想轴对齐置换相差 {:.2f} deg；"
        "打符链路将直接读取 config 外参、用 R_fix 做旋转补偿（不再只加平移）。"
        "偏差较大，请复核标定是否有效。",
        dev_deg);
    } else {
      tools::logger()->info(
        "[BuffSolver] R_camera2gimbal 与理想置换相差 {:.2f} deg，打符链路直接使用 config 外参。",
        dev_deg);
    }
  }

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  // HW PnP 需要相机内参
  pnp_solver_.set_camera_matrix(camera_matrix_, distort_coeffs_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();

  // 检测输出 4 角点顺序固定：上/左/下/右（t/l/b/r）
  const auto & pts = p.target().points;
  if (pts.size() < 4) return;

  buff_algo::BuffDetection det;
  det.t = {pts[0].x, pts[0].y};
  det.l = {pts[1].x, pts[1].y};
  det.b = {pts[2].x, pts[2].y};
  det.r = {pts[3].x, pts[3].y};
  det.confidence = 1.0f;

  auto poses = pnp_solver_.solve_pnp({det});
  if (poses.empty()) return;
  const auto & pose = poses[0];  // 相机系（前x左y上z）

  // 存相机系位姿 + 外参，供 target 的 HW tracker 使用
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix().cast<double>();
  T.block<3, 1>(0, 3) = pose.translation.cast<double>();
  p.pose_camera = T;
  p.R_gimbal2world = R_gimbal2world_;
  p.t_camera2gimbal = t_camera2gimbal_;

  // 相机系(前x左y上z) -> 云台系：直接读 config 标定外参。
  // HW 输出已由 PnP 内部 cv_to_tf 变到"前x左y上z"，故修正旋转
  //   R_fix = R_camera2gimbal · Tᵀ
  // 下游所有 cam->gimbal（位置、符中心、姿态）统一乘它。
  Eigen::Matrix3d R_cv2tf;
  R_cv2tf << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  const Eigen::Matrix3d R_fix = R_camera2gimbal_ * R_cv2tf.transpose();
  p.R_fix = R_fix;

  // cam(前x左y上z) -> gimbal：乘真实外参旋转 R_fix，再加平移
  Eigen::Vector3d xyz_in_camera = pose.translation.cast<double>();
  Eigen::Vector3d xyz_in_gimbal = R_fix * xyz_in_camera + t_camera2gimbal_;

  // gimbal -> world
  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  // ---- 符中心（R 标）----
  // 以前是"扇叶中心 + buff 系 z 轴 0.7m"硬算出来的，依赖 0.7m 与扇叶法向两个假设。
  // 现在优先用模型直接输出的 R 字中心像素点：把它反投影成射线，与扇叶所在平面求交。
  //   扇叶平面：过扇叶中心 xyz_in_camera，法向 = buff 系 x 轴（HW 的 BUFF_POINTS 全在 x=0 平面上）
  const Eigen::Matrix3d R_buff2cam = pose.rotation.toRotationMatrix().cast<double>();
  Eigen::Vector3d R_center_cam;
  bool r_center_ok = r_center_by_ray(p.r_center, R_buff2cam, xyz_in_camera, R_center_cam);
  if (r_center_ok) {
    // 符中心到扇叶中心的距离物理上应在 0.7m 量级，超出范围说明求交异常，回退
    const double d = (R_center_cam - xyz_in_camera).norm();
    if (d < 0.3 || d > 1.2) {
      tools::logger()->debug(
        "[BuffSolver] R 中心射线求交结果异常(距扇叶中心 {:.3f} m)，回退几何法", d);
      r_center_ok = false;
    }
  }
  if (!r_center_ok) {
    // 回退：扇叶面板中心沿 buff 系 -z 方向 0.7m（旧实现）
    R_center_cam = R_buff2cam * Eigen::Vector3d(0, 0, -0.7) + xyz_in_camera;
  }
  p.r_center_is_measured = r_center_ok;  // 供调试区分两种来源

  p.blade_xyz_in_world = R_gimbal2world_ * (R_fix * R_center_cam + t_camera2gimbal_);
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  // ypr_in_world：R_buff2world 的欧拉角（cam->gimbal 同样先乘 R_fix）
  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_fix * R_buff2cam;
  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
}

// 反投影模型给出的 R 字中心像素 -> 与扇叶平面求交 -> 相机系(前x左y上z) 3D 点
bool Solver::r_center_by_ray(
  const cv::Point2f & r_center_px, const Eigen::Matrix3d & R_buff2cam,
  const Eigen::Vector3d & t_buff2cam, Eigen::Vector3d & r_center_cam) const
{
  if (camera_matrix_.empty() || !std::isfinite(r_center_px.x) || !std::isfinite(r_center_px.y))
    return false;

  // 去畸变得到归一化像平面坐标 (x/z, y/z)，射线方向在 opencv 相机系为 (x, y, 1)
  std::vector<cv::Point2d> src{cv::Point2d(r_center_px.x, r_center_px.y)}, dst;
  cv::undistortPoints(src, dst, camera_matrix_, distort_coeffs_);
  if (dst.empty()) return false;

  // opencv 相机系(右x,下y,前z) -> 前x左y上z：R = [[0,0,1],[-1,0,0],[0,-1,0]]（即 HW 的 cv_to_tf）
  const Eigen::Vector3d ray(1.0, -dst[0].x, -dst[0].y);

  // 扇叶平面：法向取 buff 系 x 轴
  const Eigen::Vector3d n = R_buff2cam * Eigen::Vector3d::UnitX();
  const double denom = n.dot(ray);
  if (std::fabs(denom) < 1e-6) return false;  // 视线与平面近乎平行，求交不稳定

  const double s = n.dot(t_buff2cam) / denom;
  if (!(s > 0.1 && s < 100.0)) return false;  // 交点在相机后方或远得离谱
  r_center_cam = s * ray;
  return true;
}

// 调试用
cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  // buff坐标系(单位:m)到像素坐标系
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

// xyz_in_world2xyz_in_pix
std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  // get R_buff2camera t_buff2camera
  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
