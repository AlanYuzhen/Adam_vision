#include "target.hpp"

#include <algorithm>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_RADIUS = 0.2765;                // m
constexpr double OUTPOST_ROTATION_SPEED = 0.6 * CV_PI;  // 0.4 r/s
constexpr double OUTPOST_DIRECTION_THRESHOLD = 2.0;     // rad/s
constexpr double OUTPOST_HEIGHT_ERROR_SCALE = 0.10;     // m
constexpr double OUTPOST_HEIGHT_MIN = 0;              // m
constexpr double OUTPOST_HEIGHT_MAX = 0.21;              // m
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, double v1, double v2)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  v1_(v1),
  v2_(v2),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  outpost_rotation_direction_(0),
  switch_count_(0)
{
  const bool is_outpost = name == ArmorName::outpost && armor_num_ == 3;
  auto r = is_outpost ? OUTPOST_RADIUS : radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1; for the outpost, z1 - z0
  // h: z2 - z1; for the outpost, z2 - z0
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  if (name == ArmorName::outpost && armor_num_ == 3) {
    tools::logger()->debug(
      "[Target][Outpost] init: armor 0 anchored at z={:.3f} m; armor 1/2 heights unknown",
      center_z);
  }
  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  enforce_outpost_radius();
}

Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4), outpost_rotation_direction_(0)//初始化状态
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1 = v1_, v2 = v2_;
  if (name == ArmorName::outpost) {
    v1 = 10;  // 前哨站加速度方差
    v2 = (outpost_rotation_direction_ == 0) ? 200 : 0.1;
  }

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (name == ArmorName::outpost && outpost_rotation_direction_ != 0) {
    ekf_.x[7] = outpost_rotation_direction_ * OUTPOST_ROTATION_SPEED;
  }

  ekf_.predict(F, Q, f);
  enforce_outpost_radius();
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id = 0;
  auto min_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 普通车辆只考虑距离相机最近的三块板；前哨站恰好有三块板。
  const int candidate_count = std::min(3, armor_num_);
  for (int i = 0; i < candidate_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    const int candidate_id = xyza_i_list[i].second;
    const auto error = association_error(armor, xyza, candidate_id);

    if (error < min_error) {
      id = candidate_id;
      min_error = error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  initialize_outpost_height(armor, id);
  update_ypda(armor, id);
  enforce_outpost_radius();
  clamp_outpost_height_offsets();
  lock_outpost_rotation_direction();
}

double Target::association_error(
  const Armor & armor, const Eigen::Vector4d & xyza, int id) const
{
  const Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
  auto error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
               std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

  if (
    name == ArmorName::outpost && id >= 0 && id < outpost_height_initialized_.size() &&
    outpost_height_initialized_[id]) {
    error += std::abs(armor.xyz_in_world[2] - xyza[2]) / OUTPOST_HEIGHT_ERROR_SCALE;
  }

  return error;
}

void Target::initialize_outpost_height(const Armor & armor, int id)
{
  if (
    name != ArmorName::outpost || armor_num_ != 3 || id <= 0 ||
    id >= outpost_height_initialized_.size() || outpost_height_initialized_[id])
    return;

  const int state_index = (id == 1) ? 9 : 10;
  ekf_.x[state_index] = armor.xyz_in_world[2] - ekf_.x[4];
  outpost_height_initialized_[id] = true;
  clamp_outpost_height_offsets();
  tools::logger()->debug(
    "[Target][Outpost] armor {} height offset initialized to {:.3f} m", id,
    ekf_.x[state_index]);
}

void Target::clamp_outpost_height_offsets()
{
  if (name != ArmorName::outpost || armor_num_ != 3) return;

  for (int id = 1; id < 3; ++id) {
    if (!outpost_height_initialized_[id]) continue;

    const int state_index = (id == 1) ? 9 : 10;
    const double sign = ekf_.x[state_index] < 0.0 ? -1.0 : 1.0;
    const double magnitude = std::clamp(
      std::abs(ekf_.x[state_index]), OUTPOST_HEIGHT_MIN, OUTPOST_HEIGHT_MAX);
    ekf_.x[state_index] = sign * magnitude;
  }
}

void Target::enforce_outpost_radius()
{
  if (name != ArmorName::outpost || armor_num_ != 3) return;

  ekf_.x[8] = OUTPOST_RADIUS;
  ekf_.P.row(8).setZero();
  ekf_.P.col(8).setZero();
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  if (name == ArmorName::outpost) return !r_ok;

  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 20 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

void Target::lock_outpost_rotation_direction()
{
  if (
    name != ArmorName::outpost || outpost_rotation_direction_ != 0 || update_count_ <= 10 ||
    std::abs(ekf_.x[7]) <= OUTPOST_DIRECTION_THRESHOLD)
    return;

  outpost_rotation_direction_ = ekf_.x[7] > 0 ? 1 : -1;
  ekf_.x[7] = outpost_rotation_direction_ * OUTPOST_ROTATION_SPEED;

  // Once the random direction is identified, the known outpost speed is exact.
  ekf_.P.row(7).setZero();
  ekf_.P.col(7).setZero();
}

// 计算出装甲板中心的坐标（考虑长短轴和高度差）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = x[4];
  if (name == ArmorName::outpost && armor_num_ == 3) {
    if (id == 1) armor_z += x[9];
    if (id == 2) armor_z += x[10];
  } else if (use_l_h) {
    armor_z += x[10];
  }

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto is_outpost = name == ArmorName::outpost && armor_num_ == 3;

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dl = (is_outpost && id == 1) ? 1.0 : 0.0;
  auto dz_dh = ((is_outpost && id == 2) || (!is_outpost && use_l_h)) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0, dz_dl, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
