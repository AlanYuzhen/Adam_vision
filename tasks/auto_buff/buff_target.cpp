#include "buff_target.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace auto_buff
{
/// Target

Target::Target() : unsolvable_(true), lost_cn_(0), first_in_(true) {}

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d &) const
{
  return aim_point_world_;
}

bool Target::is_unsolve() const { return unsolvable_; }

Eigen::VectorXd Target::ekf_x() const
{
  Eigen::VectorXd x(7);
  x.setZero();

  // R 心世界系球坐标（调试用）：cam->gimbal 乘 R_fix_ 再加平移
  Eigen::Vector3d r_center_world =
    R_gimbal2world_ * (R_fix_ * state_.r_center.cast<double>() + t_camera2gimbal_);
  Eigen::Vector3d ypd = tools::xyz2ypd(r_center_world);
  x[0] = ypd[0];  // R_yaw
  x[2] = ypd[1];  // R_pitch
  x[3] = ypd[2];  // R_dis
  x[4] = state_.yaw;
  x[5] = state_.roll;
  x[6] = state_.roll_velocity;
  return x;
}

namespace
{
// 从 config 读取 HW 配置目录并构造 tracker/predictor
void init_hw_chain(
  const std::string & config_path, buff_algo::BuffMode mode,
  std::shared_ptr<buff_algo::BuffTracker> & tracker,
  std::shared_ptr<buff_algo::BuffPredictor> & predictor)
{
  auto yaml = YAML::LoadFile(config_path);
  const std::string config_dir = yaml["hw_buff"]["config_dir"].as<std::string>();

  cv::FileStorage tracker_fs(config_dir + "/tracker/tracker.yaml", cv::FileStorage::READ);
  cv::FileStorage predictor_fs(config_dir + "/predictor/predictor.yaml", cv::FileStorage::READ);

  // HW 库只接受 cv::FileNode，无法自己校验路径；这里必须显式检查，
  // 否则会退化成空节点，进而在 KalmanFilter 构造里抛 std::invalid_argument（main 无 catch）。
  if (!tracker_fs.isOpened() || !predictor_fs.isOpened()) {
    throw std::runtime_error(
      "[auto_buff] 无法打开 HW 打符配置，请检查 hw_buff.config_dir: " + config_dir +
      "（需要 tracker/tracker.yaml 与 predictor/predictor.yaml）");
  }

  tracker = std::make_shared<buff_algo::BuffTracker>(tracker_fs["buff_tracker"]);
  tracker->set_mode(mode);
  predictor = std::make_shared<buff_algo::BuffPredictor>(predictor_fs["buff_predictor"]);
}

// 共用观测更新：push + update + set_state，返回是否可解
//
// 注意 is_unsolve() 的语义 = "tracker 仍有可外推的状态"，不等于"本帧检测到了目标"：
// HW 状态机丢帧后会先进入 TEMP_LOST（仍 != LOST），所以本函数在 p 为空时也会返回 true。
// 因此调用方千万不要用 is_unsolve() 去推断 optional<PowerRune> 一定有值。
bool update_hw(
  const std::optional<PowerRune> & p, double ts,
  std::shared_ptr<buff_algo::BuffTracker> & tracker,
  std::shared_ptr<buff_algo::BuffPredictor> & predictor,
  buff_algo::BuffState & state, Eigen::Matrix3d & R_g2w, Eigen::Vector3d & t_cg,
  Eigen::Matrix3d & R_fix, Eigen::Vector3d & aim_world, bool & first_in)
{
  if (!tracker || !predictor) return false;

  if (!p.has_value()) {
    // 丢失：仍推进状态机
    tracker->update(ts);
    state = tracker->get_state();
    return tracker->status() != buff_algo::StatusType::LOST;
  }

  const PowerRune & pr = p.value();
  R_g2w = pr.R_gimbal2world;
  t_cg = pr.t_camera2gimbal;
  R_fix = pr.R_fix;

  buff_algo::Pose3f pose;
  pose.rotation = Eigen::Quaternionf(pr.pose_camera.block<3, 3>(0, 0).cast<float>());
  pose.translation = pr.pose_camera.block<3, 1>(0, 3).cast<float>();

  tracker->push(buff_algo::Buff(pose));
  tracker->update(ts);
  state = tracker->get_state();
  predictor->set_state(state, ts, ts);
  first_in = false;

  const bool ready = tracker->status() != buff_algo::StatusType::LOST;
  if (ready) {
    // 立即算一帧打击点（相机系 前x左y上z -> 云台系乘 R_fix 再加平移 -> 世界系）
    Eigen::Vector3f leaf_cam = predictor->predict_position(0.0f);
    aim_world = R_g2w * (R_fix * leaf_cam.cast<double>() + t_cg);
  }
  return ready;
}
}  // namespace

/// SmallTarget

SmallTarget::SmallTarget() : Target() {}

SmallTarget::SmallTarget(const std::string & config_path) : Target()
{
  init_hw_chain(config_path, buff_algo::BuffMode::SMALL_BUFF, tracker_, predictor_);
}

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  if (!tracker_) {
    unsolvable_ = true;
    return;
  }
  const double ts = std::chrono::duration<double>(timestamp.time_since_epoch()).count();

  if (!p.has_value()) {
    lost_cn_++;
    if (lost_cn_ > 15) {
      tracker_->reset();
      first_in_ = true;
      unsolvable_ = true;
      return;
    }
  } else {
    lost_cn_ = 0;
  }

  unsolvable_ = !update_hw(
    p, ts, tracker_, predictor_, state_, R_gimbal2world_, t_camera2gimbal_, R_fix_,
    aim_point_world_, first_in_);
}

void SmallTarget::predict(double dt)
{
  if (!tracker_ || !predictor_ || unsolvable_) return;
  Eigen::Vector3f leaf_cam = predictor_->predict_position(static_cast<float>(dt));
  aim_point_world_ = R_gimbal2world_ * (R_fix_ * leaf_cam.cast<double>() + t_camera2gimbal_);
  spd = state_.roll_velocity;
}

/// BigTarget

BigTarget::BigTarget() : Target() {}

BigTarget::BigTarget(const std::string & config_path) : Target()
{
  init_hw_chain(config_path, buff_algo::BuffMode::BIG_BUFF, tracker_, predictor_);
}

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  if (!tracker_) {
    unsolvable_ = true;
    return;
  }
  const double ts = std::chrono::duration<double>(timestamp.time_since_epoch()).count();

  if (!p.has_value()) {
    lost_cn_++;
    if (lost_cn_ > 15) {
      tracker_->reset();
      first_in_ = true;
      unsolvable_ = true;
      return;
    }
  } else {
    lost_cn_ = 0;
  }

  unsolvable_ = !update_hw(
    p, ts, tracker_, predictor_, state_, R_gimbal2world_, t_camera2gimbal_, R_fix_,
    aim_point_world_, first_in_);
}

void BigTarget::predict(double dt)
{
  if (!tracker_ || !predictor_ || unsolvable_) return;
  Eigen::Vector3f leaf_cam = predictor_->predict_position(static_cast<float>(dt));
  aim_point_world_ = R_gimbal2world_ * (R_fix_ * leaf_cam.cast<double>() + t_camera2gimbal_);
  spd = state_.roll_velocity;
}

}  // namespace auto_buff
