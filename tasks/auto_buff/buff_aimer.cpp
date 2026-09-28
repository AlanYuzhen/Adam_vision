#include "buff_aimer.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{
namespace
{
// 取某个预测时域下"瞄准点"在世界系的方位角与俯仰角，用于求前馈角速度/角加速度。
// point_buff2world 返回的就是 HW predictor 在世界系的预测打击点。
void sample_aim(Target & t, double horizon, double & yaw, double & pitch)
{
  t.predict(horizon);
  const auto w = t.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  yaw = std::atan2(w[1], w[0]);
  pitch = std::atan2(w[2], std::hypot(w[0], w[1]));
}
}  // namespace

Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  reload(yaml);

  last_fire_t_ = std::chrono::steady_clock::now();
}

void Aimer::reload(const YAML::Node & yaml)
{
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;      // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;  // degree to rad
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();
  tools::logger()->info(
    "[BuffAimer] yaw_offset={:.2f}deg pitch_offset={:.2f}deg fire_gap={:.3f}s predict_time={:.3f}s",
    yaw_offset_ * 57.3, pitch_offset_ * 57.3, fire_gap_time_, predict_time_);
}

io::Command Aimer::aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp,
  double bullet_speed, bool to_now)
{
  io::Command command = {false, false, 0, 0};
  if (target.is_unsolve()) return command;

  // 如果子弹速度小于10，将其设为24
  if (bullet_speed < 10) bullet_speed = 24;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    command.yaw = yaw;
    command.pitch = -pitch;  //世界坐标系下的pitch向上为负
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      command.control = false;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;
  }

  if (switch_fanblade_) {
    // 只抑制当帧开火，不再重置 fire_gap 计时器。
    // 原实现这里的 last_fire_t_ = now 会被每一次"扇叶跳变"触发，而跳变频率
    // （实测约 5~12 次/秒）远高于 1/fire_gap_time，导致计时器几乎永远走不完，
    // 实测 800 帧只发出 3~6 发（理论上限约 18 发）。
    command.shoot = false;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

auto_aim::Plan Aimer::mpc_aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  if (target.is_unsolve()) return plan;

  double bullet_speed;
  // 如果子弹速度小于10，将其设为24
  if (gs.bullet_speed < 10)
    bullet_speed = 24;
  else
    bullet_speed = gs.bullet_speed;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch, horizon = 0.0;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch, &horizon)) {
    plan.yaw = yaw;
    plan.pitch = -pitch;  //世界坐标系下的pitch向上为负
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      plan.control = false;

      first_in_aimer_ = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;

    if (plan.control) {
      if (first_in_aimer_) {
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
        first_in_aimer_ = false;
      } else {
        // 前馈角速度/角加速度：对"瞄准点世界坐标的方位角/俯仰角"做前向一阶/二阶差分。
        //
        // 不能再用 predict_time_ * -1 去取"上一时刻"：
        //   1) HW BuffPredictor::predict_position 会把 dt<0 截断成 0，取不到过去；
        //   2) get_send_angle 的最终输出由弹道飞行时间 trajectory0.fly_time 决定，
        //      几乎与传入的 predict_time 无关，所以两次调用的角度差 ≈ 0。
        // 实测原实现的前馈只有真实指令变化率的 0.05%（约等于没有前馈）。
        const double h = horizon;                          // get_send_angle 实际使用的预测时域
        const double d = predict_time_ > 1e-3 ? predict_time_ : 0.1;
        double y0, p0, y1, p1, y2, p2;
        sample_aim(target, h, y0, p0);
        sample_aim(target, h + d, y1, p1);
        sample_aim(target, h + 2 * d, y2, p2);
        target.predict(0.0);  // 复原 predictor 的瞄准点，避免影响后续调用

        const double dy1 = tools::limit_rad(y1 - y0);
        const double dy2 = tools::limit_rad(y2 - y1);
        plan.yaw_vel = dy1 / d;
        plan.yaw_acc = (dy2 - dy1) / (d * d);

        const double dp1 = p1 - p0;
        const double dp2 = p2 - p1;
        plan.pitch_vel = -dp1 / d;               // plan.pitch = -pitch，故取负
        plan.pitch_acc = -(dp2 - dp1) / (d * d);
      }
    }
  }

  if (switch_fanblade_) {
    // 同 aim()：只抑制当帧，不重置 fire_gap 计时器
    plan.fire = false;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
}

bool Aimer::get_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  const bool to_now, double & yaw, double & pitch, double * horizon)
{
  // 考虑detecor所消耗的时间，此外假设aimer的用时可忽略不计
  // 如果 to_now 为 true，则根据当前时间和时间戳预测目标位置,deltatime = 现在时间减去当时照片时间，加上0.1
  target.predict(predict_time);
  // std::cout << "gap: " << detect_now_gap << std::endl;
  angle = target.ekf_x()[5];

  // 计算目标点的空间坐标
  auto aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  double d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  double h = aim_in_world[2];

  // 创建弹道对象
  tools::Trajectory trajectory0(bullet_speed, d, h);
  if (trajectory0.unsolvable) {  // 如果弹道无法解算，返回未命中结果
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    return false;
  }

  // 根据第一个弹道飞行时间预测目标位置
  target.predict(trajectory0.fly_time);
  angle = target.ekf_x()[5];

  // 计算新的目标点的空间坐标
  aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  d = fsqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  h = aim_in_world[2];
  tools::Trajectory trajectory1(bullet_speed, d, h);
  if (trajectory1.unsolvable) {  // 如果弹道无法解算，返回未命中结果
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory1: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    return false;
  }

  // 计算时间误差
  auto time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.025) {  // 如果时间误差过大，返回未命中结果
    tools::logger()->debug("[Aimer] Large time error: {:.3f}", time_error);
    return false;
  }

  // 计算偏航角和俯仰角，并返回命中结果
  // 注意：最终预测时域是 trajectory0.fly_time（弹道飞行时间），不是入参 predict_time
  if (horizon) *horizon = trajectory0.fly_time;
  yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
  pitch = trajectory1.pitch + pitch_offset_;
  return true;
};

}  // namespace auto_buff