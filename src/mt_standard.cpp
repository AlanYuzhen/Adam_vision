#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target(config_path);
  auto_buff::BigTarget buff_big_target(config_path);
  auto_buff::Aimer buff_aimer(config_path);

  auto_aim::multithread::CommandGener commandgener(shooter, aimer, gimbal, plotter);

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        camera.read(img, t);
        detector.push(img, t);
      } else
        continue;
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode.load()));
      last_mode = mode.load();
    }

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      static int fps_count = 0;
      static auto last_time = std::chrono::steady_clock::now();
      fps_count++;
      auto now_time = std::chrono::steady_clock::now();
      // 每隔 1 秒打印一次 FPS
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now_time - last_time).count() >= 1000) {
          tools::logger()->info("Algorithm FPS: {}", fps_count);
          fps_count = 0;
          last_time = now_time;
      }
      auto [img, armors, t] = detector.debug_pop();
      Eigen::Quaterniond q = gimbal.q(t - 1ms);

      // recorder.record(img, q, t);

      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto targets = tracker.track(armors, t);

      commandgener.push(targets, t, gimbal.state().bullet_speed, ypr);  // 发送给决策线程

    }

    /// 打符
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      cv::Mat img;
      Eigen::Quaterniond q;
      std::chrono::steady_clock::time_point t;

      camera.read(img, t);
      q = gimbal.q(t - 1ms);

      // recorder.record(img, q, t);

      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);

      buff_solver.solve(power_runes);

      const double bullet_speed = gimbal.state().bullet_speed;
      io::Command buff_command;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_command = buff_aimer.aim(target_copy, t, bullet_speed, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_command = buff_aimer.aim(target_copy, t, bullet_speed, true);
      }
      // io::Command -> VisionToGimbal：速度/加速度当前未给，置 0
      gimbal.send(
        buff_command.control, buff_command.shoot, static_cast<float>(buff_command.yaw), 0.0f, 0.0f,
        static_cast<float>(buff_command.pitch), 0.0f, 0.0f);

    } else
      continue;
  }

  detect_thread.join();

  return 0;
}