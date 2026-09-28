#include <fmt/core.h>

#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"

// 临时性能统计：跑打符视频
int main(int argc, char * argv[])
{
  if (argc < 2) {
    fmt::print("usage: buff_perf <video_no_ext> [config] [end]\n");
    return 1;
  }
  std::string input = argv[1];
  std::string config = argc >= 3 ? argv[2] : "configs/sentry.yaml";
  int end = argc >= 4 ? std::atoi(argv[3]) : 0;

  cv::VideoCapture video(fmt::format("{}.avi", input));
  if (!video.isOpened()) {
    fmt::print("cannot open video\n");
    return 1;
  }

  auto_buff::Buff_Detector detector(config);
  auto_buff::Solver solver(config);
  auto_buff::BigTarget target(config);
  auto_buff::Aimer aimer(config);

  auto t0 = std::chrono::steady_clock::now();
  cv::Mat img;
  int total = 0, detect_hit = 0, shoots = 0;
  int first_detect = -1;
  std::map<int, int> status_cnt;
  double rv_sum = 0; int rv_n = 0; double rv_max = 0;
  double w_sum = 0; int w_n = 0;
  double lead_sum = 0; int lead_n = 0;
  float w_last = 0, a_last = 0, phi_last = 0, b_last = 0, rv_last = 0;

  while (true) {
    if (end > 0 && total > end) break;
    video.read(img);
    if (img.empty()) break;

    auto timestamp = t0 + std::chrono::microseconds(total * 1000000 / 30);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

    auto power_runes = detector.detect(img);
    solver.solve(power_runes);
    target.get_target(power_runes, timestamp);

    auto target_copy = target;
    auto command = aimer.aim(target_copy, timestamp, 22, false);

    total++;
    int st = static_cast<int>(target.tracker_status());
    status_cnt[st]++;

    if (power_runes.has_value()) {
      detect_hit++;
      if (first_detect < 0) first_detect = total;
    }
    if (command.shoot) shoots++;

    auto bs = target.buff_state();
    rv_last = bs.roll_velocity;
    w_last = bs.w; a_last = bs.a; phi_last = bs.phi; b_last = bs.b;
    if (std::fabs(rv_last) > 0.05) {
      rv_sum += std::fabs(rv_last); rv_n++;
      rv_max = std::max(rv_max, (double)std::fabs(rv_last));
    }
    if (std::fabs(w_last) > 0.05) { w_sum += std::fabs(w_last); w_n++; }

    if (!target.is_unsolve()) {
      auto c0 = target; c0.predict(0.0);
      auto p0 = c0.point_buff2world(Eigen::Vector3d(0, 0, 0));
      auto c1 = target; c1.predict(0.213);
      auto p1 = c1.point_buff2world(Eigen::Vector3d(0, 0, 0));
      double lead = (p1 - p0).norm();
      if (lead > 1e-4 && lead < 1.0) { lead_sum += lead; lead_n++; }
    }
  }

  auto sname = [](int s) -> std::string {
    switch (s) {
      case 0: return "CONVERGING";
      case 1: return "TRACKING";
      case 2: return "TEMP_LOST";
      case 3: return "LOST";
      default: return "?";
    }
  };

  fmt::print("===== {} =====\n", input);
  fmt::print("total frames      : {}\n", total);
  fmt::print("first detect frame: {}\n", first_detect);
  fmt::print("detect_hit        : {} ({:.1f}% all, {:.1f}% after first)\n",
    detect_hit, 100.0 * detect_hit / total,
    first_detect > 0 ? 100.0 * detect_hit / (total - first_detect + 1) : 0);
  fmt::print("tracker status    :");
  for (auto & kv : status_cnt)
    fmt::print("  {}={} ({:.0f}%)", sname(kv.first), kv.second, 100.0 * kv.second / total);
  fmt::print("\n");
  fmt::print("shoots commanded  : {}\n", shoots);
  fmt::print("roll_velocity     : avg|rv| {:.3f} max {:.3f} rad/s over {} frames, last {:.3f}\n",
    rv_n ? rv_sum / rv_n : 0, rv_max, rv_n, rv_last);
  fmt::print("sine w            : avg|w| {:.3f} over {} frames, last {:.3f}\n",
    w_n ? w_sum / w_n : 0, w_n, w_last);
  fmt::print("sine params last  : a {:.3f}  phi {:.3f}  b {:.3f}\n", a_last, phi_last, b_last);
  fmt::print("lead @0.213s      : avg {:.4f} m over {} frames\n",
    lead_n ? lead_sum / lead_n : 0, lead_n);
  return 0;
}
