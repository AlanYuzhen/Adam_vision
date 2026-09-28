#include "camera.hpp"

#include <filesystem>
#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  camera_name_ = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");

  if (camera_name_ == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  else if (camera_name_ == "hikrobot") {
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name_ + "!");
  }

  tools::logger()->info(
    "[Camera] Initialized: camera={} exposure_ms={:.1f} (config: {})",
    camera_name_, exposure_ms, std::filesystem::absolute(config_path).string());
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}

void Camera::reload(const YAML::Node & yaml)
{
  auto exposure_ms = yaml["exposure_ms"].as<double>();
  if (camera_name_ == "hikrobot") {
    auto gain = yaml["gain"].as<double>();
    auto * hik = dynamic_cast<HikRobot *>(camera_.get());
    if (hik) {
      hik->update_params(exposure_ms, gain);
    }
  }
}

}  // namespace io