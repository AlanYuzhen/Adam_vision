#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace io
{
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
};

class Camera
{
public:
  Camera(const std::string & config_path);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  /// @brief 热重载相机参数（曝光、增益）
  void reload(const YAML::Node & yaml);

private:
  std::unique_ptr<CameraBase> camera_;
  std::string camera_name_;
};

}  // namespace io

#endif  // IO__CAMERA_HPP