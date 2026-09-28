#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
const int LOSE_MAX = 20;  // 丢失的阙值

// SZU RuneModel 在全局命名空间（network_deployment_interface.hpp）
class RuneModel;

namespace auto_buff
{

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);
  ~Buff_Detector();

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  void handle_lose();

  std::unique_ptr<RuneModel> rune_model_;
  Track_status status_;      // 仅供调试观察，不参与决策
  int lose_;                 // 丢失的次数
  bool net_error_logged_ = false;  // 推理异常只报一次，避免刷屏
  int last_class_id_ = -2;         // 上一次选中的类别，变化时打一行日志
  std::optional<PowerRune> last_powerrune_ = std::nullopt;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
