#include "buff_detector.hpp"

#include <exception>
#include <opencv2/core.hpp>

#include "network_deployment_interface.hpp"
#include "tools/logger.hpp"

namespace auto_buff
{
Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0)
{
  auto yaml = YAML::LoadFile(config);
  const std::string model_path = yaml["rune_model"]["path"].as<std::string>();
  // SZU RuneModel：OpenVINO 后端，Intel 集显用 GPU device
  //
  // 必须用 sync：async 模式下 netProcess(第 N 帧) 返回的是第 N-1 帧的推理结果
  // （SZU 库 network_deployment_openvino.cpp 的 asyncInfer 提交当前帧、却取上一帧的输出，
  //  调用方需要自行按 pipeline_delay 对齐）。而本模块用当前帧的 IMU 与时间戳做 PnP/预测，
  // 不补偿就会引入整整 1 帧的系统性滞后；SZU 库本身也提示"标准情况下打符应该使用 sync 推理"。
  try {
    rune_model_ = std::make_unique<RuneModel>(model_path, "sync", "openvino", "GPU");
  } catch (const std::exception & e) {
    // 模型缺失 / device 不可用 / 形状不匹配等：不要让整个程序 terminate
    tools::logger()->error(
      "[BuffDetector] RuneModel 加载失败，打符检测将全程返回空: {} (path={})", e.what(), model_path);
    rune_model_.reset();
  }
}

Buff_Detector::~Buff_Detector() = default;

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    last_powerrune_ = std::nullopt;
    status_ = LOSE;
  } else {
    status_ = TEM_LOSE;
  }
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  return detect(bgr_img);
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  if (!rune_model_) {
    handle_lose();
    return std::nullopt;
  }

  std::vector<NetRuneResult> results;
  try {
    results = rune_model_->netProcess(bgr_img);
  } catch (const std::exception & e) {
    if (!net_error_logged_) {
      tools::logger()->error("[BuffDetector] 推理抛异常，之后按丢失处理: {}", e.what());
      net_error_logged_ = true;
    }
    handle_lose();
    return std::nullopt;
  }

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  // 取置信度最高的一个结果（SZU 后处理已按 score 降序，results[0] 即最高分）
  const auto & r = results[0];

  // 类别变化时打一行日志：0=未激活, 1=小符已激活, 2=大符已激活。
  // 用于确认模型确实在报"小符已激活"，以及 results[0] 取到的不是"未激活"的扇叶。
  if (r.class_id != last_class_id_) {
    last_class_id_ = r.class_id;
    tools::logger()->info(
      "[BuffDetector] 选中类别变化: class_id={} (\"{}\") score={:.3f} 候选数={}", r.class_id,
      r.class_name, r.score, results.size());
  }

  // SZU 5 点: top, left, R, right, bottom；PnP 只需 4 角点（上/左/下/右）
  const cv::Point2f top(r.top), left(r.left), bottom(r.bottom), right(r.right);
  const cv::Point2f point_r(r.point_R);
  std::vector<cv::Point2f> kpt = {top, left, bottom, right};
  std::vector<FanBlade> fanblades;
  fanblades.emplace_back(FanBlade(kpt, point_r, _light));

  /// 生成PowerRune（r_center 直接使用模型输出的 R 字中心）
  PowerRune powerrune(fanblades, point_r, last_powerrune_);

  /// handle error
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f)
{
  return detect(bgr_img);
}

}  // namespace auto_buff
