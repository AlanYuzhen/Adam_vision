#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  // Keep the original model as the default; the nano model is opt-in.
  auto variant = yaml["yolov5_variant"] ? yaml["yolov5_variant"].as<std::string>() : "original";
  if (variant == "yolov5n") {
    if (!yaml["yolov5n_model_path"]) {
      throw std::runtime_error("yolov5_variant is yolov5n but yolov5n_model_path is missing");
    }
    model_path_ = yaml["yolov5n_model_path"].as<std::string>();
  } else if (variant == "original") {
    model_path_ = yaml["yolov5_model_path"].as<std::string>();
  } else {
    throw std::runtime_error("Unknown yolov5_variant: " + variant);
  }
  use_direct_fp32_input_ = (variant == "yolov5n");
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);
  auto model = core_.read_model(model_path_);
  ov::AnyMap compile_config;
  compile_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
  if (use_direct_fp32_input_ && device_ == "GPU")
    compile_config[ov::hint::inference_precision.name()] = ov::element::f16;
  if (!use_direct_fp32_input_) {
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();
    input.tensor().set_element_type(ov::element::u8).set_shape({1, 640, 640, 3})
      .set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    input.model().set_layout("NCHW");
    input.preprocess().convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB).scale(255.0);
    model = ppp.build();
  }
  compiled_model_ = core_.compile_model(
    model, device_, compile_config);
  if (use_direct_fp32_input_) {
    for (std::size_t i = 0; i < async_requests_.size(); ++i) {
      async_requests_[i] = compiled_model_.create_infer_request();
      async_inputs_[i] = cv::Mat(1, 3 * 640 * 640, CV_32F);
    }
    async_inputs_[0].setTo(0.0f);
    ov::Tensor warmup(ov::element::f32, {1, 3, 640, 640}, async_inputs_[0].ptr<float>());
    async_requests_[0].set_input_tensor(warmup);
    async_requests_[0].start_async();
    async_submit_ = 1;
    async_ready_ = 0;
    async_started_ = true;
  } else {
    infer_request_ = compiled_model_.create_infer_request();
  }
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = std::min(640, static_cast<int>(std::round(bgr_img.rows * scale)));
  auto w = std::min(640, static_cast<int>(std::round(bgr_img.cols * scale)));
  pad_x_ = (640 - w) / 2;
  pad_y_ = (640 - h) / 2;
  inference_image_size_ = bgr_img.size();

  // Infantry-v5n uses centered letterbox preprocessing with a gray border.
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(124, 124, 124));
  auto roi = cv::Rect(pad_x_, pad_y_, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  cv::Mat input_blob;
  if (use_direct_fp32_input_) {
    input_blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(640, 640),
      cv::Scalar(), true, false, CV_32F);
    input_blob.copyTo(async_inputs_[async_submit_]);
  }

  // infer
  if (use_direct_fp32_input_) {
    ov::Tensor input_tensor(ov::element::f32, {1, 3, 640, 640},
      async_inputs_[async_submit_].ptr<float>());
    async_requests_[async_submit_].set_input_tensor(input_tensor);
    async_requests_[async_submit_].start_async();
    async_requests_[async_ready_].wait();
    auto output_tensor = async_requests_[async_ready_].get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 25200 ||
        output_shape[2] != 22) {
      tools::logger()->error("Unexpected YOLOv5 output shape");
      return {};
    }
    cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
    async_submit_ = (async_submit_ + 1) % async_requests_.size();
    async_ready_ = (async_ready_ + 1) % async_requests_.size();
    return parse(scale, output, raw_img, frame_count);
  } else {
    ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);
    infer_request_.set_input_tensor(input_tensor);
  }
  infer_request_.infer();

  // postprocess
  auto output_tensor = infer_request_.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 25200 ||
      output_shape[2] != 22) {
    tools::logger()->error("Unexpected YOLOv5 output shape");
    return {};
  }
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  if (output.empty() || output.rows != 25200 ||
      output.cols != 8 + 1 + color_num_ + class_num_) {
    tools::logger()->error(
      "Unexpected YOLOv5 output matrix shape: {}x{}", output.rows, output.cols);
    return {};
  }

  if (scale <= 0.0 || inference_image_size_.width <= 0 ||
      inference_image_size_.height <= 0) {
    tools::logger()->error("Invalid YOLOv5 postprocess parameters");
    return {};
  }

  // The model emits LT, LB, RB, RT. Armor expects LT, RT, RB, LB.
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  // Column 8 is a raw objectness logit. Color and class columns are already
  // sigmoid outputs in Infantry-v5n.
  const double raw_score_threshold =
    std::log(static_cast<double>(score_threshold_) / (1.0 - score_threshold_));

  for (int r = 0; r < output.rows; r++) {
    const double raw_score = output.at<float>(r, 8);
    if (raw_score < raw_score_threshold) continue;
    const float score = static_cast<float>(sigmoid(raw_score));

    int color_id = 0;
    for (int i = 1; i < color_num_; ++i) {
      if (output.at<float>(r, 9 + i) > output.at<float>(r, 9 + color_id)) {
        color_id = i;
      }
    }

    int class_id = 0;
    for (int i = 1; i < class_num_; ++i) {
      if (output.at<float>(r, 13 + i) > output.at<float>(r, 13 + class_id)) {
        class_id = i;
      }
    }

    if (color_id == 3) continue;  // Ignore purple armor.

    auto decode_point = [&](int point_index) {
      const float x =
        static_cast<float>((output.at<float>(r, point_index * 2) - pad_x_) / scale);
      const float y =
        static_cast<float>((output.at<float>(r, point_index * 2 + 1) - pad_y_) / scale);
      return cv::Point2f(x, y);
    };

    const std::vector<cv::Point2f> model_key_points = {
      decode_point(0), decode_point(1), decode_point(2), decode_point(3)};
    const std::vector<cv::Point2f> armor_key_points = {
      model_key_points[0], model_key_points[3], model_key_points[2], model_key_points[1]};

    if (std::any_of(
          armor_key_points.begin(), armor_key_points.end(), [](const cv::Point2f & point) {
            return !std::isfinite(point.x) || !std::isfinite(point.y);
          })) {
      continue;
    }

    color_ids.emplace_back(color_id);
    num_ids.emplace_back(class_id);
    boxes.emplace_back(cv::boundingRect(armor_key_points));
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    const auto & points = armors_key_points[i];
    const bool points_in_image = std::all_of(
      points.begin(), points.end(), [this](const cv::Point2f & point) {
        return point.x >= 0.0F && point.x <= inference_image_size_.width &&
               point.y >= 0.0F && point.y <= inference_image_size_.height;
      });
    if (!points_in_image) continue;

    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost && armor.name != ArmorName::base);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // MultiThreadDetector calls this entry point directly, so reconstruct the
  // same letterbox parameters used by detect().
  inference_image_size_ = bgr_img.size();
  const auto h = std::min(640, static_cast<int>(std::round(bgr_img.rows * scale)));
  const auto w = std::min(640, static_cast<int>(std::round(bgr_img.cols * scale)));
  pad_x_ = (640 - w) / 2;
  pad_y_ = (640 - h) / 2;
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
