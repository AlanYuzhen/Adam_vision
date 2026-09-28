#ifndef TOOLS__CONFIG_RELOADER_HPP
#define TOOLS__CONFIG_RELOADER_HPP

#include <sys/stat.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "logger.hpp"
#include "yaml.hpp"

namespace tools
{

/// @brief 配置热重载器，监控YAML文件变化并触发回调
/// 使用方式:
///   ConfigReloader reloader(config_path);
///   reloader.add_callback([&](const YAML::Node& yaml) { aimer.reload(yaml); });
///   // 在主循环中定期调用:
///   reloader.check();
///   // 或手动触发:
///   reloader.force_reload();
class ConfigReloader
{
public:
  using Callback = std::function<void(const YAML::Node &)>;

  explicit ConfigReloader(const std::string & config_path, double check_interval_s = 1.0)
  : config_path_(std::filesystem::absolute(config_path).string()),
    check_interval_(check_interval_s),
    last_check_time_(std::chrono::steady_clock::now()),
    last_mtime_(0)
  {
    logger()->info("[ConfigReloader] Watching config file: {}", config_path_);
    update_mtime();
  }

  /// @brief 注册配置变更回调
  void add_callback(Callback cb) { callbacks_.push_back(std::move(cb)); }

  /// @brief 检查文件是否变更（定期调用，内部有节流）
  /// @return true 如果配置被重新加载
  bool check()
  {
    auto now = std::chrono::steady_clock::now();
    double elapsed =
      std::chrono::duration<double>(now - last_check_time_).count();
    if (elapsed < check_interval_) return false;
    last_check_time_ = now;

    auto new_mtime = get_mtime();
    if (new_mtime == last_mtime_) return false;

    last_mtime_ = new_mtime;
    return do_reload();
  }

  /// @brief 强制重新加载配置
  /// @return true 如果加载成功
  bool force_reload()
  {
    update_mtime();
    return do_reload();
  }

  const std::string & config_path() const { return config_path_; }

private:
  std::string config_path_;
  double check_interval_;
  std::chrono::steady_clock::time_point last_check_time_;
  time_t last_mtime_;
  std::vector<Callback> callbacks_;

  time_t get_mtime() const
  {
    struct stat st;
    if (stat(config_path_.c_str(), &st) != 0) return 0;
    return st.st_mtime;
  }

  void update_mtime() { last_mtime_ = get_mtime(); }

  bool do_reload()
  {
    try {
      auto yaml = YAML::LoadFile(config_path_);
      logger()->info("[ConfigReloader] ===== Reloading config from: {} =====", config_path_);
      for (auto & cb : callbacks_) {
        cb(yaml);
      }
      logger()->info("[ConfigReloader] ===== Config reload complete =====");
      return true;
    } catch (const std::exception & e) {
      logger()->error("[ConfigReloader] Failed to reload config: {}", e.what());
      return false;
    }
  }
};

}  // namespace tools

#endif  // TOOLS__CONFIG_RELOADER_HPP
