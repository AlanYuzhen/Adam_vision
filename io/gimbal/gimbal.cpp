#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <cmath>

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);  // 设置波特率
    auto timeout = serial::Timeout::simpleTimeout(100);  // 100ms超时
    serial_.setTimeout(timeout);
    serial_.open();
    tools::logger()->info("[Gimbal] Serial opened on {}", com_port);
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  // 等待一个合法的四元数（跳过全零/未初始化的包）
  Eigen::Quaterniond q0;
  int skip_count = 0;
  while (true) {
    auto [q_tmp, t_tmp] = queue_.pop();
    double norm2 = q_tmp.squaredNorm();
    if (norm2 > 0.5 && norm2 < 1.5 && !std::isnan(norm2)) {
      q0 = q_tmp;
      break;
    }
    skip_count++;
    if (skip_count % 10 == 0)
      tools::logger()->warn("[Gimbal] Skipped {} invalid quaternions (norm2={:.4f})", skip_count, norm2);
  }
  if (skip_count > 0)
    tools::logger()->warn("[Gimbal] Skipped {} invalid quaternion packets before getting valid data", skip_count);

  Eigen::Vector3d euler0 = tools::eulers(q0, 2, 1, 0);
  tools::logger()->info(
    "[Gimbal] First valid q received. IMU init: yaw={:.3f}deg pitch={:.3f}deg roll={:.3f}deg",
    euler0[0] * 57.3, euler0[1] * 57.3, euler0[2] * 57.3);
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    // 1. 判空保护
    if (queue_.empty()) return Eigen::Quaterniond::Identity();

    // 2. 取出队首元素 (q_a) 并移除
    auto [q_a, t_a] = queue_.pop(); // 确保这一行只出现一次！
    
    // 3. 如果队列空了（只有一帧数据），直接返回 q_a，无法插值
    if (queue_.empty()) return q_a; 

    // 4. 查看新的队首元素 (q_b)，不移除
    auto [q_b, t_b] = queue_.front();
    
    // 5. 计算时间差
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    
    // 防止除零错误 (如果时间戳完全相同)
    if (std::abs(t_ab) < 1e-6) return q_b;

    // 6. 插值
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();

    // 情况 A: 目标时间 t 在 t_a 之前（延迟较大），直接返回插值结果
    if (t < t_a) return q_c;
    
    // 情况 B: 目标时间 t 落在区间 [t_a, t_b] 内，返回插值结果
    if (t <= t_b) return q_c;

    // 情况 C: 目标时间 t 比 t_b 还要晚 (t > t_b)
    // 说明 t_a 太旧了，不满足要求，continue 进入下一轮循环
    // 此时 q_a 已经被 pop 移除了，q_b 会在下一轮变成新的 q_a
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    std::lock_guard<std::mutex> lock(serial_mutex_);

    if (serial_.isOpen()) {
        serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    std::lock_guard<std::mutex> lock(serial_mutex_);

    if (serial_.isOpen()) {
        serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    // read 函数通常不需要锁，因为只有 read_thread 一个线程在读
    // 但如果 serial 库不是线程安全的，可能需要锁，这里暂时不加锁以避免阻塞
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 100) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (rx_data_.head[0] != 'S' || rx_data_.head[1] != 'P') continue;

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    error_count = 0;
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);

    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        {
          static int inv_cnt = 0;
          if (inv_cnt++ % 500 == 0)
            tools::logger()->warn("[Gimbal] Invalid mode: {} (x{})", rx_data_.mode, inv_cnt);
        }
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    
    {
        std::lock_guard<std::mutex> lock(serial_mutex_); // 【关键修复3】加锁防止重连时主线程写入
        try {
            if (serial_.isOpen()) serial_.close();
        } catch (...) {}
    } // 锁在这里释放

    std::this_thread::sleep_for(std::chrono::seconds(1));

    try {
      std::lock_guard<std::mutex> lock(serial_mutex_); // 【关键修复3】加锁进行重连
      serial_.open();  
      
      if (serial_.isOpen()) {
        queue_.clear();
        tools::logger()->info("[Gimbal] Reconnected serial successfully.");
        break;
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io