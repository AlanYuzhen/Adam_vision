#include "cboard.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
CBoard::CBoard(const std::string & config_path)
: mode(Mode::idle),
  shoot_mode(ShootMode::left_shoot),
  bullet_speed(0),
  queue_(5000),
  running_(true),
  data_ahead_{Eigen::Quaterniond::Identity(), std::chrono::steady_clock::now()},
  data_behind_{Eigen::Quaterniond::Identity(), std::chrono::steady_clock::now()}
{
  read_yaml(config_path);

  try {
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[CBoard] Failed to open serial: {}", e.what());
  }

  if (serial_.isOpen()) {
    tools::logger()->info("[CBoard] Serial opened on port.");
    receive_thread_ = std::thread(&CBoard::receive_loop, this);
  }

  tools::logger()->info("[Cboard] Opened.");
}

CBoard::~CBoard()
{
  running_ = false;
  if (receive_thread_.joinable()) receive_thread_.join();
  if (serial_.isOpen()) serial_.close();
}

Eigen::Quaterniond CBoard::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  // 如果队列为空，直接返回当前已有的最新数据，防止崩溃
  if (queue_.empty()) return data_ahead_.q.normalized();

  if (data_behind_.timestamp < timestamp) data_ahead_ = data_behind_;
  while (true) {
    // 增加超时退出或判空逻辑
    if (queue_.empty()) break; 
    queue_.pop(data_behind_);
    if (data_behind_.timestamp > timestamp) break;
    data_ahead_ = data_behind_;
  }
  
  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();
  auto t_ac = timestamp - data_ahead_.timestamp;
  auto t_ab = data_behind_.timestamp - data_ahead_.timestamp;
  
  // 防止除以零
  if (t_ab.count() == 0) return q_b;

  double k = std::chrono::duration<double>(t_ac).count() / std::chrono::duration<double>(t_ab).count();
  return q_a.slerp(k, q_b).normalized();
}

void CBoard::send(Command command)
{
  if (!serial_.isOpen()) return;
  uint8_t buffer[11];
  buffer[0] = 0xA5;
  buffer[1] = (uint8_t)send_canid_;
  buffer[2] = (command.control) ? 1 : 0;
  buffer[3] = (command.shoot) ? 1 : 0;
  buffer[4] = (int16_t)(command.yaw * 1e4) >> 8;
  buffer[5] = (int16_t)(command.yaw * 1e4);
  buffer[6] = (int16_t)(command.pitch * 1e4) >> 8;
  buffer[7] = (int16_t)(command.pitch * 1e4);
  buffer[8] = (int16_t)(command.horizon_distance * 1e4) >> 8;
  buffer[9] = (int16_t)(command.horizon_distance * 1e4);
  uint8_t sum = 0;
  for (int i = 1; i < 10; i++) sum += buffer[i];
  buffer[10] = sum;
  
  try {
      serial_.write(buffer, 11);
  } catch (std::exception &e) {
      tools::logger()->warn("[CBoard] Send failed: {}", e.what());
  }
}

void CBoard::receive_loop()
{
  uint8_t head[1];
  uint8_t body[10];
  int valid_packet_count = 0;
  int invalid_packet_count = 0;

  tools::logger()->info("[CBoard] Receive loop started, waiting for data...");

  while (running_) {
    try {
        if (serial_.available() < 11) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        serial_.read(head, 1);
        if (head[0] != 0xA5) {
            invalid_packet_count++;
            if (invalid_packet_count % 1000 == 0) {
                tools::logger()->warn("[CBoard] Invalid head: 0x{:02X}, count={}", head[0], invalid_packet_count);
            }
            continue;
        }

        if (serial_.read(body, 10) != 10) continue;

        uint8_t sum = 0;
        for (int i = 0; i < 9; i++) sum += body[i];

        if (sum == body[9]) {
            valid_packet_count++;
            if (valid_packet_count % 100 == 0) {
                tools::logger()->debug("[CBoard] Valid packets received: {}", valid_packet_count);
            }
            process_packet(body[0], body + 1);
        } else {
            invalid_packet_count++;
            if (invalid_packet_count % 100 == 0) {
                tools::logger()->warn("[CBoard] Checksum error, count={}", invalid_packet_count);
            }
        }
    } catch (std::exception &e) {
        tools::logger()->warn("[CBoard] Receive error: {}", e.what());
    }
  }

  tools::logger()->info("[CBoard] Receive loop stopped. Valid: {}, Invalid: {}",
                        valid_packet_count, invalid_packet_count);
}

void CBoard::process_packet(uint8_t id, const uint8_t* data)
{
  auto timestamp = std::chrono::steady_clock::now();

  // 添加调试日志
  static int packet_count = 0;
  if (packet_count++ % 100 == 0) {  // 每100个包打印一次
    tools::logger()->debug("[CBoard] Received packet id={}, expecting quaternion_canid={}, bullet_speed_canid={}",
                           id, quaternion_canid_, bullet_speed_canid_);
  }

  if (id == quaternion_canid_) {
    auto x = (int16_t)(data[0] << 8 | data[1]) / 1e4;
    auto y = (int16_t)(data[2] << 8 | data[3]) / 1e4;
    auto z = (int16_t)(data[4] << 8 | data[5]) / 1e4;
    auto w = (int16_t)(data[6] << 8 | data[7]) / 1e4;
    queue_.push({{w, x, y, z}, timestamp});

    if (packet_count % 100 == 0) {
      tools::logger()->debug("[CBoard] Quaternion: w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}", w, x, y, z);
    }
  } else if (id == bullet_speed_canid_) {
    bullet_speed = (int16_t)(data[0] << 8 | data[1]) / 1e2;
    // 带范围校验：越界字节直接落到 idle/both_shoot，避免非法枚举值后续索引 MODES 越界
    mode = mode_from_byte(data[2]);
    shoot_mode = shoot_mode_from_byte(data[3]);
  } else {
    // 收到未知ID的包
    if (packet_count % 100 == 0) {
      tools::logger()->warn("[CBoard] Unknown packet id={}", id);
    }
  }
}

void CBoard::read_yaml(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  quaternion_canid_ = tools::read<int>(yaml, "quaternion_canid");
  bullet_speed_canid_ = tools::read<int>(yaml, "bullet_speed_canid");
  send_canid_ = tools::read<int>(yaml, "send_canid");
  
  serial_.setPort(tools::read<std::string>(yaml, "serial_port"));
  serial_.setBaudrate(tools::read<int>(yaml, "serial_baud"));
  
  // ----- 修复错误的关键点 -----
  // 先创建变量，再传引用
  auto timeout = serial::Timeout::simpleTimeout(10); 
  serial_.setTimeout(timeout);
}

}  // namespace io