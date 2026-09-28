#include <iostream>
#include <iomanip>
#include <cstring>
#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"

int main() {
    // 创建一个VisionToGimbal数据包
    io::VisionToGimbal tx_data;

    // 设置测试数据
    tx_data.mode = 1;  // 控制但不开火
    tx_data.yaw = 0.5f;
    tx_data.yaw_vel = 0.1f;
    tx_data.yaw_acc = 0.0f;
    tx_data.pitch = 0.3f;
    tx_data.pitch_vel = 0.05f;
    tx_data.pitch_acc = 0.0f;
    tx_data.crc16 = tools::get_crc16(
        reinterpret_cast<uint8_t *>(&tx_data), sizeof(tx_data) - sizeof(tx_data.crc16));

    std::cout << "VisionToGimbal packet structure:" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Total size: " << sizeof(tx_data) << " bytes" << std::endl;
    std::cout << std::endl;

    std::cout << "Field values:" << std::endl;
    std::cout << "  head[0]: 0x" << std::hex << (int)tx_data.head[0] << " ('" << tx_data.head[0] << "')" << std::endl;
    std::cout << "  head[1]: 0x" << std::hex << (int)tx_data.head[1] << " ('" << tx_data.head[1] << "')" << std::endl;
    std::cout << "  mode: " << std::dec << (int)tx_data.mode << std::endl;
    std::cout << "  yaw: " << tx_data.yaw << std::endl;
    std::cout << "  yaw_vel: " << tx_data.yaw_vel << std::endl;
    std::cout << "  yaw_acc: " << tx_data.yaw_acc << std::endl;
    std::cout << "  pitch: " << tx_data.pitch << std::endl;
    std::cout << "  pitch_vel: " << tx_data.pitch_vel << std::endl;
    std::cout << "  pitch_acc: " << tx_data.pitch_acc << std::endl;
    std::cout << "  crc16: 0x" << std::hex << tx_data.crc16 << std::dec << std::endl;
    std::cout << std::endl;

    std::cout << "Raw bytes (hex):" << std::endl;
    std::cout << "  ";
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&tx_data);
    for (size_t i = 0; i < sizeof(tx_data); i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl << "  ";
    }
    std::cout << std::dec << std::endl;

    return 0;
}
