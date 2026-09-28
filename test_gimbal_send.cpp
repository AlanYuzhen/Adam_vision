#include <iostream>
#include <iomanip>
#include "io/gimbal/gimbal.hpp"
#include "tools/logger.hpp"

void print_hex(const uint8_t* data, size_t len) {
    std::cout << "Hex data (" << len << " bytes): ";
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

int main() {
    tools::logger()->info("Testing Gimbal send...");

    // 创建Gimbal对象
    io::Gimbal gimbal("configs/calibration.yaml");

    tools::logger()->info("Gimbal initialized, waiting 2 seconds...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 发送测试数据
    tools::logger()->info("Sending test data...");

    for (int i = 0; i < 5; i++) {
        // 发送控制指令
        gimbal.send(true, false, 0.1f * i, 0.0f, 0.0f, 0.05f * i, 0.0f, 0.0f);

        tools::logger()->info("Sent packet #{}", i + 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    tools::logger()->info("Test complete. Check if lower machine received data.");

    return 0;
}
