#include <iostream>
#include "serial/serial.h"

int main() {
    try {
        serial::Serial ser;
        ser.setPort("/dev/ttyACM0");
        ser.setBaudrate(115200);
        auto timeout = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(timeout);
        ser.open();

        std::cout << "Serial opened successfully" << std::endl;
        std::cout << "Available bytes: " << ser.available() << std::endl;

        uint8_t buffer[64];
        for (int i = 0; i < 10; i++) {
            size_t bytes_read = ser.read(buffer, 64);
            std::cout << "Read " << bytes_read << " bytes: ";
            for (size_t j = 0; j < bytes_read && j < 10; j++) {
                printf("%02X ", buffer[j]);
            }
            std::cout << std::endl;
        }

        ser.close();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
