cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)
# ===== SZU RuneModel + HW buff 算法链 =====


# ===== 可执行程序 =====
# 交接提交把原 CMakeLists 中的 add_executable 段整体丢失了，导致
# `cmake -B build && make` 只能编出库、没有任何可运行程序（文档里的
# ./build/standard、./build/auto_buff_test 等只在本机残留的旧二进制上存在）。
# 下面按 build/CMakeFiles/*/link.txt 与 *.dir/build.make 的物证恢复原有目标集合。
# tools 是 OBJECT 库且未在自己的 CMakeLists 里声明依赖，fmt 由可执行目标显式链接
# （物证：旧 link.txt 中存在 /usr/lib/x86_64-linux-gnu/libfmt.so.9.1.0）
set(SP_LIBS auto_buff auto_aim io tools fmt::fmt)

# src/*.cpp（实车/调试主程序）
foreach(t
    standard
    standard_mpc
    mt_standard
    uav
    uav_debug
    auto_aim_debug_mpc
    auto_buff_debug
    auto_buff_debug_mpc
    mt_auto_aim_debug)
  add_executable(${t} src/${t}.cpp)
  target_link_libraries(${t} ${SP_LIBS})
endforeach()

# tests/*.cpp（离线回放与硬件自检）
foreach(t
    auto_aim_test
    auto_buff_test
    buff_perf_tmp
    camera_detect_test
    camera_test
    camera_thread_test
    cboard_test
    detector_video_test
    dm_test
    fire_test
    gimbal_response_test
    gimbal_test
    handeye_test
    minimum_vision_system
    multi_usbcamera_test
    planner_test
    planner_test_offline
    usbcamera_detect_test
    usbcamera_test)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} ${SP_LIBS})
endforeach()

# calibration/*.cpp（标定工具）
foreach(t
    calibrate_camera
    calibrate_handeye
    calibrate_robotworld_handeye
    capture
    split_video)
  add_executable(${t} calibration/${t}.cpp)
  target_link_libraries(${t} ${SP_LIBS})
endforeach()

add_executable(print_packet print_packet.cpp)
target_link_libraries(print_packet ${SP_LIBS})

