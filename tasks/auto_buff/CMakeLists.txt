cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

# ===== 合入的算法子库（2026-09-27：从外部同级目录改为仓库内自包含）=====
add_subdirectory(hw_buff_algo)  # HWauto_buff2026: PnP/tracker/predictor
add_subdirectory(szu_nn)        # SZU 26_NNDeployment: RuneModel 检测（OpenVINO）

add_library(auto_buff OBJECT
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
)

# 合并 SZU RuneModel + HW buff 算法链
target_link_libraries(auto_buff auto_aim openvino::runtime ${CERES_LIBRARIES}
  NNdeployment_lib buff_algorithm)

# 头文件：合入子库的公开头
target_include_directories(auto_buff PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hw_buff_algo/include
  ${CMAKE_CURRENT_SOURCE_DIR}/szu_nn/interface
)
