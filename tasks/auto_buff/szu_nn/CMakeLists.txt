cmake_minimum_required(VERSION 3.22.0)
project(NNdeployment)

# SZU 26_NNDeployment 检测库（RuneModel，OpenVINO 后端），已合入 tasks/auto_buff/szu_nn
# 仅编译 OpenVINO 推理路径；TensorRT 不编译

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

set(NNdeployment_src
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/network_deployment_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_deployment_lib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_openvino.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/performance/network_performance.cpp
)

add_library(NNdeployment_lib STATIC ${NNdeployment_src})

target_include_directories(NNdeployment_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(NNdeployment_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

target_link_libraries(NNdeployment_lib PUBLIC
    ${OpenCV_LIBS}
)
target_link_libraries(NNdeployment_lib PRIVATE
    openvino::runtime
)

target_compile_definitions(NNdeployment_lib PRIVATE
    NNDEPLOYMENT_WITH_OPENVINO=1
    NNDEPLOYMENT_WITH_TENSORRT=0
)
