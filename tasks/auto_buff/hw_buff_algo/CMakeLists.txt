cmake_minimum_required(VERSION 3.10)
project(buff_algorithm)

# HWauto_buff2026 算法链（PnP/tracker/predictor），已合入 tasks/auto_buff/hw_buff_algo
# 仅编译 locate/predictor/selector/tracker；detector 依赖 TensorRT，不使用

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O3 -Wall -Wextra)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB BUFF_ALGO_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/locate/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/predictor/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/selector/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tracker/*.cpp"
)

add_library(buff_algorithm STATIC ${BUFF_ALGO_SOURCES})

target_include_directories(buff_algorithm PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(buff_algorithm PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

target_include_directories(buff_algorithm PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
