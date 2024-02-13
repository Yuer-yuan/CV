cmake_minimum_required(VERSION 3.22.1)
project(disparity)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereo main.cpp stereo_vision.hpp)

target_link_libraries(stereo ${OpenCV_LIBS})