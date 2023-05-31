cmake_minimum_required(VERSION 3.22.1)
project(stereo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Development)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_executable(stereo main.cpp stereo_vision.cpp stereo_vision.h data/config.h)

target_link_libraries(stereo ${OpenCV_LIBS})
target_link_libraries(stereo Python3::Python)