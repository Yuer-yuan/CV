cmake_minimum_required(VERSION 3.22.1)
project(disparity)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Development)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_executable(disparity main.cpp stereo_vision.cpp stereo_vision.h)

target_link_libraries(disparity ${OpenCV_LIBS})
target_link_libraries(disparity Python3::Python)