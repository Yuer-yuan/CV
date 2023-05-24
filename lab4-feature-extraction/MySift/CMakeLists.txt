cmake_minimum_required(VERSION 3.22.1)
project(MySift)

find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Development)

message(STATUS "Opencv_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Opencv_LIBS: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_executable(main main.cpp sift.cpp sift.h util.h)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main Python3::Python)

