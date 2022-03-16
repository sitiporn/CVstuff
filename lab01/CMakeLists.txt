cmake_minimum_required(VERSION 3.12)

project(lab01)
set(OpenCV_DIR /usr/include/opencv4)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDES_DIRS})
add_executable(lab01 optical_flow.cpp)
target_link_libraries(lab01 ${OpenCV_LIBS})
