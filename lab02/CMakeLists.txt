cmake_minimum_required(VERSION 3.12)
project(lab02)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 4.5.2 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(lab02 rectify_homo.cpp)

target_link_libraries(lab02 ${OpenCV_LIBS})

