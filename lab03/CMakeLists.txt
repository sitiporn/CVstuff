cmake_minimum_required(VERSION 3.12)

project(lab03)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 4.5.2 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})


add_executable(get-homography main.cpp HomographyData.cpp)
target_include_directories(get-homography PUBLIC ${OpenCV_LIBS}) 
target_link_libraries(get-homography ${OpenCV_LIBS})

add_executable(playback playback.cpp HomographyData.cpp)
target_include_directories(playback PUBLIC ${OpenCV_LIBS})
target_link_libraries(playback ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})

add_executable(playback3 playback_v3.cpp HomographyData.cpp)
target_include_directories(playback3 PUBLIC ${OpenCV_LIBS})
target_link_libraries(playback3 ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})


add_executable(playback4 playback_v4.cpp HomographyData.cpp)
target_include_directories(playback4 PUBLIC ${OpenCV_LIBS})
target_link_libraries(playback4 ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})


add_executable(playback5 playback_v5.cpp HomographyData.cpp)
target_include_directories(playback5 PUBLIC ${OpenCV_LIBS})
target_link_libraries(playback5 ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})
