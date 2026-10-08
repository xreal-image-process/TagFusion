cmake_minimum_required(VERSION 3.16)
project(apriltag_ethz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB APRILTAG_ETHZ_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/ethz_apriltag2/src/*.cc"
)

add_library(apriltag STATIC
    ${APRILTAG_ETHZ_SRCS}
    src/apriltag.cpp
)

# Keep legacy alias name used by some callers
add_library(apriltag_3rd ALIAS apriltag)

target_include_directories(apriltag
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/ethz_apriltag2/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(apriltag
    PUBLIC
        ${OpenCV_LIBS}
)
