cmake_minimum_required(VERSION 3.16)
project(ORB_SLAM3)

# Generate compile_commands.json for IDE (clangd / C++ IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# C++17：ua_tag 使用 std::optional；其余模块保持兼容
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++17.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support (required for std::optional in TagObservation).")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV >= 4.2 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)

# Prefer a locally-built Pangolin with matching C++11 ABI.
# /usr/local/lib/libpangolin.so may be an older ABI build (std::string without __cxx11),
# which causes undefined references when linking examples against libORB_SLAM3.so.
if(NOT Pangolin_DIR)
    set(_PANGOLIN_CANDIDATE "${PROJECT_SOURCE_DIR}/../../Pangolin/build/src")
    if(EXISTS "${_PANGOLIN_CANDIDATE}/PangolinConfig.cmake")
        set(Pangolin_DIR "${_PANGOLIN_CANDIDATE}" CACHE PATH "Pangolin CMake config directory" FORCE)
        message(STATUS "Using local Pangolin: ${Pangolin_DIR}")
    endif()
    unset(_PANGOLIN_CANDIDATE)
endif()
find_package(Pangolin REQUIRED)
find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/ua_tag/TagFrameData.cc
src/ua_tag/MapTagData.cc
src/ua_tag/TagInitializer.cc
src/ua_tag/TagTracker.cc
src/ua_tag/TagOptimizer.cc
src/ua_tag/TagMapExporter.cc
src/ua_tag/TagPoseConstraints.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/ua_tag/TagObservation.h
include/ua_tag/TagFrameData.h
include/ua_tag/MapTagData.h
include/ua_tag/TagOptimizer.h
include/ua_tag/TagMapExporter.h
include/ua_tag/TagPoseConstraints.h)

add_subdirectory(Thirdparty/g2o)

# Thirdparty: AprilTag
option(ORB_SLAM3_WITH_APRILTAG "Build and link AprilTag detection library" ON)
if(ORB_SLAM3_WITH_APRILTAG)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
    add_subdirectory(Thirdparty/apriltag)
endif()

# Thirdparty: GTSAM (use 4.2.2 for compatibility with CMake 3.16)
option(ORB_SLAM3_WITH_GTSAM "Build and link GTSAM optimization library" ON)
if(ORB_SLAM3_WITH_GTSAM)
    set(GTSAM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_TIMING_ALWAYS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "" FORCE)
    set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
    add_subdirectory(Thirdparty/gtsam-4.2.2)
endif()

set(ORB_SLAM3_THIRDPARTY_LIBS
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)
if(ORB_SLAM3_WITH_APRILTAG)
    list(APPEND ORB_SLAM3_THIRDPARTY_LIBS apriltag)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_APRILTAG)
    target_sources(${PROJECT_NAME} PRIVATE
        src/ua_tag/AprilTagDetector.cc
        src/ua_tag/AprilTagVisualizer.cc
        src/ua_tag/TagPoseLogger.cc
        src/ua_tag/TagViewer.cc
        include/ua_tag/TagViewer.h)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/ethz_apriltag2/include)
endif()
if(ORB_SLAM3_WITH_GTSAM)
    list(APPEND ORB_SLAM3_THIRDPARTY_LIBS gtsam)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_GTSAM)
endif()

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${ORB_SLAM3_THIRDPARTY_LIBS}
-lboost_serialization
-lcrypto
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

# TagFusion examples (per-dataset folders under Examples/TagFusion/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/TagFusion/test)

add_executable(stereo_test_tag
        Examples/TagFusion/test/stereo_test_tag.cc)
target_link_libraries(stereo_test_tag ${PROJECT_NAME})

add_executable(stereo_inertial_test_tag
        Examples/TagFusion/test/stereo_inertial_test_tag.cc)
target_link_libraries(stereo_inertial_test_tag ${PROJECT_NAME})

add_executable(test_tag_corner_jac
        Examples/TagFusion/test/test_tag_corner_jac.cc)
target_link_libraries(test_tag_corner_jac ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/TagFusion/PennCOSYVIO)
add_executable(mono_penncosyvio_tag
        Examples/TagFusion/PennCOSYVIO/mono_penncosyvio_tag.cc)
target_link_libraries(mono_penncosyvio_tag ${PROJECT_NAME})

add_executable(stereo_penncosyvio_tag
        Examples/TagFusion/PennCOSYVIO/stereo_penncosyvio_tag.cc)
target_link_libraries(stereo_penncosyvio_tag ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/TagFusion/XREAL_Glass)
add_executable(mono_xreal_glass_tag
        Examples/TagFusion/XREAL_Glass/mono_xreal_glass_tag.cc)
target_link_libraries(mono_xreal_glass_tag ${PROJECT_NAME})

add_executable(stereo_xreal_glass_tag
        Examples/TagFusion/XREAL_Glass/stereo_xreal_glass_tag.cc)
target_link_libraries(stereo_xreal_glass_tag ${PROJECT_NAME})

add_executable(stereo_inertial_xreal_glass_tag
        Examples/TagFusion/XREAL_Glass/stereo_inertial_xreal_glass_tag.cc)
target_link_libraries(stereo_inertial_xreal_glass_tag ${PROJECT_NAME})

# ethz_apriltag2 CLI for Python scripts (same backend as ua_tag::AprilTagDetector)
if(ORB_SLAM3_WITH_APRILTAG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts)
    add_executable(ethz_tag_detect scripts/ethz_tag_detect_cli.cc)
    target_link_libraries(ethz_tag_detect apriltag ${OpenCV_LIBS})
    target_include_directories(ethz_tag_detect PRIVATE
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/ethz_apriltag2/include)

    find_package(Threads REQUIRED)
    add_executable(ethz_tag_enhance_eval scripts/ethz_tag_enhance_eval.cc)
    target_link_libraries(ethz_tag_enhance_eval apriltag ${OpenCV_LIBS} Threads::Threads)
    target_include_directories(ethz_tag_enhance_eval PRIVATE
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/apriltag/ethz_apriltag2/include)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})

add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})

add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})

add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
