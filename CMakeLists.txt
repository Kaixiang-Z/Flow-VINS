cmake_minimum_required(VERSION 3.0)
project(vio_system)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wno-error=deprecated-declarations -Wno-deprecated-declarations")

# set cv_bridge link to OpenCV 4.7.0
set(cv_bridge_DIR  /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        geometry_msgs
        nav_msgs
        tf
        cv_bridge
        image_transport
        imu_data
        )

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# OpenCV
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV 4.7.0 REQUIRED)
message(STATUS "OpenCV Libs: ${OpenCV_LIBS}\n")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV Headers: ${OpenCV_INCLUDE_DIRS}")

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")

# Ceres
find_package(Ceres REQUIRED)

# Eigen
find_package(Eigen3)


include_directories(
        ${catkin_INCLUDE_DIRS} 
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS} 
        ${CUDA_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})

catkin_package()

# build library
add_library(vio_system_lib
        src/Common.cpp
        src/Feature.cpp
        src/Estimator.cpp
        src/FactorGraph.cpp
        src/Initialize.cpp
        src/LoopFusion.cpp
        src/Segment.cpp
        src/Publisher.cpp
        src/Parameters.cpp
        src/KalmanFilter.cpp
        src/HealthyCheck.cpp

        thirdparty/CameraModel/Camera.cpp
        thirdparty/CameraModel/Gpl.cpp
        thirdparty/CameraModel/CameraFactory.cpp
        thirdparty/CameraModel/CataCamera.cpp
        thirdparty/CameraModel/EquidstantCamera.cpp
        thirdparty/CameraModel/PinholeCamera.cpp
        thirdparty/DBoW/BowVector.cpp
        thirdparty/DBoW/FBrief.cpp
        thirdparty/DBoW/FeatureVector.cpp
        thirdparty/DBoW/QueryResults.cpp
        thirdparty/DBoW/ScoringObject.cpp
        thirdparty/DUtils/Random.cpp
        thirdparty/DUtils/Timestamp.cpp
        thirdparty/DVision/BRIEF.cpp
        thirdparty/VocabularyBinary.cpp

        )

target_link_libraries(vio_system_lib 
        nvinfer
        nvonnxparser
        cudart
        nvinfer_plugin 
        ${catkin_LIBRARIES} 
        ${OpenCV_LIBS} 
        ${CERES_LIBRARIES}
        )

# generate vio system executable

add_executable(vio_system_node vio_system_node.cc)

target_link_libraries(vio_system_node vio_system_lib)

