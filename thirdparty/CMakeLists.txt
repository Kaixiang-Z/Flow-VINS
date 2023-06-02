cmake_minimum_required(VERSION 2.8.3)
project(vins_third_party)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


catkin_package(
    INCLUDE_DIRS CameraModel DBoW DUtils DVision
    LIBRARIES vins_third_party
    CATKIN_DEPENDS roscpp std_msgs
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    )



add_library(vins_third_party
    CameraModel/camera.cc
    CameraModel/camera_factory.cc
    CameraModel/pinhole_camera.cc
    CameraModel/cata_camera.cc
    CameraModel/equidstant_camera.cc
    CameraModel/gpl.cc
    DBoW/BowVector.cpp
    DBoW/FBrief.cpp
    DBoW/FeatureVector.cpp
    DBoW/QueryResults.cpp
    DBoW/ScoringObject.cpp
    DUtils/Random.cpp
    DUtils/Timestamp.cpp
    DVision/BRIEF.cpp
    VocabularyBinary.cpp
    )


target_link_libraries(vins_third_party ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
