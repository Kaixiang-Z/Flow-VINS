cmake_minimum_required(VERSION 3.8)
project(thirdparty)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wno-error=deprecated-declarations -Wno-deprecated-declarations")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# OpenCV
find_package(OpenCV 4.7.0 REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

# Eigen
find_package(Eigen3)


ament_export_libraries(thirdparty)
ament_export_dependencies(rclcpp std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros image_transport Boost OpenCV Ceres Eigen3)

add_library(thirdparty
    cameramodels/camera.cc
    cameramodels/camerafactory.cc
    cameramodels/pinholecamera.cc
    cameramodels/catacamera.cc
    cameramodels/equidstantcamera.cc
    cameramodels/gpl.cc
    cameramodels/visualization.cc
    dbow/bowvector.cc
    dbow/fbrief.cc
    dbow/featurevector.cc
    dbow/queryresults.cc
    dbow/scoringobject.cc
    dbow/vocabularybinary.cc
    dutils/drandom.cc
    dutils/timestamp.cc
    dvision/brief.cc
    )


ament_target_dependencies(thirdparty rclcpp std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros image_transport Boost OpenCV Ceres Eigen3)

install(TARGETS thirdparty
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY cameramodels dbow dvision dutils
    DESTINATION include
)
  
ament_package()
