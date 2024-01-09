cmake_minimum_required(VERSION 3.8)
project(loopfusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wno-error=deprecated-declarations -Wno-deprecated-declarations")


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(thirdparty REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# OpenCV
find_package(OpenCV 4.7.0 REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

# Eigen
find_package(Eigen3)


include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS} 
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})

        # build library
add_library(loopfusion_lib
        src/keyframe.cc
        src/posegraph.cc
        src/parameter.cc
)
target_link_libraries(loopfusion_lib  ${OpenCV_LIBS} ${CERES_LIBRARIES})
ament_target_dependencies(loopfusion_lib rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge image_transport thirdparty OpenCV Boost Ceres Eigen3)

add_executable(loopfusion_node loopfusion_node.cc interface.cc)
ament_target_dependencies(loopfusion_node rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge image_transport thirdparty OpenCV Boost Ceres Eigen3)
target_link_libraries(loopfusion_node loopfusion_lib) 



# Install nodes
install(
TARGETS loopfusion_node
DESTINATION lib/${PROJECT_NAME}
)

ament_package()
