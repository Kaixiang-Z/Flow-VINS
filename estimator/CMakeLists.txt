cmake_minimum_required(VERSION 3.8)
project(estimator)

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
message(STATUS "OpenCV Libs: ${OpenCV_LIBS}\n")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV Headers: ${OpenCV_INCLUDE_DIRS}")

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
add_library(estimator_lib
        src/estimator.cc
        src/feature.cc
        src/factor.cc
        src/initialize.cc
        src/parameter.cc
        src/kalmanfilter.cc
)
target_link_libraries(estimator_lib  ${OpenCV_LIBS} ${CERES_LIBRARIES})
ament_target_dependencies(estimator_lib rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge image_transport thirdparty OpenCV Boost Ceres Eigen3)

add_executable(estimator_node estimator_node.cc interface.cc)
ament_target_dependencies(estimator_node rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge image_transport thirdparty OpenCV Boost Ceres Eigen3)
target_link_libraries(estimator_node estimator_lib) 



# Install nodes
install(
TARGETS estimator_node
DESTINATION lib/${PROJECT_NAME}
)

ament_package()
