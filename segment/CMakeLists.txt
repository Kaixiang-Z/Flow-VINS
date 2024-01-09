cmake_minimum_required(VERSION 3.8)
project(segment)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
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

# OpenCV
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



include_directories(
        ${OpenCV_INCLUDE_DIRS} 
        ${CUDA_INCLUDE_DIRS})

enable_language(CUDA)

        # build library
add_library(segment_lib
        src/parameter.cc
        src/segment.cc
)

target_link_libraries(segment_lib  
        nvinfer
        nvonnxparser
        cudart
        nvinfer_plugin
        ${OpenCV_LIBS})
        
ament_target_dependencies(segment_lib rclcpp rcpputils std_msgs cv_bridge tf2 tf2_ros image_transport geometry_msgs visualization_msgs nav_msgs OpenCV CUDA)

add_executable(segment_node segment_node.cc)
ament_target_dependencies(segment_node rclcpp rcpputils std_msgs cv_bridge tf2 tf2_ros image_transport geometry_msgs visualization_msgs nav_msgs OpenCV CUDA)
target_link_libraries(segment_node segment_lib) 



# Install nodes
install(
TARGETS segment_node
DESTINATION lib/${PROJECT_NAME}
)

ament_package()
