/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#pragma once
#include "../../utils/logger.h"
#include "../../utils/tictoc.h"
#include "../../utils/utils.h"
#include "../interface.h"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_resource.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <thread>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>

namespace FLOW_VINS {

// loop fusion
extern int USE_IMU;
extern std::string IMAGE_TOPIC;

extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string LOOP_FUSION_RESULT_PATH;
extern std::string VOCABULARY_PATH;
extern std::string CAMERA_PATH;
extern Eigen::Vector3d LOOP_TIC;
extern Eigen::Matrix3d LOOP_QIC;

/**
 * @brief: set loop detect process parameters from config file
 */
void readLoopFusionParameters(const std::string& config_file);

} // namespace FLOW_VINS