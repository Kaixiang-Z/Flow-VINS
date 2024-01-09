/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vector>

namespace FLOW_VINS {

extern std::string IMAGE_TOPIC;

/**
 * @brief: set semantic segmentation process parameters from config file
 */
void readSemanticParameters(const std::string& config_file);

} // namespace FLOW_VINS