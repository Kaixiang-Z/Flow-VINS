/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 15:12:57
 * @Description:
 */

#pragma once

#include "../thirdparty/cameramodels/visualization.h"
#include "../utils/tictoc.h"
#include "../utils/utils.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

namespace FLOW_VINS {
// vio estimator
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
extern nav_msgs::msg::Path path;

class Estimator;

/**
 * @brief: register vio estimator node publisher
 */
void registerEstimatorPub(rclcpp::Node::SharedPtr n);

/**
 * @brief: print vio estimator status message
 */
void printStatistics(const Estimator& estimator, double t);

/**
 * @brief: publish tracking image
 */
void pubTrackImage(const cv::Mat& imgTrack, double t);

/**
 * @brief: publish odometry & path message and write files
 */
void pubOdometry(const Estimator& estimator, const std_msgs::msg::Header& header);

/**
 * @brief: publish key poses maker (only in window size)
 */
void pubKeyPoses(const Estimator& estimator, const std_msgs::msg::Header& header);

/**
 * @brief: publish camera visualization pose
 */
void pubCameraPose(const Estimator& estimator, const std_msgs::msg::Header& header);

/**
 * @brief: publish point cloud
 */
void pubPointCloud(const Estimator& estimator, const std_msgs::msg::Header& header);

/**
 * @brief: publish tf state
 */
void pubTF(const Estimator& estimator, const std_msgs::msg::Header& header);

/**
 * @brief: publish key frame
 */
void pubKeyframe(const Estimator& estimator);

} // namespace FLOW_VINS
