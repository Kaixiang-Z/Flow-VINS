/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 15:12:57
 * @Description:
 */
#pragma once

#include "../thirdparty/cameramodels/visualization.h"
#include "../utils/logger.h"
#include "../utils/tictoc.h"
#include "../utils/utils.h"
#include <chrono>
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
// loop fusion
extern rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_camera_pose_visual;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_loop_odometry_rect;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_loop_point_cloud;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_loop_margin_cloud;
extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_loop_match_img;

extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_pose_graph_path;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_base_path;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_path[10];

extern CameraPoseVisualization loop_fusion_camera_pose_visual;

class PoseGraph;

/**
 * @brief: register vio estimator node publisher
 */
void registerLoopFusionPub(rclcpp::Node::SharedPtr n);

/**
 * @brief: publish pose graph message
 */
void pubPoseGraph(const PoseGraph& pose_graph);

} // namespace FLOW_VINS
