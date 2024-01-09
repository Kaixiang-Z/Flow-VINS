/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-30 16:49:46
 * @Description:
 */
#include "interface.h"
#include "include/posegraph.h"

namespace FLOW_VINS {

// loop fusion
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_camera_pose_visual;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_loop_odometry_rect;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_loop_point_cloud;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_loop_margin_cloud;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_loop_match_img;

rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_pose_graph_path;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_base_path;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_loop_path[10];

CameraPoseVisualization loop_fusion_camera_pose_visual(1, 0, 0, 1);

void registerLoopFusionPub(rclcpp::Node::SharedPtr n) {
	pub_loop_match_img = n->create_publisher<sensor_msgs::msg::Image>("match_image", 1000);
	pub_loop_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
	pub_loop_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud_loop_rect", 1000);
	pub_loop_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud_loop_rect", 1000);
	pub_loop_odometry_rect = n->create_publisher<nav_msgs::msg::Odometry>("odometry_rect", 1000);

	pub_loop_pose_graph_path = n->create_publisher<nav_msgs::msg::Path>("pose_graph_path", 1000);
	pub_loop_base_path = n->create_publisher<nav_msgs::msg::Path>("base_path", 1000);
	for (int i = 1; i < 10; i++)
		pub_loop_path[i] = n->create_publisher<nav_msgs::msg::Path>("path_" + std::to_string(i), 1000);

	loop_fusion_camera_pose_visual.setScale(0.1);
	loop_fusion_camera_pose_visual.setLineWidth(0.01);
}

void pubPoseGraph(const PoseGraph& pose_graph) {
	for (int i = 1; i <= pose_graph.sequence_cnt; i++) {
		pub_loop_pose_graph_path->publish(pose_graph.path[i]);
		pub_loop_path[i]->publish(pose_graph.path[i]);
	}
	pub_loop_base_path->publish(pose_graph.base_path);
}

} // namespace FLOW_VINS