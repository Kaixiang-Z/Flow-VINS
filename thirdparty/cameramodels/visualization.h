/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: ROS display
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class CameraPoseVisualization {
public:
	std::string marker_ns;
	/**
	 * @brief: set camera color and instance
	 */
	CameraPoseVisualization(float r, float g, float b, float a);

	/**
	 * @brief: set camera scale, scale relates to the size of camera display
	 */
	void setScale(double s);

	/**
	 * @brief: set camera line width
	 */
	void setLineWidth(double width);

	/**
	 * @brief: set camera pose and orientation
	 */
	void add_pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q);

	/**
	 * @brief: reset camera display
	 */
	void reset();

	/**
	 * @brief: publish camera display topic
	 */
	void publish_by(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub,
	                const std_msgs::msg::Header& header);

	/**
	 * @brief: add camera edge line to display
	 */
	void add_edge(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);

	/**
	 * @brief: add camera loop edge line to display
	 */
	void add_loopedge(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);

private:
	std::vector<visualization_msgs::msg::Marker> markers;
	std_msgs::msg::ColorRGBA image_boundary_color;
	std_msgs::msg::ColorRGBA optical_center_connector_color;
	double scale;
	double line_width;

	static const Eigen::Vector3d imlt;
	static const Eigen::Vector3d imlb;
	static const Eigen::Vector3d imrt;
	static const Eigen::Vector3d imrb;
	static const Eigen::Vector3d oc;
	static const Eigen::Vector3d lt0;
	static const Eigen::Vector3d lt1;
	static const Eigen::Vector3d lt2;
};