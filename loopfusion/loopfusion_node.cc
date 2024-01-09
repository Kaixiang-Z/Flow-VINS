/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: loop fusion node
 */
#include "../thirdparty/cameramodels/visualization.h"
#include "include/posegraph.h"
#include <rclcpp/rclcpp.hpp>

using namespace FLOW_VINS;

PoseGraph pose_graph;
std::queue<sensor_msgs::msg::Image::ConstPtr> image_buf;
std::queue<sensor_msgs::msg::PointCloud::ConstPtr> point_buf;
std::queue<nav_msgs::msg::Odometry::ConstPtr> pose_buf;
std::queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;

int sequence = 1;
Eigen::Vector3d last_t(-100, -100, -100);

/**
 * @brief: if image is unstable, start a new sequence
 */
void new_sequence() {
	LOGGER_DEBUG("new sequence");
	sequence++;
	LOGGER_DEBUG("sequence cnt %d ", sequence);
	if (sequence > 5) {
		LOGGER_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
		assert(0);
	}
	// publish topic
	for (int i = 1; i <= pose_graph.sequence_cnt; i++) {
		pub_loop_pose_graph_path->publish(pose_graph.path[i]);
		pub_loop_path[i]->publish(pose_graph.path[i]);
	}
	pub_loop_base_path->publish(pose_graph.base_path);
	// clear buffer
	m_buf.lock();
	while (!image_buf.empty())
		image_buf.pop();
	while (!point_buf.empty())
		point_buf.pop();
	while (!pose_buf.empty())
		pose_buf.pop();
	while (!odometry_buf.empty())
		odometry_buf.pop();
	m_buf.unlock();
}

/**
 * @brief: get vio odometry topic data from estimator
 */
void vio_callback(const nav_msgs::msg::Odometry::ConstPtr& pose_msg) {
	Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
	                      pose_msg->pose.pose.position.z);
	Eigen::Quaterniond vio_q;
	vio_q.w() = pose_msg->pose.pose.orientation.w;
	vio_q.x() = pose_msg->pose.pose.orientation.x;
	vio_q.y() = pose_msg->pose.pose.orientation.y;
	vio_q.z() = pose_msg->pose.pose.orientation.z;

	vio_t = pose_graph.w_r_vio * vio_t + pose_graph.w_t_vio;
	vio_q = pose_graph.w_r_vio * vio_q;

	vio_t = pose_graph.r_drift * vio_t + pose_graph.t_drift;
	vio_q = pose_graph.r_drift * vio_q;

	nav_msgs::msg::Odometry odometry;
	odometry.header = pose_msg->header;
	odometry.header.frame_id = "world";
	odometry.pose.pose.position.x = vio_t.x();
	odometry.pose.pose.position.y = vio_t.y();
	odometry.pose.pose.position.z = vio_t.z();
	odometry.pose.pose.orientation.x = vio_q.x();
	odometry.pose.pose.orientation.y = vio_q.y();
	odometry.pose.pose.orientation.z = vio_q.z();
	odometry.pose.pose.orientation.w = vio_q.w();
	pub_loop_odometry_rect->publish(odometry);

	Eigen::Vector3d vio_t_cam;
	Eigen::Quaterniond vio_q_cam;
	vio_t_cam = vio_t + vio_q * LOOP_TIC;
	vio_q_cam = vio_q * LOOP_QIC;

	loop_fusion_camera_pose_visual.reset();
	loop_fusion_camera_pose_visual.add_pose(vio_t_cam, vio_q_cam);

	loop_fusion_camera_pose_visual.publish_by(pub_loop_camera_pose_visual, pose_msg->header);
}

/**
 * @brief: get left image topic data from camera
 */
void image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
	static double last_image_time = -1;

	m_buf.lock();
	image_buf.push(image_msg);
	m_buf.unlock();

	// detect unstable camera stream
	if (last_image_time == -1)
		last_image_time = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9);
	else if (image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9) - last_image_time > 1.0 ||
	         image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9) < last_image_time) {
		LOGGER_WARN("image discontinue! detect a new sequence!");
		new_sequence();
	}
	last_image_time = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9);
}

/**
 * @brief: get pose topic from estimator
 */
void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg) {
	m_buf.lock();
	pose_buf.push(pose_msg);
	m_buf.unlock();
}

/**
 * @brief: get extrinsic topic data from estimator
 */
void extrinsic_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg) {
	m_buf.lock();
	LOOP_TIC =
	    Eigen::Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
	LOOP_QIC = Eigen::Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
	                              pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
	               .toRotationMatrix();
	m_buf.unlock();
}

/**
 * @brief: get world point topic data from estimator
 */
void point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg) {
	m_buf.lock();
	point_buf.push(point_msg);
	m_buf.unlock();
	// for visualization
	sensor_msgs::msg::PointCloud point_cloud;
	point_cloud.header = point_msg->header;
	for (auto point : point_msg->points) {
		cv::Point3f p_3d;
		p_3d.x = point.x;
		p_3d.y = point.y;
		p_3d.z = point.z;
		Eigen::Vector3d tmp = pose_graph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + pose_graph.t_drift;
		geometry_msgs::msg::Point32 p;
		p.x = static_cast<float>(tmp(0));
		p.y = static_cast<float>(tmp(1));
		p.z = static_cast<float>(tmp(2));
		point_cloud.points.push_back(p);
	}
	pub_loop_point_cloud->publish(point_cloud);
}

/**
 * @brief: get margin point topic data from estimator
 */
void margin_point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg) {
	sensor_msgs::msg::PointCloud point_cloud;
	point_cloud.header = point_msg->header;
	for (auto point : point_msg->points) {
		cv::Point3f p_3d;
		p_3d.x = point.x;
		p_3d.y = point.y;
		p_3d.z = point.z;
		Eigen::Vector3d tmp = pose_graph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + pose_graph.t_drift;
		geometry_msgs::msg::Point32 p;
		p.x = static_cast<float>(tmp(0));
		p.y = static_cast<float>(tmp(1));
		p.z = static_cast<float>(tmp(2));
		point_cloud.points.push_back(p);
	}
	pub_loop_margin_cloud->publish(point_cloud);
}

/**
 * @brief: convert ROS Gray image to Opencv format
 */
cv::Mat getGrayImageFromMsg(const sensor_msgs::msg::Image::ConstPtr& img_msg) {
	cv_bridge::CvImageConstPtr ptr;
	if (img_msg->encoding == "8UC1") {
		sensor_msgs::msg::Image img;
		img.header = img_msg->header;
		img.height = img_msg->height;
		img.width = img_msg->width;
		img.is_bigendian = img_msg->is_bigendian;
		img.step = img_msg->step;
		img.data = img_msg->data;
		img.encoding = "mono8";
		ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	} else
		ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

	cv::Mat img = ptr->image.clone();
	return img;
}

/**
 * @brief: main process of loop fusion
 */
void process() {
	static int frame_index = 0;

	while (true) {
		sensor_msgs::msg::Image::ConstPtr image_msg = nullptr;
		sensor_msgs::msg::PointCloud::ConstPtr point_msg = nullptr;
		nav_msgs::msg::Odometry::ConstPtr pose_msg = nullptr;
		// find out the messages with same time stamp
		double time;
		m_buf.lock();
		if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty()) {
			// synchronical control
			if (image_buf.front()->header.stamp.sec + image_buf.front()->header.stamp.nanosec * (1e-9) >
			    pose_buf.front()->header.stamp.sec + pose_buf.front()->header.stamp.nanosec * (1e-9)) {
				pose_buf.pop();
				LOGGER_INFO("throw pose at beginning\n");
			} else if (image_buf.front()->header.stamp.sec + image_buf.front()->header.stamp.nanosec * (1e-9) >
			           point_buf.front()->header.stamp.sec + point_buf.front()->header.stamp.nanosec * (1e-9)) {
				point_buf.pop();
				LOGGER_INFO("throw point at beginning\n");
			} else if (image_buf.back()->header.stamp.sec + image_buf.back()->header.stamp.nanosec * (1e-9) >=
			               pose_buf.front()->header.stamp.sec + pose_buf.front()->header.stamp.nanosec * (1e-9) &&
			           point_buf.back()->header.stamp.sec + point_buf.back()->header.stamp.nanosec * (1e-9) >=
			               pose_buf.front()->header.stamp.sec + pose_buf.front()->header.stamp.nanosec * (1e-9)) {
				pose_msg = pose_buf.front();
				pose_buf.pop();
				while (!pose_buf.empty())
					pose_buf.pop();
				while (image_buf.front()->header.stamp.sec + image_buf.front()->header.stamp.nanosec * (1e-9) <
				       pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * (1e-9))
					image_buf.pop();
				image_msg = image_buf.front();
				image_buf.pop();

				while (point_buf.front()->header.stamp.sec + point_buf.front()->header.stamp.nanosec * (1e-9) <
				       pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * (1e-9))
					point_buf.pop();
				point_msg = point_buf.front();
				point_buf.pop();
			}
		}
		m_buf.unlock();

		if (pose_msg != nullptr) {
			// get iamge from image msg
			cv::Mat image = getGrayImageFromMsg(image_msg);
			// build keyframe
			Eigen::Vector3d T = Eigen::Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
			                                    pose_msg->pose.pose.position.z);
			Eigen::Matrix3d R = Eigen::Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
			                                       pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
			                        .toRotationMatrix();
			// just set every frame as keyframe
			if ((T - last_t).norm() > 0) {
				vector<cv::Point3f> point_3d;
				vector<cv::Point2f> point_2d_uv;
				vector<cv::Point2f> point_2d_normal;
				vector<double> point_id;

				for (unsigned int i = 0; i < point_msg->points.size(); i++) {
					cv::Point3f p_3d;
					p_3d.x = point_msg->points[i].x;
					p_3d.y = point_msg->points[i].y;
					p_3d.z = point_msg->points[i].z;
					point_3d.push_back(p_3d);

					cv::Point2f p_2d_uv, p_2d_normal;
					double p_id;
					p_2d_normal.x = point_msg->channels[i].values[0];
					p_2d_normal.y = point_msg->channels[i].values[1];
					p_2d_uv.x = point_msg->channels[i].values[2];
					p_2d_uv.y = point_msg->channels[i].values[3];
					p_id = point_msg->channels[i].values[4];
					point_2d_normal.push_back(p_2d_normal);
					point_2d_uv.push_back(p_2d_uv);
					point_id.push_back(p_id);
				}

				// build keyframe and add to pose graph
				auto* keyframe =
				    new KeyFrame(pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * (1e-9), frame_index, T,
				                 R, image, point_3d, point_2d_uv, point_2d_normal, point_id, sequence);

				pose_graph.addKeyFrame(keyframe, true);

				frame_index++;
				last_t = T;
			}
		}
		// sleep 5ms
		std::chrono::milliseconds dura(5);
		std::this_thread::sleep_for(dura);
	}
}

/**
 * @brief: loop fusion node code entrance
 */
int main(int argc, char* argv[]) {
	rclcpp::init(argc, argv);
	auto n = rclcpp::Node::make_shared("_", "loop_fusion");
	Logger::setLoggerLevel(ANSI::LogLevel::INFO);

	if (argc < 2) {
		LOGGER_INFO("please input: ros2 run loop_fusion loop_fusion_node [config file] ", "\n",
		            "for example: ros2 run loop_fusion loop_fusion_node "
		            "src/Flow-VINS/config/euroc/euroc_stereo_imu_config.yaml");
		return 0;
	}

	// read configuration file data
	readLoopFusionParameters(argv[1]);
	// set pose graph parameter
	pose_graph.setParameter();
	// register loop fusion publisher
	registerLoopFusionPub(n);

	// subscribe vio odometry topic
	auto sub_vio = n->create_subscription<nav_msgs::msg::Odometry>("/vio_estimator/odometry",
	                                                               rclcpp::QoS(rclcpp::KeepLast(2000)), vio_callback);

	// subscribe left image topic
	auto sub_image = n->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)),
	                                                                 image_callback);

	// subscribe key frame pose topic
	auto sub_pose = n->create_subscription<nav_msgs::msg::Odometry>("/vio_estimator/keyframe_pose",
	                                                                rclcpp::QoS(rclcpp::KeepLast(2000)), pose_callback);

	// subscribe extrinsic parameters topic
	auto sub_extrinsic = n->create_subscription<nav_msgs::msg::Odometry>(
	    "/vio_estimator/extrinsic", rclcpp::QoS(rclcpp::KeepLast(2000)), extrinsic_callback);

	// subscribe key frame point topic
	auto sub_point = n->create_subscription<sensor_msgs::msg::PointCloud>(
	    "/vio_estimator/keyframe_point", rclcpp::QoS(rclcpp::KeepLast(2000)), point_callback);

	// subscribe margin point cloud topic
	auto sub_margin_point = n->create_subscription<sensor_msgs::msg::PointCloud>(
	    "/vio_estimator/margin_cloud", rclcpp::QoS(rclcpp::KeepLast(2000)), margin_point_callback);

	// loop fusion thread
	std::thread measurement_process(process);

	rclcpp::spin(n);

	return 0;
}
