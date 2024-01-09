/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: estimator node
 */

#include "include/estimator.h"
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <unistd.h>

using namespace FLOW_VINS;

static Estimator estimator;

static std::queue<sensor_msgs::msg::Image::ConstPtr> img0_buf;
static std::queue<sensor_msgs::msg::Image::ConstPtr> img1_buf;
static std::queue<sensor_msgs::msg::Image::ConstPtr> semantic_buf;
static std::mutex m_buf;

/**
 * @brief: subscribe IMU topic data and sent it to estimator
 */
void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
	double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
	double dx = imu_msg->linear_acceleration.x;
	double dy = imu_msg->linear_acceleration.y;
	double dz = imu_msg->linear_acceleration.z;
	double rx = imu_msg->angular_velocity.x;
	double ry = imu_msg->angular_velocity.y;
	double rz = imu_msg->angular_velocity.z;

	Eigen::Vector3d acc(dx, dy, dz);
	Eigen::Vector3d gyr(rx, ry, rz);

	estimator.inputImu(t, acc, gyr);
}

/**
 * @brief: subscribe left image topic data and sent it to estimator
 */
void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
	m_buf.lock();
	img0_buf.emplace(img_msg);
	m_buf.unlock();
}

/**
 * @brief: subscribe right image of depth image topic data and sent it to estimator
 */
void rightImageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
	m_buf.lock();
	img1_buf.emplace(img_msg);
	m_buf.unlock();
}

/**
 * @brief: subscribe semantic image topic data and sent it to estimator
 */
void semanticImageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
	m_buf.lock();
	semantic_buf.emplace(img_msg);
	m_buf.unlock();
}

/**
 * @brief: check the semantic image is available
 */
bool semanticAvailable(double t) {
	if (!semantic_buf.empty()) {
		if (t == semantic_buf.front()->header.stamp.sec + semantic_buf.front()->header.stamp.nanosec * (1e-9)) {
			return true;
		} else {
			m_buf.lock();
			semantic_buf.pop();
			m_buf.unlock();
			return false;
		}
	} else
		return false;
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
 * @brief: convert ROS depth image to Opencv format
 */
cv::Mat getDepthImageFromMsg(const sensor_msgs::msg::Image::ConstPtr& img_msg) {
	cv::Mat depth_img;
	if (img_msg->encoding == "mono16" || img_msg->encoding == "16UC1") {
		depth_img = cv_bridge::toCvShare(img_msg)->image;
	} else if (img_msg->encoding == "32FC1") {
		depth_img = cv_bridge::toCvShare(img_msg)->image;
		depth_img.convertTo(depth_img, CV_16UC1, 1000);
	} else {
		LOGGER_ERROR("Unknown depth encoding!");
	}
	cv::medianBlur(depth_img, depth_img, 5);
	return depth_img;
}

/**
 * @brief: extract image from image buffer and sent it to estimator
 */
void process() {
	while (true) {
		if (STEREO || DEPTH) {
			cv::Mat img0, img1, mask;
			double time = 0;
			if (!img0_buf.empty() && !img1_buf.empty()) {
				double time0 = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
				double time1 = img1_buf.front()->header.stamp.sec + img1_buf.front()->header.stamp.nanosec * (1e-9);

				m_buf.lock();
				// binoculer image time delay less than 0.003s
				if (time0 < time1 - 0.003) {
					img0_buf.pop();
					m_buf.unlock();
				} else if (time0 > time1 + 0.003) {
					img1_buf.pop();
					m_buf.unlock();
				} else {
					// extract the oldest frame from queue and dequeue it
					time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
					img0 = getGrayImageFromMsg(img0_buf.front());
					img0_buf.pop();
					if (DEPTH) {
						// get depth image
						img1 = getDepthImageFromMsg(img1_buf.front());
						img1_buf.pop();
					} else {
						// get right image
						img1 = getGrayImageFromMsg(img1_buf.front());
						img1_buf.pop();
					}
					m_buf.unlock();
					if (USE_SEGMENTATION) {
						while (!semanticAvailable(time)) {
							LOGGER_INFO("waiting for semantic ...");
							std::chrono::milliseconds dura(5);
							std::this_thread::sleep_for(dura);
						}
						m_buf.lock();
						// get semantic image
						mask = getGrayImageFromMsg(semantic_buf.front());
						semantic_buf.pop();
						m_buf.unlock();
					}
				}

				if (!img0.empty()) {
					estimator.inputImage(time, img0, img1, mask);
				}
			}

		} else {
			cv::Mat img;
			double time = 0;
			if (!img0_buf.empty()) {
				// extract the oldest frame from queue and dequeue it
				m_buf.lock();
				time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
				img = getGrayImageFromMsg(img0_buf.front());
				img0_buf.pop();
				m_buf.unlock();
			}
			if (!img.empty()) {
				estimator.inputImage(time, img);
			}
		}
		std::chrono::milliseconds dura(2);
		std::this_thread::sleep_for(dura);
	}
}

/**
 * @brief: vio system code entrance
 */
int main(int argc, char* argv[]) {

	rclcpp::init(argc, argv);
	auto n = rclcpp::Node::make_shared("_", "vio_estimator");
	Logger::setLoggerLevel(ANSI::LogLevel::DEBUG);

	if (argc < 2) {
		LOGGER_INFO("please input: ros2 run estimator system_node [config file] \n",
		            "for example: ros2 run estimator system_node ",
		            "src/Hyper_Vins/config/euroc_stereo_imu_config.yaml ");
		return 1;
	}

	// read configuration file data
	readEstimatorParameters(argv[1]);
	// set estimator parameter
	estimator.setParameter();
	// register estimator publisher
	registerEstimatorPub(n);
	// subscribe IMU topic
	rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu = nullptr;
	if (USE_IMU)
		sub_imu =
		    n->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), imuCallback);
	// subscribe left image
	auto sub_img0 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE0_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)),
	                                                                leftImageCallback);
	// subscribe right image
	auto sub_img1 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE1_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)),
	                                                                rightImageCallback);
	// subscribe semantic mask
	auto sub_semantic_mask = n->create_subscription<sensor_msgs::msg::Image>(
	    SEMANTIC_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), semanticImageCallback);

	// sync process
	std::thread sync_thread(process);
	rclcpp::spin(n);

	return 0;
}
