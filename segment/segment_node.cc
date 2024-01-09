/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-11 09:18:27
 * @Description: semantic segmentation node
 */

#include "include/parameter.h"
#include "include/segment.h"
#include <iostream>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <unistd.h>

using namespace FLOW_VINS;

// semantic segmentation
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_semantic;
rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image;

static std::queue<sensor_msgs::msg::Image::ConstPtr> image_buf;
static std::mutex m_buf;

std::string model_path = "/home/zkx/Work/Slam/HyperVins/src/HyperVins/config/support/yolov8n-seg.engine";
YOLOv8_seg yolo(model_path);

/**
 * @brief: subscribe image topic data and sent it to estimator
 */
void imageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
	m_buf.lock();
	image_buf.push(img_msg);
	m_buf.unlock();
}

/**
 * @brief: publish image topic data and sent it to estimator
 */
void pubSemanticImage(const cv::Mat& img, const std_msgs::msg::Header& header) {
	sensor_msgs::msg::Image::SharedPtr SemanticMsg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();
	pub_semantic->publish(*SemanticMsg);
}

/**
 * @brief: convert ROS RGB image to OpenCV format
 */
cv::Mat getRgbImageFromMsg(const sensor_msgs::msg::Image::ConstPtr& img_msg) {
	cv_bridge::CvImageConstPtr ptr;
	sensor_msgs::msg::Image img;
	img.header = img_msg->header;
	img.height = img_msg->height;
	img.width = img_msg->width;
	img.is_bigendian = img_msg->is_bigendian;
	img.step = img_msg->step;
	img.data = img_msg->data;
	img.encoding = sensor_msgs::image_encodings::BGR8;
	ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
	cv::Mat semantic_img = ptr->image.clone();
	return semantic_img;
}

/**
 * @brief: main process of semantic segmentation
 */
void process() {
	while (true) {
		cv::Mat img, mask;
		cv::Size size = cv::Size{640, 640};
		std::vector<Object> objs;
		std_msgs::msg::Header header;
		// extract the oldest frame and dequeue it
		m_buf.lock();
		if (!image_buf.empty()) {
			header = image_buf.front()->header;
			img = getRgbImageFromMsg(image_buf.front());
			image_buf.pop();
		}
		m_buf.unlock();
		// semantic segmentation process and publish semantic mask image
		if (!img.empty()) {

			objs.clear();
			yolo.copyFromMat(img, size);
			yolo.inference();
			yolo.postProcess(objs);
			yolo.drawObjects(img, mask, objs);

			pubSemanticImage(mask, header);
		}
	}
}

int main(int argc, char* argv[]) {
	rclcpp::init(argc, argv);
	auto n = rclcpp::Node::make_shared("_", "segment_node");

	if (argc < 2) {
		std::cout << "please input: ros2 run segment segment_node [config file] \n"
		             " for example: ros2 run segment segment_node src/Hyper_Vins/config/tum/tum_mono_config.yaml"
		          << std::endl;
		return 1;
	}
	std::cout << "semantic segmentation process" << std::endl;

	readSemanticParameters(argv[1]);
	yolo.makePipe();
	// subscribe RGB-D image
	sub_image =
	    n->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), imageCallback);
	// publish segmentation image
	pub_semantic = n->create_publisher<sensor_msgs::msg::Image>("semantic_mask", 1000);

	std::thread semantic_thread(process);
	rclcpp::spin(n);

	return 0;
}