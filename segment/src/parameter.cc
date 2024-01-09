/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#include "../include/parameter.h"

namespace FLOW_VINS {

std::string IMAGE_TOPIC;

void readSemanticParameters(const std::string& config_file) {
	// confirm that the configuration file path is correct
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if (!fsSettings.isOpened()) {
		std::cerr << "ERROR: Wrong path to settings" << std::endl;
	}
	// get image topic
	fsSettings["image0_topic"] >> IMAGE_TOPIC;
	fsSettings.release();
}
} // namespace FLOW_VINS