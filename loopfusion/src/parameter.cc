/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#include "../include/parameter.h"

namespace FLOW_VINS {

// loop fusion
int USE_IMU;
std::string IMAGE_TOPIC;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
std::string BRIEF_PATTERN_FILE;
std::string LOOP_FUSION_RESULT_PATH;
std::string VOCABULARY_PATH;
std::string CAMERA_PATH;
Eigen::Vector3d LOOP_TIC;
Eigen::Matrix3d LOOP_QIC;

/**
 * YAML配置读取
 */
void readLoopFusionParameters(const std::string& config_file) {
	// confirm that the configuration file path is correct
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if (!fsSettings.isOpened()) {
		LOGGER_ERROR("ERROR: Wrong path to settings");
		assert(0);
	}

	// loop fusion config parameter
	VISUALIZATION_SHIFT_X = 0;
	VISUALIZATION_SHIFT_Y = 0;

	// set brief config file
	std::filesystem::path file_path = config_file;
	std::filesystem::path support_file_folder = file_path.parent_path();
	std::string support_file_path = support_file_folder.string() + "/../support";
	VOCABULARY_PATH = support_file_path + "/brief_k10L6.bin";
	LOGGER_INFO("vocabulary file : ", VOCABULARY_PATH);
	BRIEF_PATTERN_FILE = support_file_path + "/brief_pattern.yml";
	LOGGER_INFO("brief pattern file : ", BRIEF_PATTERN_FILE);

	// set camera config parameters
	int pn = static_cast<int>(config_file.find_last_of('/'));
	std::string config_path = config_file.substr(0, pn);
	std::string cam0_calib;
	fsSettings["cam0_calib"] >> cam0_calib;
	CAMERA_PATH = config_path + "/" + cam0_calib;
	LOGGER_INFO("cam calib path: ", CAMERA_PATH.c_str());

	fsSettings["image0_topic"] >> IMAGE_TOPIC;
	fsSettings["output_path"] >> LOOP_FUSION_RESULT_PATH;

	// set loop fusion result file output path
	LOOP_FUSION_RESULT_PATH = LOOP_FUSION_RESULT_PATH + "/vio_loop.csv";
	if (FILE* file = fopen(LOOP_FUSION_RESULT_PATH.c_str(), "r"))
		if (remove(LOOP_FUSION_RESULT_PATH.c_str()) == 0) {
			LOGGER_INFO("remove loop result file success.");
		}
	// check if use IMU
	USE_IMU = fsSettings["imu"];
	fsSettings.release();
}

} // namespace FLOW_VINS
