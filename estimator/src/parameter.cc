/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#include "../include/parameter.h"

namespace FLOW_VINS {

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
Eigen::Vector3d G{0.0, 0.0, 9.8};
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string SEMANTIC_TOPIC;
std::string RELO_TOPIC;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int SHOW_TRACK;
int FLOW_BACK;
double F_THRESHOLD;
int FREQ;
int USE_MAGNETOMETER;
int DEPTH;
double DEPTH_MIN_DIST;
double DEPTH_MAX_DIST;
int STATIC_INIT;
int MAX_SOLVE_CNT;

// semantic segmentation
int USE_SEGMENTATION;
int USE_GPU_ACC;

/**
 * YAML配置读取
 */
void readEstimatorParameters(const std::string& config_file) {
	// confirm that the configuration file path is correct
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if (!fsSettings.isOpened()) {
		LOGGER_WARN("ERROR: Wrong path to settings");
		assert(0);
	}
	// get camera ROS topic name
	fsSettings["image0_topic"] >> IMAGE0_TOPIC;
	fsSettings["image1_topic"] >> IMAGE1_TOPIC;
	// get image row and col
	ROW = fsSettings["image_height"];
	COL = fsSettings["image_width"];
	LOGGER_INFO("ROW: ", ROW, " COL: ", COL);
	//  get ceres solver parameters
	SOLVER_TIME = fsSettings["max_solver_time"];
	NUM_ITERATIONS = fsSettings["max_num_iterations"];
	MAX_SOLVE_CNT = fsSettings["max_solve_cnt"];
	// get min parallax
	MIN_PARALLAX = fsSettings["keyframe_parallax"];
	MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
	// set the odometry file save path
	fsSettings["output_path"] >> OUTPUT_FOLDER;
	VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
	LOGGER_INFO("result path ", VINS_RESULT_PATH);
	if (FILE* file = fopen(VINS_RESULT_PATH.c_str(), "r")) {
		if (remove(VINS_RESULT_PATH.c_str()) == 0)
			LOGGER_INFO("init result file success.");
	}
	// set multiply freq control
	FREQ = fsSettings["freq_ctrl_num"];
	// set the maximum number of the detected feature points and the distance between feature points
	MAX_CNT = fsSettings["max_cnt"];
	MIN_DIST = fsSettings["min_dist"];
	// set the fundmental matrix detect threshold
	F_THRESHOLD = fsSettings["F_threshold"];
	// set whether to show tracking image
	SHOW_TRACK = fsSettings["show_track"];
	// set whether to use reverse optical flow
	FLOW_BACK = fsSettings["flow_back"];
	// set whether to use semantic segmemtation
	USE_SEGMENTATION = fsSettings["use_segmentation"];
	// set whether to use GPU acceleration
	USE_GPU_ACC = fsSettings["use_gpu_acc"];
	// set whether to enable multithreading
	MULTIPLE_THREAD = fsSettings["multiple_thread"];
	// set whether to use IMU, timedelay between camera and IMU and whether to enable correcting online
	USE_IMU = fsSettings["imu"];
	LOGGER_INFO("USE_IMU: ", USE_IMU);
	TD = fsSettings["td"];
	ESTIMATE_TD = fsSettings["estimate_td"];
	ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];

	if (USE_IMU) {
		fsSettings["imu_topic"] >> IMU_TOPIC;
		LOGGER_INFO("IMU_TOPIC: ", IMU_TOPIC.c_str());
		ACC_N = fsSettings["acc_n"];
		ACC_W = fsSettings["acc_w"];
		GYR_N = fsSettings["gyr_n"];
		GYR_W = fsSettings["gyr_w"];
		G.z() = fsSettings["g_norm"];
	} else {
		ESTIMATE_EXTRINSIC = 0;
		ESTIMATE_TD = 0;
		LOGGER_INFO("no imu, fix extrinsic param; no time offset calibration");
	}

	if (ESTIMATE_TD)
		LOGGER_INFO("synchronized sensors, online estimate time offset, initial td: ", TD);

	else
		LOGGER_INFO("synchronized sensors, fix time offset: ", TD);
	if (ESTIMATE_EXTRINSIC == 1) {
		LOGGER_INFO("optimize extrinsic param around initial guess!");

		EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
	} else
		LOGGER_INFO("fix extrinsic param!");

	// get external parameter matrix from camera coordinates to body coordinates
	cv::Mat cv_T;
	fsSettings["body_T_cam0"] >> cv_T;
	Eigen::Matrix4d T;
	cv::cv2eigen(cv_T, T);
	RIC.emplace_back(T.block<3, 3>(0, 0));
	TIC.emplace_back(T.block<3, 1>(0, 3));
	// camera numbers(monocular / binocular)
	NUM_OF_CAM = fsSettings["num_of_cam"];
	LOGGER_INFO("camera number ", NUM_OF_CAM);

	if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
		LOGGER_INFO("num_of_cam should be 1 or 2");
		assert(0);
	}
	// get camera configuration file path
	int pn = static_cast<int>(config_file.find_last_of('/'));
	std::string config_path = config_file.substr(0, pn);
	std::string cam0_calib;
	fsSettings["cam0_calib"] >> cam0_calib;
	std::string cam0_path = config_path + "/" + cam0_calib;
	CAM_NAMES.emplace_back(cam0_path);

	if (NUM_OF_CAM == 2) {
		STEREO = 1;
		std::string cam1_calib;
		fsSettings["cam1_calib"] >> cam1_calib;
		std::string cam1_path = config_path + "/" + cam1_calib;
		CAM_NAMES.emplace_back(cam1_path);

		cv::Mat cv_T;
		fsSettings["body_T_cam1"] >> cv_T;
		Eigen::Matrix4d T;
		cv::cv2eigen(cv_T, T);
		RIC.emplace_back(T.block<3, 3>(0, 0));
		TIC.emplace_back(T.block<3, 1>(0, 3));
	}

	// direct initialization (only for RGB-D camera)
	STATIC_INIT = fsSettings["static_init"];
	DEPTH = fsSettings["depth"];
	if (DEPTH == 1) {
		cv::Mat cv_T;
		fsSettings["body_T_cam1"] >> cv_T;
		Eigen::Matrix4d T;
		cv::cv2eigen(cv_T, T);
		RIC.emplace_back(T.block<3, 3>(0, 0));
		TIC.emplace_back(T.block<3, 1>(0, 3));
		NUM_OF_CAM++;
	}

	INIT_DEPTH = 5.0;
	DEPTH_MIN_DIST = fsSettings["depth_min_dist"];
	DEPTH_MAX_DIST = fsSettings["depth_max_dist"];

	LOGGER_INFO("STEREO ", STEREO, "DEPTH", DEPTH);

	// get semantic segmentation topic from semantic node
	SEMANTIC_TOPIC = "/segment_node/semantic_mask";
	// get relocate topic from loop fusion node
	RELO_TOPIC = "/pose_graph/match_points";
	fsSettings.release();
}

} // namespace FLOW_VINS