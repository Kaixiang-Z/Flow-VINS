/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#pragma once

#include "../../utils/logger.h"
#include "../interface.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <vector>

namespace FLOW_VINS {

#define GPU_MODE 1

// vio estimator
const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
extern int MAX_SOLVE_CNT;
extern int USE_MAGNETOMETER;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string SEMANTIC_TOPIC;
extern std::string RELO_TOPIC;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern int FREQ;

extern int DEPTH;
extern double DEPTH_MIN_DIST;
extern double DEPTH_MAX_DIST;
extern int STATIC_INIT;
extern double F_THRESHOLD;

// semantic segmentation
extern int USE_SEGMENTATION;
extern int USE_GPU_ACC;

/**
 * @brief: set vio estimator parameters from config file
 */
void readEstimatorParameters(const std::string& config_file);

/**
 * @brief: ceres solver parameters dims
 */
enum SIZE_PARAMETERIZATION { SIZE_POSE = 7, SIZE_SPEEDBIAS = 9, SIZE_FEATURE = 1 };

/**
 * @brief: state vector order
 */
enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

/**
 * @brief: noise vector order
 */
enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };

} // namespace FLOW_VINS