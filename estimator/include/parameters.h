/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <vector>
#include <mutex>
#include <thread>
#include "../../thirdparty/CameraModel/camera_factory.h"
#include "../../thirdparty/CameraModel/cata_camera.h"
#include "../../thirdparty/CameraModel/pinhole_camera.h"
#include "common.h"

namespace FLOW_VINS {

// const parameters
const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
const int NUM_MARGIN_THREADS = 4;
const int MIN_LOOP_NUM = 25;

// flags
extern int MULTIPLE_THREAD;
extern int USE_SEGMENTATION;
extern int USE_GPU_ACC;
extern int USE_IMU;
extern int ESTIMATE_EXTRINSIC;
extern int ESTIMATE_TD;
extern int STEREO;
extern int DEPTH;
extern int STATIC_INIT;
extern int FIX_DEPTH;
extern int SHOW_TRACK;
extern int FLOW_BACK;

// imu parameters
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d LOOP_TIC;
extern Eigen::Matrix3d LOOP_QIC;
extern Eigen::Vector3d G;
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

// ceres paramters
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;

// ros topics
extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string IMU_TOPIC;
extern double TD;

// image parameters
extern int ROW, COL;
extern int NUM_OF_CAM;
extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int FREQ;
extern double DEPTH_MIN_DIST;
extern double DEPTH_MAX_DIST;
extern double F_THRESHOLD;

// file paths
extern std::string EX_CALIB_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string BRIEF_PATTERN_FILE;
extern std::string VINS_RESULT_PATH;
extern std::string VOCABULARY_PATH;
extern std::string SEGMENT_MODEL_FILE;

// mutex lock
extern std::mutex mutex_image;
extern std::mutex mutex_relocation;
extern std::mutex mutex_estimator;
extern std::mutex mutex_loopfusion;
extern std::mutex mutex_segment;
extern std::mutex mutex_keyframe;
extern std::mutex mutex_optimize;
extern std::mutex mutex_drift;

// thread
extern std::thread thread_estimator;
extern std::thread thread_loopfusion;
extern std::thread thread_segment;
extern std::thread thread_optimize;

/**
 * @brief: set vio estimator parameters from config file
 */
void readParameters(const std::string &config_file);

/**
 * @brief: ceres solver parameters dims
 */
enum SIZE_PARAMETERIZATION {
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

/**
 * @brief: state vector order
 */
enum StateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

/**
 * @brief: noise vector order
 */
enum NoiseOrder {
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

} // namespace FLOW_VINS