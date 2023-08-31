/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: global parameters
 */

#include "../include/Parameters.h"

namespace FLOW_VINS {

// flags
int MULTIPLE_THREAD;
int USE_SEGMENTATION;
int USE_MAGNETOMETER;
int USE_GPU_ACC;
int USE_IMU;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int STEREO;
int DEPTH;
int STATIC_INIT;
int FIX_DEPTH;
int SHOW_TRACK;
int FLOW_BACK;

// imu parameters
vector<Eigen::Matrix3d> RIC;
vector<Eigen::Vector3d> TIC;
Eigen::Vector3d LOOP_TIC;
Eigen::Matrix3d LOOP_QIC;
Eigen::Vector3d G{0.0, 0.0, 9.8};
double ACC_N, ACC_W;
double GYR_N, GYR_W;

// ceres paramters
double SOLVER_TIME;
int NUM_ITERATIONS;
int MAX_SOLVE_CNT;

// ros topics
string IMAGE0_TOPIC, IMAGE1_TOPIC;
string IMU_TOPIC;
double TD;

// image parameters
int ROW, COL;
int NUM_OF_CAM;
double INIT_DEPTH;
double MIN_PARALLAX;
vector<string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int FREQ;
double DEPTH_MIN_DIST;
double DEPTH_MAX_DIST;
double F_THRESHOLD;

// file paths
string EX_CALIB_RESULT_PATH;
string OUTPUT_FOLDER;
string BRIEF_PATTERN_FILE;
string VINS_RESULT_PATH;
string VOCABULARY_PATH;
string SEGMENT_MODEL_FILE;

// mutex lock
mutex mutex_image;
mutex mutex_relocation;
mutex mutex_estimator;
mutex mutex_loopfusion;
mutex mutex_segment;
mutex mutex_keyframe;
mutex mutex_optimize;
mutex mutex_drift;

// thread
thread thread_estimator;
thread thread_loopfusion;
thread thread_segment;
thread thread_optimize;

void readParameters(const string &config_file) {
    // confirm that the configuration file path is correct
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        ROS_WARN("ERROR: Wrong path to settings");
        ROS_BREAK();
    }
    // get camera ROS topic name
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    // get image row and col
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    cout << "ROW: " << ROW << " COL: " << COL << endl;
    // get ceres solver parameters
    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MAX_SOLVE_CNT = fsSettings["max_solve_cnt"];
    // get min parallax
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
    // set the odometry file save path
    fsSettings["output_path"] >> OUTPUT_FOLDER;
    // set loop fusion result file output path
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio_loop.txt";
    if (FILE *file = fopen(VINS_RESULT_PATH.c_str(), "r"))
        if (remove(VINS_RESULT_PATH.c_str()) == 0) {
            cout << "remove loop result file success." << endl;
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
    // set whether to use magnetometer
    USE_MAGNETOMETER = fsSettings["use_magnetometer"];
    // set whether to use GPU acceleration
    USE_GPU_ACC = fsSettings["use_gpu_acc"];
    // set whether to enable multithreading
    MULTIPLE_THREAD = fsSettings["multiple_thread"];
    // set whether to use IMU, timedelay between camera and IMU and whether to enable correcting online
    USE_IMU = fsSettings["imu"];
    cout << "USE_IMU: " << USE_IMU << endl;
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];

    if (USE_IMU) {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        cout << "IMU_TOPIC: " << IMU_TOPIC.c_str() << endl;
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    } else {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        cout << "no imu, fix extrinsic param; no time offset calibration" << endl;
    }

    if (ESTIMATE_TD)
        cout << "synchronized sensors, online estimate time offset, initial td: " << TD << endl;
    else
        cout << "synchronized sensors, fix time offset: " << TD << endl;
    if (ESTIMATE_EXTRINSIC == 1) {
        cout << "optimize extrinsic param around initial guess!" << endl;
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    } else
        cout << "fix extrinsic param " << endl;

    // get external parameter matrix from camera coordinates to body coordinates
    cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    RIC.emplace_back(T.block<3, 3>(0, 0));
    TIC.emplace_back(T.block<3, 1>(0, 3));
    // camera numbers(monocular / binocular)
    NUM_OF_CAM = fsSettings["num_of_cam"];
    cout << "camera number " << NUM_OF_CAM << endl;
    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
        cout << "num_of_cam should be 1 or 2" << endl;
        assert(0);
    }
    // get camera configuration file path
    int pn = static_cast<int>(config_file.find_last_of('/'));
    string config_path = config_file.substr(0, pn);
    string cam0_calib;
    fsSettings["cam0_calib"] >> cam0_calib;
    string cam0_path = config_path + "/" + cam0_calib;
    CAM_NAMES.emplace_back(cam0_path);

    if (NUM_OF_CAM == 2) {
        STEREO = 1;
        string cam1_calib;
        fsSettings["cam1_calib"] >> cam1_calib;
        string cam1_path = config_path + "/" + cam1_calib;
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

    cout << "STEREO: " << STEREO << " DEPTH: " << DEPTH << endl;
    // set whether to fix depth in optimization
    if (!fsSettings["fix_depth"].empty())
        FIX_DEPTH = fsSettings["fix_depth"];
    else
        FIX_DEPTH = 1;

    // set brief config file
    string pkg_path = ros::package::getPath("vio_system");
    VOCABULARY_PATH = pkg_path + "/config/brief/brief_k10L6.bin";
    cout << "vocabulary file : " << VOCABULARY_PATH << endl;

    BRIEF_PATTERN_FILE = pkg_path + "/config/brief/brief_pattern.yml";
    cout << "brief pattern file : " << BRIEF_PATTERN_FILE << endl;

    SEGMENT_MODEL_FILE = pkg_path + "/config/semantic/yolov8n-seg.engine";
    cout << "segment model file : " << SEGMENT_MODEL_FILE << endl;

    fsSettings.release();
}

} // namespace FLOW_VINS