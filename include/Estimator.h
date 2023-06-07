/**
 * @Author: Zhang Kaixiang
 * @Date: 2022-12-21 17:55:38
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Descripttion: 位姿估计
 */

#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <thread>
#include <unordered_map>
#include "Parameters.h"
#include "Publisher.h"
#include "Common.h"
#include "Feature.h"
#include "Initialize.h"
#include "FactorGraph.h"
#include "LoopFusion.h"
#include "Segment.h"

namespace FLOW_VINS {
/**
 * @brief: class for ceres manifold optimization 
 */
class ManifoldParameterization : public ceres::Manifold {
    /**
     * @brief: pose and orientation states addition on manifold
     */
    bool Plus(const double *x, const double *delta,
              double *x_plus_delta) const override;

    /**
     * @brief: compute jacobian matrix for input x plus operation
     */
    bool PlusJacobian(const double *x, double *jacobian) const override;

    /**
     * @brief: dimension of the ambient space in which the manifold is embedded: x, y, z, q0, q1, q2, q3 
     */
    int AmbientSize() const override {
        return 7;
    }

    /**
     * @brief: dimension of the manifold/tangent space: x, y, z, q1, q2, q3
     */
    int TangentSize() const override {
        return 6;
    }

    /**
     * @brief: implements minus operation for the manifold
     */
    bool Minus(const double *y, const double *x, double *y_minus_x) const override {
        return true;
    }

    /**
     * @brief: compute jacobian matrix for input x minus operation
     */
    bool MinusJacobian(const double *x, double *jacobian) const override {
        return true;
    }
};

/**
 * @brief: class for vio estimator
 */
class Estimator {
public:
    /**
     * @brief: estimator construct function, state parameters clear
     */
    Estimator();

    /**
     * @brief: estimator destructor function, waiting for the process thread to be released
     */
    ~Estimator();

    /**
     * @brief: clear estimator parameter and state, the function will be transfered when system reboot or failure detected
     */
    void clearState();

    /**
     * @brief: set estimator parameters from configuration file
     */
    void setParameter();

    /**
     * @brief: input IMU data, then fast predict and update latest PQV state
     */
    void inputImu(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

    /**
     * @brief: input image data, feature tracking and fill the feature into feature buffer
     */
    void inputImage(double t, const cv::Mat &_img,
                    const cv::Mat &_img1 = cv::Mat(),
                    const cv::Mat &_mask = cv::Mat());

    /**
     * @brief: imput relocate frame match points, used for relocate
     */
    void inputReloFrame(double frame_stamp, int frame_index,
                        vector<Vector3d> &match_points, Vector3d relo_t,
                        Matrix3d relo_r);

    /**
     * @brief: main process of IMU pre-integration, use midian integration estimate PQV state
     */
    void processIMU(double t, double dt,
                    const Vector3d &linear_acceleration,
                    const Vector3d &angular_velocity);

    /**
     * @brief: main process of VIO system, initial and backend solver
     */
    void processImage(FeatureFrame &image, double header);

    /**
     * @brief: transfer process IMU and process Image functions, publish relate topic data
     */
    void processMeasurements();

    /**
     * @brief: initialize the slam system 
     */
    void initialize(const double &header);

    /**
     * @brief: main process of initialization 
     */
    bool initialStructure();

    /**
     * @brief: initialize first IMU frame pose, alignment first acceleration with gravity vector to get initial rotation
     */
    void initFirstImuPose(vector<pair<double, Vector3d>> &acc_vector);

    /**
     * @brief: check if IMU data is available
     */
    bool ImuAvailable(double t);

    /**
     * @brief: from the IMU data queue, extract the data of the time period (t0, t1)
     */
    bool getImuInterval(double t0, double t1,
                        vector<pair<double, Vector3d>> &acc_vector,
                        vector<pair<double, Vector3d>> &gyr_vector);

    /**
     * @brief: check IMU observibility
     */
    bool checkImuObservibility();

    /**
     * @brief: build sfm_feature for SFM 
     */
    void buildSfmFeature(vector<SFMFeature> &sfm_feature);

    /**
     * @brief: find base frame l in sliding windows and get relative rotation and translation between frame l and the newest frame
     */
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    /**
     * @brief: recover Matrix R and T from two frames match points
     */
    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &rotation, Vector3d &translation);

    /**
     * @brief: get rotation and translation for all frame and 3D coordinates of all features in frame l without scaler
     */
    bool solvePnPForAllFrame(Quaterniond Q[], Vector3d T[], map<int, Vector3d> &sfm_tracked_points);

    /**
     * @brief: loosely coupled IMU-visual initialization 
     */
    bool visualInitialAlign();

    /**
     * @brief: get index frame P and Q in world coordinate
     */
    void getPoseInWorldFrame(int index, Matrix4d &T);

    /**
     * @brief: predict feature points in next frame to get better feature track
     */
    void predictPtsInNextFrame();

    /**
     * @brief: main process of vio estimator backend
     */
    void backend();

    /**
     * @brief: main process of backend
     */
    void solveOdometry();

    /**
     * @brief: main process of solveOdometry
     */
    void optimization();

    /**
     * @brief: nonlinear optimization problem construct
     */
    void nonLinearOptimization(ceres::Problem &problem, ceres::LossFunction *loss_function);

    /**
     * @brief: change status values from vector to double in favor of ceres
     */
    void vector2double();

    /**
     * @brief: recover status values from double to vector 
     */
    void double2vector();

    /**
     * @brief: detect failure 
     */
    bool failureDetection();

    /**
     * @brief: marginizate old frames for big Hessian matrix 
     */
    void margOld(ceres::LossFunction *loss_function);

    /**
     * @brief: marginizate second new frames for big Hessian matrix 
     */
    void margNew();

    /**
     * @brief: main process of sliding window for status value 
     */
    void slideWindow();

    /**
     * @brief: marginizate new frames
     */
    void slideWindowNew();

    /**
     * @brief: marginizate old frames
     */
    void slideWindowOld();

    /**
     * @brief: calculate reprojection error
     */
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                             Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                             double depth, Vector3d &uvi, Vector3d &uvj);

    /**
     * @brief: calculate reprojection error in 3D
     */
    double reprojectionError3D(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                               Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                               double depth, Vector3d &uvi, Vector3d &uvj);

    /**
     * @brief: main process of moving consistency check 
     */
    void movingConsistencyCheck(set<int> &remove_index);

    /**
     * @brief: convert estimator data to loop fusion data
     */
    void loopFusionDataConvert();

    /**
     * @brief: if image is unstable, start a new sequence 
     */
    void newSequence();

    /**
    * @brief: main process of loop fusion 
    */
    void processLoopFusion();

    /**
     * @brief: ceres solver flag
     */
    enum SolverFlag {
        INITIAL,
        NON_LINEAR
    };

    /**
     * @brief: sliding window marginalization flag
     */
    enum MarginalizationFlag {
        MARGIN_OLD,
        MARGIN_SECOND_NEW
    };

    // common data structures
    queue<pair<double, Vector3d>> acc_buf;
    queue<pair<double, Vector3d>> gyr_buf;
    queue<pair<double, FeatureFrame>> feature_buf;
    double prev_time{}, cur_time{};
    bool open_ex_estimation{};

    // common class in vio estimator system
    FeatureManager feature_manager;
    //Backend backend;

    int frame_count{};
    int image_count{};
    // IMU to camera external matrix
    Matrix3d ric[2];
    Vector3d tic[2];
    Vector3d g;

    // IMU coordinate data in sliding window
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td{};

    // backup last marginalization the oldest pose data, last pose data and the newest pose data in sliding window
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double headers[(WINDOW_SIZE + 1)]{};

    // IMU pre-integration variable
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration{};
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]{};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // frame position in sliding window and class for marginalization
    vector<Vector3d> key_poses;
    double initial_timestamp{};
    MarginalizationInfo *last_marginalization_info{};
    vector<double *> last_marginalization_parameter_blocks;

    // parameters in the sliding windows stored in array, used for ceres optimize
    double para_pose[WINDOW_SIZE + 1][SIZE_POSE]{};
    double para_speed_bias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]{};
    double para_feature[NUM_OF_F][SIZE_FEATURE]{};
    double para_ex_pose[2][SIZE_POSE]{};
    double para_td[1][1]{};

    vector<int> param_feature_id;
    map<int, int> param_feature_id_to_index;

    // state flag
    bool init_first_pose_flag{};
    bool init_thread_flag;
    bool first_imu_flag{};
    bool failure_occur_flag{};
    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;

    // relocalization variable
    bool relocalization_info{};
    double relo_frame_stamp{};
    double relo_frame_index{};
    int relo_frame_local_index{};
    vector<Vector3d> relo_match_points;
    double relo_pose[SIZE_POSE]{};
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw{};

    // for loop fusion
    queue<pair<double, cv::Mat>> loop_image_buf;
    queue<pair<double, Matrix4d>> loop_pose_buf;
    queue<pair<double, PointCloud>> loop_point_buf;

    PoseGraph pose_graph;
    int frame_index = 0;
    int sequence = 1;
    Vector3d last_t{-100, -100, -100};
    double last_image_time = -1;

    // for segmentation
    YOLOv8_seg yolo;
};
} // namespace FLOW_VINS