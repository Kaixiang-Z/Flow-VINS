/**
 * @Author: Zhang Kaixiang
 * @Date: 2022-12-21 17:55:38
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Descripttion: 位姿估计
 */

#pragma once

#include "factor.h"
#include "feature.h"
#include "initialize.h"
#include "kalmanfilter.h"
#include "parameter.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <thread>
#include <unordered_map>

namespace FLOW_VINS {
/**
 * @brief: class for ceres manifold optimization
 */
class ManifoldParameterization : public ceres::Manifold {
	/**
	 * @brief: pose and orientation states addition on manifold
	 */
	bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;

	/**
	 * @brief: compute jacobian matrix for input x plus operation
	 */
	bool PlusJacobian(const double* x, double* jacobian) const override;

	/**
	 * @brief: dimension of the ambient space in which the manifold is embedded: x, y, z, q0, q1, q2, q3
	 */
	int AmbientSize() const override { return 7; }

	/**
	 * @brief: dimension of the manifold/tangent space: x, y, z, q1, q2, q3
	 */
	int TangentSize() const override { return 6; }

	/**
	 * @brief: implements minus operation for the manifold
	 */
	bool Minus(const double* y, const double* x, double* y_minus_x) const override { return true; }

	/**
	 * @brief: compute jacobian matrix for input x minus operation
	 */
	bool MinusJacobian(const double* x, double* jacobian) const override { return true; }
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
	 * @brief: clear estimator parameter and state, the function will be transfered when system reboot or failure
	 * detected
	 */
	void clearState();

	/**
	 * @brief: set estimator parameters from configuration file
	 */
	void setParameter();

	/**
	 * @brief: input IMU data, then fast predict and update latest PQV state
	 */
	void inputImu(double t, const Eigen::Vector3d& linear_acceleration, const Eigen::Vector3d& angular_velocity);

	/**
	 * @brief: input AHRS data, include magnetometer, then fast predict and update latest PQV state
	 */
	void inputAhrs(double t, const Eigen::Vector3d& linear_acceleration, const Eigen::Vector3d& angular_velocity,
	               const Eigen::Vector3d& magnetometer);

	/**
	 * @brief: input image data, feature tracking and fill the feature into feature buffer
	 */
	void inputImage(double t, const cv::Mat& _img, const cv::Mat& _img1 = cv::Mat(), const cv::Mat& _mask = cv::Mat());

	/**
	 * @brief: main process of IMU pre-integration, use midian integration estimate PQV state
	 */
	void processIMU(double t, double dt, const Eigen::Vector3d& linear_acceleration,
	                const Eigen::Vector3d& angular_velocity);

	/**
	 * @brief: main process of VIO system, initial and backend solver
	 */
	void processImage(FeatureFrame& image, double header);

	/**
	 * @brief: transfer process IMU and process Image functions, publish relate topic data
	 */
	void processMeasurements();

	/**
	 * @brief: initialize the slam system
	 */
	void initialize(const double& header);

	/**
	 * @brief: main process of initialization
	 */
	bool initialStructure();

	/**
	 * @brief: initialize first IMU frame pose, alignment first acceleration with gravity vector to get initial rotation
	 */
	void initFirstImuPose(std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector);

	/**
	 * @brief: initialize first AHRS frame pose, alignment first acceleration with gravity vector to get initial
	 * rotation
	 */
	void initFirstAhrsPose(std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
	                       std::vector<std::pair<double, Eigen::Vector3d>>& mag_vector);

	/**
	 * @brief: check if IMU data is available
	 */
	bool ImuAvailable(double t);

	/**
	 * @brief: from the IMU data queue, extract the data of the time period (t0, t1)
	 */
	bool getImuInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
	                    std::vector<std::pair<double, Eigen::Vector3d>>& gyr_vector);

	/**
	 * @brief: from the AHRS data queue, extract the data of the time period (t0, t1)
	 */
	bool getAhrsInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
	                     std::vector<std::pair<double, Eigen::Vector3d>>& gyr_vector,
	                     std::vector<std::pair<double, Eigen::Vector3d>>& mag_vector);

	/**
	 * @brief: check IMU observibility
	 */
	bool checkImuObservibility();

	/**
	 * @brief: build sfm_feature for SFM
	 */
	void buildSfmFeature(std::vector<SFMFeature>& sfm_feature);

	/**
	 * @brief: find base frame l in sliding windows and get relative rotation and translation between frame l and the
	 * newest frame
	 */
	bool relativePose(Eigen::Matrix3d& relative_R, Eigen::Vector3d& relative_T, int& l);

	/**
	 * @brief: recover Matrix R and T from two frames match points
	 */
	bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& corres,
	                     Eigen::Matrix3d& rotation, Eigen::Vector3d& translation);

	/**
	 * @brief: get rotation and translation for all frame and 3D coordinates of all features in frame l without scaler
	 */
	bool solvePnPForAllFrame(Eigen::Quaterniond Q[], Eigen::Vector3d T[],
	                         std::map<int, Eigen::Vector3d>& sfm_tracked_points);

	/**
	 * @brief: loosely coupled IMU-visual initialization
	 */
	bool visualInitialAlign();

	/**
	 * @brief: get index frame P and Q in world coordinate
	 */
	void getPoseInWorldFrame(int index, Eigen::Matrix4d& T);

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
	void nonLinearOptimization(ceres::Problem& problem, ceres::LossFunction* loss_function);

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
	void margOld(ceres::LossFunction* loss_function);

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
	double reprojectionError(Eigen::Matrix3d& Ri, Eigen::Vector3d& Pi, Eigen::Matrix3d& rici, Eigen::Vector3d& tici,
	                         Eigen::Matrix3d& Rj, Eigen::Vector3d& Pj, Eigen::Matrix3d& ricj, Eigen::Vector3d& ticj,
	                         double depth, Eigen::Vector3d& uvi, Eigen::Vector3d& uvj);

	/**
	 * @brief: calculate reprojection error in 3D
	 */
	double reprojectionError3D(Eigen::Matrix3d& Ri, Eigen::Vector3d& Pi, Eigen::Matrix3d& rici, Eigen::Vector3d& tici,
	                           Eigen::Matrix3d& Rj, Eigen::Vector3d& Pj, Eigen::Matrix3d& ricj, Eigen::Vector3d& ticj,
	                           double depth, Eigen::Vector3d& uvi, Eigen::Vector3d& uvj);

	/**
	 * @brief: main process of moving consistency check
	 */
	void movingConsistencyCheck(std::set<int>& remove_index);

	/**
	 * @brief: ceres solver flag
	 */
	enum SolverFlag { INITIAL, NON_LINEAR };

	/**
	 * @brief: sliding window marginalization flag
	 */
	enum MarginalizationFlag { MARGIN_OLD, MARGIN_SECOND_NEW };

	// common data structures
	std::thread process_thread;
	std::mutex mutex_process;
	std::mutex mutex_buf;
	std::mutex mutex_propagate;
	std::queue<std::pair<double, Eigen::Vector3d>> acc_buf;
	std::queue<std::pair<double, Eigen::Vector3d>> gyr_buf;
	std::queue<std::pair<double, Eigen::Vector3d>> mag_buf;
	std::queue<std::pair<double, FeatureFrame>> feature_buf;
	double prev_time{}, cur_time{};
	bool open_ex_estimation{};

	// common class in vio estimator system
	FeatureManager feature_manager;
	// Backend backend;
	ESKF_Attitude eskf_estimator;

	int frame_count{};
	int image_count{};
	// IMU to camera external matrix
	Eigen::Matrix3d ric[2];
	Eigen::Vector3d tic[2];
	Eigen::Vector3d g;

	// IMU coordinate data in sliding window
	Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
	Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
	double td{};

	// backup last marginalization the oldest pose data, last pose data and the newest pose data in sliding window
	Eigen::Matrix3d back_R0, last_R, last_R0;
	Eigen::Vector3d back_P0, last_P, last_P0;
	double headers[(WINDOW_SIZE + 1)]{};

	// IMU pre-integration variable
	std::map<double, ImageFrame> all_image_frame;
	IntegrationBase* tmp_pre_integration{};
	IntegrationBase* pre_integrations[(WINDOW_SIZE + 1)]{};
	Eigen::Vector3d acc_0, gyr_0;

	std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
	std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
	std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

	// frame position in sliding window and class for marginalization
	std::vector<Eigen::Vector3d> key_poses;
	double initial_timestamp{};
	MarginalizationInfo* last_marginalization_info{};
	std::vector<double*> last_marginalization_parameter_blocks;

	// parameters in the sliding windows stored in array, used for ceres optimize
	double para_pose[WINDOW_SIZE + 1][SIZE_POSE]{};
	double para_speed_bias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]{};
	double para_feature[NUM_OF_F][SIZE_FEATURE]{};
	double para_ex_pose[2][SIZE_POSE]{};
	double para_td[1][1]{};

	std::vector<int> param_feature_id;
	std::map<int, int> param_feature_id_to_index;

	// state flag
	bool init_first_pose_flag{};
	bool init_thread_flag;
	bool first_imu_flag{};
	bool failure_occur_flag{};
	SolverFlag solver_flag;
	MarginalizationFlag marginalization_flag;

	// healthy check params
	double visual_residual_weight;
	double imu_residual_weight;
	double mag_residual_weight;
};

} // namespace FLOW_VINS