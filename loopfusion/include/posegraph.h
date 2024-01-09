/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: pose graph
 */

#pragma once

#include "../../thirdparty/dbow/dbow2.h"
#include "../../thirdparty/dbow/templatedatabase.h"
#include "../../thirdparty/dbow/templatevocabulary.h"
#include "../../thirdparty/dvision/dvision.h"
#include "keyframe.h"
#include "parameter.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>

namespace FLOW_VINS {

/**
 * @brief: class for pose graph
 */
class PoseGraph {
public:
	/**
	 * @brief: constructor function
	 */
	PoseGraph();

	/**
	 * @brief: detach thread
	 */
	~PoseGraph();

	/**
	 * @brief: set initial parameters, load vocabulary and start 4 & 6 DOF optimize
	 */
	void setParameter();

	/**
	 * @brief: load vocabulary library
	 */
	void loadVocabulary(const std::string& voc_path);

	/**
	 * @brief: confirm IMU is enabled, start 4 & 6 DOF optimize thread
	 */
	void setIMUFlag(bool _use_imu);

	/**
	 * @brief: add one keyframe
	 */
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);

	int sequence_cnt;
	// ros publish state
	nav_msgs::msg::Path path[10];
	nav_msgs::msg::Path base_path;
	// drift state
	Eigen::Vector3d t_drift;
	Eigen::Matrix3d r_drift;
	double yaw_drift;

	Eigen::Vector3d w_t_vio;
	Eigen::Matrix3d w_r_vio;

private:
	/**
	 * @brief: find old keyframe with index
	 */
	KeyFrame* getKeyFrame(int index);
	/**
	 * @brief: main process of loop detect
	 */
	int detectLoop(KeyFrame* keyframe, int frame_index);

	/**
	 * @brief: add keyframe into vocabulary, only for visualization
	 */
	void addKeyFrameIntoVoc(KeyFrame* keyframe);

	/**
	 * @brief: main process of optimize 4 DoF (VIO)
	 */
	void optimize4DoF();

	/**
	 * @brief: main process of optimize 6 DoF (VO)
	 */
	void optimize6DoF();

	/**
	 * @brief: udpate path and publish topic, save loop result file
	 */
	void updatePath();

	std::mutex m_keyframelist;
	std::mutex m_optimize_buf;
	std::mutex m_drift;
	std::thread optimization_thread;

	std::queue<int> optimize_buf;
	std::list<KeyFrame*> keyframelist;
	std::vector<bool> sequence_loop;

	// index
	int global_index;
	int earliest_loop_index;
	bool use_imu;

	// brief descriptor
	BriefDatabase db;
	BriefVocabulary* voc{};
};

/**
 * @brief: class for angle ceres optimize
 */
class AngleLocalParameterization {
public:
	/**
	 * @brief: angle states addition on manifold
	 */
	template <typename T>
	bool Plus(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const {
		*theta_radians_plus_delta = Utility::normalizeAngle(*theta_radians + *delta_theta_radians);

		return true;
	}

	/**
	 * @brief: implements minus operation for the manifold
	 */
	template <typename T>
	bool Minus(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const {
		return true;
	}

	/**
	 * @brief: create auto difference manifold, compute jacobian and residual
	 */
	static ceres::Manifold* Create() { return (new ceres::AutoDiffManifold<AngleLocalParameterization, 1, 1>); }
};

/**
 * @brief: struct for 4 DoF cost function
 */
struct FourDOFError {
	/**
	 * @brief: constructor function, initial parameters
	 */
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
	    : t_x(t_x)
	    , t_y(t_y)
	    , t_z(t_z)
	    , relative_yaw(relative_yaw)
	    , pitch_i(pitch_i)
	    , roll_i(roll_i) {}

	/**
	 * @brief: compute residual
	 */
	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const {
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		Utility::yawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		Utility::rotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		Utility::rotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = Utility::normalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	/**
	 * @brief: create cost function, residual dimension: 4 (x, y, z, yaw), yaw_i (1), xyz_i(3), yaw_j (1), xyz_j(3)
	 */
	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z, const double relative_yaw,
	                                   const double pitch_i, const double roll_i) {
		return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
		    new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	// state parameters
	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
};

/**
 * @brief: struct for 6 DoF loop cost function
 */
struct SixDOFError {
	/**
	 * @brief: constructor function, initial parameters
	 */
	SixDOFError(double t_x, double t_y, double t_z, double q_w, double q_x, double q_y, double q_z, double t_var,
	            double q_var)
	    : t_x(t_x)
	    , t_y(t_y)
	    , t_z(t_z)
	    , q_w(q_w)
	    , q_x(q_x)
	    , q_y(q_y)
	    , q_z(q_z)
	    , t_var(t_var)
	    , q_var(q_var) {}

	/**
	 * @brief: compute residual
	 */
	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j, const T* tj, T* residuals) const {
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4];
		Utility::quaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		// compute residuals
		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

		T relative_q_inv[4];
		Utility::quaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

		// reprojection residual
		residuals[3] = T(2) * error_q[1] / T(q_var);
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	/**
	 * @brief: create cost function, residual dimension: 6 (x, y, z, yaw, pitch, roll), q_i (4), t_i(3), q_j (4),
	 * xyz_j(3)
	 */
	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z, const double q_w,
	                                   const double q_x, const double q_y, const double q_z, const double t_var,
	                                   const double q_var) {
		return (new ceres::AutoDiffCostFunction<SixDOFError, 6, 4, 3, 4, 3>(
		    new SixDOFError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	// state parameters
	double t_x, t_y, t_z, t_norm{};
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;
};
} // namespace FLOW_VINS