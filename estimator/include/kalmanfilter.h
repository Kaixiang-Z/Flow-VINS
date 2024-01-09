/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-06-19 19:22:39
 * @Description: Kalman filter for AHRS
 */
#pragma once

#include "parameter.h"
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

namespace FLOW_VINS {

struct IMU_State {
	// normal state
	Eigen::Quaterniond nominal_q;
	Eigen::Vector3d nominal_angular_velocity;

	// error state
	Eigen::Vector3d error_theta;
	Eigen::Vector3d error_angular_velocity;
	Eigen::Matrix<double, 6, 6> error_covariance;

	/**
	 * @brief: calculate observe matrix and correction residual
	 */
	Eigen::Matrix<double, 6, 6> calculateObservationMatrix(Eigen::Matrix<double, 9, 1> measurement,
	                                                       Eigen::Matrix<double, 6, 1>& residual) const;
};

class ESKF_Attitude {
public:
	/**
	 * @brief: constructor for ESKF estimator
	 */
	ESKF_Attitude();

	/**
	 * @brief: initialize the true state of the estimator, including the nominal state and error state, and the
	 * covariance matrices
	 */
	void initEstimator();

	/**
	 * @brief: predict the nominal state
	 */
	void predictNominalState();

	/**
	 * @brief: predict the error state
	 */
	void predictErrorState();

	/**
	 * @brief: update filter parameters
	 */
	void updateFilter();

	/**
	 * @brief: update the nominal state
	 */
	void updateNominalState();

	/**
	 * @brief: build update quaternion from delta theta
	 */
	static Eigen::Quaterniond buildUpdateQuaternion(const Eigen::Vector3d& delta_theta);

	/**
	 * @brief: convert euler to quaternion
	 */
	static Eigen::Matrix3d euler2RotationMat(Eigen::Vector3d euler);

	/**
	 * @brief: reset the error state
	 */
	void resetErrorState();

	/**
	 * @brief: read the sensors data and normalize the accelerometer and magnetometer
	 */
	void inputSensorData(Eigen::Matrix<double, 10, 1> measurement);

	/**
	 * @brief: run ESKF estimator
	 */
	Eigen::Quaterniond run(Eigen::Matrix<double, 10, 1> measurement);

private:
	Eigen::Vector3d delta_angular_noise;
	Eigen::Vector3d delta_angular_velocity_noise;
	Eigen::Vector3d accelerate_noise;
	Eigen::Vector3d magnetometer_noise;
	Eigen::Matrix<double, 6, 6> covariance_Q;
	Eigen::Matrix<double, 6, 6> covariance_R;

	std::vector<IMU_State> state_vector;
	std::vector<Eigen::Quaterniond> quaternion;
	// sample time
	double delta_T{};
	double last_time{};

	// current and last time measurement
	Eigen::Matrix<double, 9, 1> cur_measurement;
	Eigen::Matrix<double, 9, 1> last_measurement;
};
} // namespace FLOW_VINS
