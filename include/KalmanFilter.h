/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-06-19 19:22:39
 * @Description: Kalman filter for AHRS
 */
#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <map>
#include <Eigen/Dense>
#include <unistd.h>
#include <ros/ros.h>

#include "Common.h"
#include "Parameters.h"

namespace FLOW_VINS {

    struct IMU_State {
        // normal state
        Quaterniond nominal_q;
        Vector3d nominal_angular_velocity;

        // error state
        Vector3d error_theta;
        Vector3d error_angular_velocity;
        Matrix<double, 6, 6> error_covariance;

        /**
         * @brief: calculate observe matrix and correction residual
         */
        Matrix<double, 6, 6>
        calculateObservationMatrix(Matrix<double, 9, 1> measurement, Matrix<double, 6, 1> &residual);
    };

    class ESKF_Attitude {
    public:
        /**
         * @brief: constructor for ESKF estimator
         */
        ESKF_Attitude();

        /**
         * @brief: initialize the true state of the estimator, including the nominal state and error state, and the covariance matrices
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
        Quaterniond buildUpdateQuaternion(Vector3d delta_theta);

        /**
         * @brief: convert euler to quaternion
         */
        Matrix3d euler2RotationMat(Vector3d euler);

        /**
         * @brief: reset the error state
         */
        void resetErrorState();

        /**
         * @brief: read the sensors data and normalize the accelerometer and magnetometer
         */
        void inputSensorData(Matrix<double, 10, 1> measurement);

        /**
         * @brief: run ESKF estimator
         */
        Quaterniond run(Matrix<double, 10, 1> measurement);

    private:
        Vector3d delta_angular_noise;
        Vector3d delta_angular_velocity_noise;
        Vector3d acclerate_noise;
        Vector3d magnetometer_noise;
        Matrix<double, 6, 6> covariance_Q;
        Matrix<double, 6, 6> covariance_R;

        vector <IMU_State> state_vector;
        vector <Quaterniond> quaternion;
        // sample time
        double delta_T;
        double last_time;

        // current and last time measurement
        Matrix<double, 9, 1> cur_measurement;
        Matrix<double, 9, 1> last_measurement;
    };
} // namespace FLOW_VINS
