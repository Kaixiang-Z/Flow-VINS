/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-06-19 19:22:27
 * @Description: Kalman filter for AHRS
 */
#include "../include/KalmanFilter.h"

namespace FLOW_VINS {

    Matrix<double, 6, 6>
    IMU_State::calculateObservationMatrix(Matrix<double, 9, 1> measurement, Matrix<double, 6, 1> &residual) {
        Vector3d true_acc_mea, true_mag_mea, mea_mag;
        Matrix<double, 4, 1> mag_global;
        Quaterniond q = nominal_q;

        mea_mag = measurement.block<3, 1>(6, 0);
        Quaterniond mag_q(0.0, mea_mag(0), mea_mag(1), mea_mag(2));
        mag_q = q * mag_q * q.conjugate();

        mag_global << 0, sqrt(mag_q.x() * mag_q.x() + mag_q.y() * mag_q.y()), 0, mag_q.z();

        // calculate observe matrix H
        Matrix<double, 3, 4> acc_H, mag_H;
        acc_H << 2 * q.y(), -2 * q.z(), 2 * q.w(), -2 * q.x(),
                -2 * q.x(), -2 * q.w(), -2 * q.z(), -2 * q.y(),
                0, 4 * q.x(), 4 * q.y(), 0;

        mag_H << -2 * mag_global(3) * q.y(), 2 * mag_global(3) * q.z(),
                -4 * mag_global(1) * q.y() - 2 * mag_global(3) * q.w(), -4 * mag_global(1) * q.z() +
                                                                        2 * mag_global(3) * q.x(),

                -2 * mag_global(1) * q.z() + 2 * mag_global(3) * q.x(), 2 * mag_global(1) * q.y() +
                                                                        2 * mag_global(3) * q.w(),
                2 * mag_global(1) * q.x() + 2 * mag_global(3) * q.z(), -2 * mag_global(1) * q.w() +
                                                                       2 * mag_global(3) * q.y(),

                2 * mag_global(1) * q.y(), 2 * mag_global(1) * q.z() - 4 * mag_global(3) * q.x(),
                2 * mag_global(1) * q.w() - 4 * mag_global(3) * q.y(), 2 * mag_global(1) * q.x();

        Matrix<double, 6, 7> observe_Hx;
        observe_Hx.block<3, 4>(0, 0) = acc_H;
        observe_Hx.block<3, 4>(3, 0) = mag_H;
        observe_Hx.block<3, 3>(0, 4) = Matrix3d::Zero();
        observe_Hx.block<3, 3>(3, 4) = Matrix3d::Zero();

        Matrix<double, 7, 6> observe_Xx;
        Matrix<double, 4, 3> matrix_Q;
        matrix_Q << -q.x(), -q.y(), -q.z(),
                q.w(), -q.z(), q.y(),
                q.z(), q.w(), -q.x(),
                -q.y(), q.x(), q.w();
        observe_Xx.block<4, 3>(0, 0) = 0.5 * matrix_Q;
        observe_Xx.block<4, 3>(0, 3) = Matrix<double, 4, 3>::Zero();
        observe_Xx.block<3, 3>(4, 0) = Matrix3d::Zero();
        observe_Xx.block<3, 3>(4, 3) = Matrix3d::Zero();

        Matrix<double, 6, 6> observe_matrix;
        observe_matrix = observe_Hx * observe_Xx;

        // calculate the true measurement
        true_acc_mea << -2 * (q.x() * q.z() - q.w() * q.y()),
                -2 * (q.w() * q.x() + q.y() * q.z()),
                -2 * (0.5 - q.x() * q.x() - q.y() * q.y());

        // the true magnetometer measurement
        true_mag_mea << -(2 * mag_global(1) * (0.5 - q.y() * q.y() - q.z() * q.z()) +
                          2 * mag_global(3) * (q.x() * q.z() - q.w() * q.y())),
                -(2 * mag_global(1) * (q.x() * q.y() - q.w() * q.z()) +
                  2 * mag_global(3) * (q.w() * q.x() + q.y() * q.z())),
                -(2 * mag_global(1) * (q.w() * q.y() + q.x() * q.z()) +
                  2 * mag_global(3) * (0.5 - q.x() * q.x() - q.y() * q.y()));

        // calculate the residual
        residual.block<3, 1>(0, 0) = measurement.block<3, 1>(3, 0) + true_acc_mea;
        residual.block<3, 1>(3, 0) = measurement.block<3, 1>(6, 0) + true_mag_mea;

        return observe_matrix;
    }

    ESKF_Attitude::ESKF_Attitude() {
        delta_angular_noise = 1e-5 * Eigen::Vector3d::Ones();
        delta_angular_velocity_noise = 1e-9 * Eigen::Vector3d::Ones();
        acclerate_noise = 1e-3 * Eigen::Vector3d::Ones();
        magnetometer_noise = 1e-4 * Eigen::Vector3d::Ones();

        covariance_Q = Matrix<double, 6, 6>::Zero();
        covariance_R = Matrix<double, 6, 6>::Zero();
    }

    void ESKF_Attitude::initEstimator() {
        // initialize the convariances matrices Q and R
        covariance_Q.block<3, 3>(0, 0) = delta_angular_noise.asDiagonal();
        covariance_Q.block<3, 3>(3, 3) = delta_angular_noise.asDiagonal();
        covariance_R.block<3, 3>(0, 0) = acclerate_noise.asDiagonal();
        covariance_R.block<3, 3>(3, 3) = magnetometer_noise.asDiagonal();

        // initialize the nominal state
        Vector3d acc0, gyr0, mag0;
        gyr0 = cur_measurement.block<3, 1>(0, 0);
        acc0 = cur_measurement.block<3, 1>(3, 0);
        mag0 = cur_measurement.block<3, 1>(6, 0);

        double pitch0 = asin(-acc0[0] / acc0.norm());
        double roll0 = atan2(-acc0[1], -acc0[2]);
        double yaw0 = atan2(mag0[1] * cos(roll0) - mag0[2] * sin(roll0),
                            mag0[0] * cos(pitch0) + mag0[1] * sin(pitch0) * sin(roll0) +
                            mag0[2] * sin(pitch0) * cos(roll0));

        Quaterniond q_temp = Utility::euler2Quaternion(Vector3d(roll0, pitch0, yaw0));

        q_temp.normalize();

        Vector3d angle_temp = delta_angular_noise;

        IMU_State state;
        state.nominal_q = q_temp;
        state.nominal_angular_velocity = angle_temp;

        // initialize the error state
        Vector3d delta_theta, delta_angle_velocity;
        state.error_theta = Vector3d::Zero();
        state.error_angular_velocity = Vector3d::Zero();
        state.error_covariance = Matrix<double, 6, 6>::Zero();
        state.error_covariance.block<3, 3>(0, 0) = 1e-5 * Matrix3d::Identity();
        state.error_covariance.block<3, 3>(3, 3) = 1e-7 * Matrix3d::Identity();

        state_vector.push_back(state);
        quaternion.push_back(q_temp);

        last_measurement = cur_measurement;
    }

    void ESKF_Attitude::predictNominalState() {
        Vector3d delta_theta;
        Quaterniond q_temp;
        IMU_State piror_state = state_vector.back();
        delta_theta = (0.5 * (cur_measurement.block<3, 1>(0, 0) + last_measurement.block<3, 1>(0, 0)) -
                       piror_state.nominal_angular_velocity) * delta_T;

        // gibbs vector
        q_temp.w() = 1;
        q_temp.vec() = 0.5 * delta_theta;
        q_temp = piror_state.nominal_q * q_temp;
        q_temp.normalize();

        IMU_State post_state;
        post_state.nominal_q = q_temp;
        post_state.nominal_angular_velocity = piror_state.nominal_angular_velocity;
        state_vector.push_back(post_state);
    }

    void ESKF_Attitude::predictErrorState() {
        IMU_State post_state = state_vector.back();
        state_vector.pop_back();
        IMU_State piror_state = state_vector.back();

        // calculate the transition matrix A
        Matrix<double, 6, 6> transition_A;
        Vector3d delta_theta;
        delta_theta == (last_measurement.block<3, 1>(0, 0) - piror_state.nominal_angular_velocity) * delta_T;
        Matrix3d rotation_matrix = euler2RotationMat(delta_theta);

        transition_A.block<3, 3>(0, 0) = rotation_matrix.transpose();
        transition_A.block<3, 3>(0, 3) = -delta_T * Matrix3d::Identity();
        transition_A.block<3, 3>(3, 0) = Matrix3d::Zero();
        transition_A.block<3, 3>(3, 3) = Matrix3d::Identity();

        // pirori prediction

        Matrix<double, 6, 1> error_state_temp;
        error_state_temp.block<3, 1>(0, 0) = piror_state.error_theta;
        error_state_temp.block<3, 1>(3, 0) = piror_state.error_angular_velocity;
        error_state_temp = transition_A * error_state_temp;

        post_state.error_theta = error_state_temp.block<3, 1>(0, 0);
        post_state.error_angular_velocity = error_state_temp.block<3, 1>(3, 0);

        Matrix<double, 6, 6> covariance_Qi = covariance_Q * delta_T;
        //Matrix<double, 6, 6> covariance_Qi = covariance_Q;
        Matrix<double, 6, 6> noise_Fi = Matrix<double, 6, 6>::Identity();

        post_state.error_covariance = transition_A * piror_state.error_covariance * transition_A.transpose() +
                                      noise_Fi * covariance_Qi * noise_Fi.transpose();

        state_vector.push_back(post_state);
    }

    void ESKF_Attitude::updateFilter() {
        Matrix<double, 6, 1> residual;
        Matrix<double, 6, 6> observe_matrix;
        IMU_State post_state = state_vector.back();
        state_vector.pop_back();

        // calculate the observe matrix and the correction residual
        observe_matrix = post_state.calculateObservationMatrix(cur_measurement, residual);

        // calculate kalman gain
        Matrix<double, 6, 6> post_covariance = post_state.error_covariance;
        Matrix<double, 6, 6> kalman_gain;
        kalman_gain = observe_matrix * post_covariance * observe_matrix.transpose() + covariance_R;
        kalman_gain = post_covariance * observe_matrix.transpose() * kalman_gain.inverse();

        // update error state
        Matrix<double, 6, 1> post_error_state = kalman_gain * residual;
        post_state.error_theta = post_error_state.block<3, 1>(0, 0);
        post_state.error_angular_velocity = post_error_state.block<3, 1>(3, 0);

        post_state.error_covariance =
                post_covariance -
                kalman_gain * (observe_matrix * post_covariance * observe_matrix.transpose() + covariance_R) *
                kalman_gain.transpose();

        state_vector.push_back(post_state);
    }

    Quaterniond ESKF_Attitude::buildUpdateQuaternion(Vector3d delta_theta) {
        Vector3d delta_q = 0.5 * delta_theta;
        double checknorm = delta_q.transpose() * delta_q;

        Quaterniond update_q;
        if (checknorm > 1) {
            update_q = Quaterniond(1, delta_q[0], delta_q[1], delta_q[2]);
            update_q = update_q.coeffs() * sqrt(1 + checknorm);
        } else
            update_q = Quaterniond(sqrt(1 - checknorm), delta_q[0], delta_q[1], delta_q[2]);

        update_q.normalize();

        return update_q;
    }

    Matrix3d ESKF_Attitude::euler2RotationMat(Vector3d euler) {
        Matrix3d rotate_mat;
        double theta;
        Matrix3d skew_euler;

        theta = euler.norm();
        euler.normalize();

        skew_euler = Utility::skewSymmetric(euler);

        rotate_mat = cos(theta) * Matrix3d::Identity() + sin(theta) * skew_euler +
                     (1 - cos(theta)) * skew_euler.transpose() * skew_euler;
        return rotate_mat;
    }

    void ESKF_Attitude::updateNominalState() {
        IMU_State post_state = state_vector.back();
        state_vector.pop_back();

        Quaterniond delta_q = buildUpdateQuaternion(post_state.error_theta);
        post_state.nominal_q = post_state.nominal_q * delta_q;
        post_state.nominal_q.normalize();

        post_state.nominal_angular_velocity = post_state.nominal_angular_velocity + post_state.error_angular_velocity;

        state_vector.push_back(post_state);
        quaternion.push_back(post_state.nominal_q);
    }

    void ESKF_Attitude::resetErrorState() {
        IMU_State post_state = state_vector.back();
        state_vector.pop_back();

        Matrix<double, 6, 6> matrix_G = Matrix<double, 6, 6>::Identity();

        post_state.error_theta = Vector3d::Zero();
        post_state.error_angular_velocity = Vector3d::Zero();
        post_state.error_covariance = matrix_G * post_state.error_covariance * matrix_G.transpose();

        state_vector.push_back(post_state);

        last_measurement = cur_measurement;
    }

    void ESKF_Attitude::inputSensorData(Matrix<double, 10, 1> measurement) {
        Vector3d gyro_mea, acc_mea, mag_mea;
        acc_mea = measurement.block<3, 1>(0, 0);
        gyro_mea = measurement.block<3, 1>(3, 0);
        mag_mea = measurement.block<3, 1>(6, 0);
        double time = measurement(9, 0);

        delta_T = time - last_time;
        last_time = time;

        acc_mea.normalize();
        mag_mea.normalize();

        cur_measurement.block<3, 1>(0, 0) = gyro_mea;
        cur_measurement.block<3, 1>(3, 0) = acc_mea;
        cur_measurement.block<3, 1>(6, 0) = mag_mea;
    }

    Quaterniond ESKF_Attitude::run(Matrix<double, 10, 1> measurement) {
        inputSensorData(measurement);

        if (state_vector.size() == 0 || quaternion.size() == 0)
            initEstimator();
        else {
            predictNominalState();
            predictErrorState();
            updateFilter();
            updateNominalState();
            resetErrorState();
        }

        return quaternion.back();
    }

} // namespace FLOW_VINS