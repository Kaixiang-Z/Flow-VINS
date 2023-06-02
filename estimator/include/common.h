/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-25 08:25:28
 * @Description: common functions
 */
#pragma once

#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <chrono>

using namespace Eigen;
namespace FLOW_VINS {

/**
 * @brief: convert ROS Gray image to Opencv format
 */
cv::Mat getGrayImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

/**
 * @brief: convert ROS depth image to Opencv format
 */
cv::Mat getDepthImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

/**
 * @brief: convert ROS RGB image to OpenCV format 
 */
cv::Mat getRgbImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

/**
 * @brief: Class which is used to timing, unit is ms
 */
class TicToc {
public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::system_clock::now();
    }

    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

/**
 * @brief: class for common compute
 */
class Utility {
public:
    /**
     * @brief: template function for quaternion perturbation operation
     */
    template <typename T>
    static Quaternion<typename T::Scalar>
    deltaQ(const MatrixBase<T> &theta) {
        typedef typename T::Scalar Scalar_t;

        Quaternion<Scalar_t> dq;
        Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    /**
     * @brief: template function for compute skew symmetric matrix
     */
    template <typename T>
    static Matrix<typename T::Scalar, 3, 3>
    skewSymmetric(const MatrixBase<T> &q) {
        Matrix<typename T::Scalar, 3, 3> ans;
        ans << typename T::Scalar(0), -q(2), q(1), q(2),
            typename T::Scalar(0), -q(0), -q(1), q(0),
            typename T::Scalar(0);
        return ans;
    }

    /**
     * @brief: template function for compute quaternion multiply left matrix
     */
    template <typename T>
    static Matrix<typename T::Scalar, 4, 4>
    Qleft(const QuaternionBase<T> &q) {
        Matrix<typename T::Scalar, 4, 4> ans;
        ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
        ans.template block<3, 1>(1, 0) = q.vec();
        ans.template block<3, 3>(1, 1) = q.w() * Matrix<typename T::Scalar, 3, 3>::Identity() + skewSymmetric(q.vec());
        return ans;
    }

    /**
     * @brief: template function for compute quaternion multiply right matrix
     */
    template <typename T>
    static Matrix<typename T::Scalar, 4, 4>
    Qright(const QuaternionBase<T> &p) {
        Matrix<typename T::Scalar, 4, 4> ans;
        ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
        ans.template block<3, 1>(1, 0) = p.vec();
        ans.template block<3, 3>(1, 1) = p.w() * Matrix<typename T::Scalar, 3, 3>::Identity() - skewSymmetric(p.vec());
        return ans;
    }

    /**
     * @brief: compute euler angles from rotation matrix, unit is degree
     */
    template <typename T>
    static Matrix<typename T::Scalar, 3, 1>
    R2ypr(const MatrixBase<T> &R) {
        typedef typename T::Scalar Scalar_t;

        Matrix<Scalar_t, 3, 1> n = R.col(0);
        Matrix<Scalar_t, 3, 1> o = R.col(1);
        Matrix<Scalar_t, 3, 1> a = R.col(2);

        Matrix<Scalar_t, 3, 1> ypr;
        Scalar_t y = atan2(n(1), n(0));
        Scalar_t p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        Scalar_t r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    /**
     * @brief: template function for compute rotation matrix from euler angles
     */
    template <typename T>
    static Matrix<typename T::Scalar, 3, 3>
    ypr2R(const MatrixBase<T> &ypr) {
        typedef typename T::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

        Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

        Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    /**
     * @brief: convert body gravity vector to world coordinate
     */
    template <typename T>
    static Matrix<typename T::Scalar, 3, 3>
    g2R(const MatrixBase<T> &g) {
        Matrix<typename T::Scalar, 3, 3> R0;
        // ng1 is body coordinate accelerate vector
        Matrix<typename T::Scalar, 3, 1> ng1 = g.normalized();
        // ng2 is the axis z vector in ENU coordinate (gravity direction)
        Matrix<typename T::Scalar, 3, 1> ng2{0, 0, 1.0};
        // ng2 = R0 * ng1, R0 rotate IMU coordinate to world coordinate
        R0 = Quaternion<typename T::Scalar>::FromTwoVectors(ng1, ng2).toRotationMatrix();
        // we just want to align the z axis while during the rotation the angle yaw might to be changed, so we need to rotate it back
        double yaw = Utility::R2ypr(R0).x();
        R0 = Utility::ypr2R(Matrix<typename T::Scalar, 3, 1>{-yaw, 0, 0}) * R0;
        return R0;
    }

    /**
     * @brief: template function for normalize angles in -180 to +180 degree
     */
    template <typename T>
    static T normalizeAngle(const T &angle_degrees) {
        if (angle_degrees > T(180.0))
            return angle_degrees - T(360.0);
        else if (angle_degrees < T(-180.0))
            return angle_degrees + T(360.0);
        else
            return angle_degrees;
    };

    /**
     * @brief: get quaternion inverse
     */
    template <typename T>
    static inline void quaternionInverse(const T q[4], T q_inverse[4]) {
        q_inverse[0] = q[0];
        q_inverse[1] = -q[1];
        q_inverse[2] = -q[2];
        q_inverse[3] = -q[3];
    }

    /**
     * @brief: rpy to rotation matrix
     */
    template <typename T>
    static void yawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9]) {
        T y = yaw / T(180.0) * T(M_PI);
        T p = pitch / T(180.0) * T(M_PI);
        T r = roll / T(180.0) * T(M_PI);

        R[0] = cos(y) * cos(p);
        R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
        R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
        R[3] = sin(y) * cos(p);
        R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
        R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
        R[6] = -sin(p);
        R[7] = cos(p) * sin(r);
        R[8] = cos(p) * cos(r);
    }

    /**
     * @brief: get transpose of rotation matrix
     */
    template <typename T>
    static void rotationMatrixTranspose(const T R[9], T inv_R[9]) {
        inv_R[0] = R[0];
        inv_R[1] = R[3];
        inv_R[2] = R[6];
        inv_R[3] = R[1];
        inv_R[4] = R[4];
        inv_R[5] = R[7];
        inv_R[6] = R[2];
        inv_R[7] = R[5];
        inv_R[8] = R[8];
    }

    /**
     * @brief: get rotation point
     */
    template <typename T>
    static void rotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3]) {
        r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
        r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
        r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
    }

}; // namespace FLOW_VINS

} // namespace FLOW_VINS