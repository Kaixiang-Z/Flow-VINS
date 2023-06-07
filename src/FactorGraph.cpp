/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-28 10:45:54
 * @Description: factor
 */
#include "../include/FactorGraph.h"

namespace FLOW_VINS {

Matrix2d ProjectionOneFrameTwoCamFactor::sqrt_info;
Matrix2d ProjectionTwoFrameOneCamFactor::sqrt_info;
Matrix2d ProjectionTwoFrameTwoCamFactor::sqrt_info;
Matrix2d ProjectionRelocationFactor::sqrt_info;

double ProjectionOneFrameTwoCamFactor::sum_t;
double ProjectionTwoFrameOneCamFactor::sum_t;
double ProjectionTwoFrameTwoCamFactor::sum_t;
double ProjectionRelocationFactor::sum_t;

// ---------------------------------------------IMU Residual--------------------------------------------- //

IntegrationBase::IntegrationBase(const Vector3d &_acc_0, const Vector3d &_gyr_0,
                                 Vector3d _linearized_ba, Vector3d _linearized_bg) :
    acc_0{_acc_0},
    gyr_0{_gyr_0},
    linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
    linearized_ba{move(_linearized_ba)}, linearized_bg{move(_linearized_bg)},
    jacobian{Matrix<double, 15, 15>::Identity()}, covariance{Matrix<double, 15, 15>::Zero()}, sum_dt{0.0},
    delta_p{Vector3d::Zero()}, delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()} {
    // measure the noise covariance which is considered to be fixed at each time instant
    noise = Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Matrix3d::Identity();
}

void IntegrationBase::push_back(double dt, const Vector3d &acc, const Vector3d &gyr) {
    // backup IMU data for repropagate
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);

    // propagate pre-integration state
    propagate(dt, acc, gyr);
}

void IntegrationBase::repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg) {
    // clear state and set bias as input
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    // repropagate IMU pre-integration
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
        propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

void IntegrationBase::midPointIntegration(double _dt, const Vector3d &_acc_0, const Vector3d &_gyr_0,
                                          const Vector3d &_acc_1, const Vector3d &_gyr_1,
                                          const Vector3d &delta_p, const Quaterniond &delta_q,
                                          const Vector3d &delta_v, const Vector3d &linearized_ba,
                                          const Vector3d &linearized_bg, Vector3d &result_delta_p,
                                          Quaterniond &result_delta_q, Vector3d &result_delta_v,
                                          Vector3d &result_linearized_ba, Vector3d &result_linearized_bg,
                                          bool update_jacobian) {
    // PVQ is state in world coordinate, acceleration and gyroscope is in IMU(body) coordinate
    // acceleration at the previous moment
    Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
    // the midian value of gyro data between the previous moment and the current moment
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    // rotation matrix Q at the current moment
    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    // acceleration at the current moment
    Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
    // the midian value of acceleration data between the previous moment and the current moment
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // update the current moment P and V, where Ba, Bg remain unchanged
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    // calculate jacobian matrix
    if (update_jacobian) {
        Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Vector3d a_0_x = _acc_0 - linearized_ba;
        Vector3d a_1_x = _acc_1 - linearized_ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1), a_1_x(0), 0;

        // differentiation of the error at the current moment to the error at the previous moment
        MatrixXd F = MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) =
            -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
        F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
        F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        // differentiation of the error to the noise at the current moment
        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

        // iterate jacobian and covariance matrix
        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
}

void IntegrationBase::propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1) {
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    // midian integration
    midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q,
                        delta_v, linearized_ba, linearized_bg, result_delta_p,
                        result_delta_q, result_delta_v, result_linearized_ba,
                        result_linearized_bg, true);
    // update state
    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
}

Matrix<double, 15, 1> IntegrationBase::evaluate(const Vector3d &Pi, const Quaterniond &Qi,
                                                const Vector3d &Vi, const Vector3d &Bai,
                                                const Vector3d &Bgi, const Vector3d &Pj,
                                                const Quaterniond &Qj, const Vector3d &Vj,
                                                const Vector3d &Baj, const Vector3d &Bgj) {
    // residual: the difference of PVQ to bias between the previous frame and the current frame
    Matrix<double, 15, 1> residuals;
    // get jacobian of state to bias from orginal jacobian
    Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);
    Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
    Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    // get bias error at the first moment of pre-integration
    Vector3d dba = Bai - linearized_ba;
    Vector3d dbg = Bgi - linearized_bg;

    // here it is assumed that the pre-integration change is linear in relation to bias
    // update PVQ after update bias
    Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
    Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    // Use the odometer pose transformation corresponding to the start and end moments of the pre-integration,
    // and subtract the pre-integration amount to construct the residual
    residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    // only save the imaginary part of quaternion in rotation
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}

bool IMUFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    // value of the optimization variable, PVQ，Ba，Bg
    Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

    Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

    // calculate the residual of pre-integration
    Map<Matrix<double, 15, 1>> residual(residuals);
    residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    // information matrix
    Matrix<double, 15, 15> sqrt_info = LLT<Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();

    residual = sqrt_info * residual;
    // calculate the jacobian of the residuals with respect to the optimization variables
    if (jacobians) {
        // the differential of the pre-integration end time error relative to the pre-integration start time error is used to correct the pre-integration increment
        double sum_dt = pre_integration->sum_dt;
        Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
        Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);
        Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
        Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

        // numerical check
        if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8) {
            ROS_DEBUG("numerical unstable in preintegration");
        }

        // find the optimization variable by perturbation
        if (jacobians[0]) {
            Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();

            jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
            Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
            jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
            jacobian_pose_i = sqrt_info * jacobian_pose_i;

            if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
                ROS_DEBUG("numerical unstable in preintegration");
            }
        }
        if (jacobians[1]) {
            Map<Matrix<double, 15, 9, RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();

            jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
            jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
            jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
            jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Matrix3d::Identity();
            jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Matrix3d::Identity();
            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        }
        if (jacobians[2]) {
            Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();

            jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
            Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
            jacobian_pose_j = sqrt_info * jacobian_pose_j;
        }
        if (jacobians[3]) {
            Map<Matrix<double, 15, 9, RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();

            jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
            jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Matrix3d::Identity();
            jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Matrix3d::Identity();
            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        }
    }
    return true;
}

// ---------------------------------------------Projection Residual--------------------------------------------- //

/**
 * @brief: all of visual reporojection residual only used pinhole camera model instead of spherical camera model
 */

ProjectionOneFrameTwoCamFactor::ProjectionOneFrameTwoCamFactor(
    Vector3d _pts_i, Vector3d _pts_j,
    const Vector2d &_velocity_i, const Vector2d &_velocity_j,
    const double _td_i, const double _td_j) :
    pts_i(move(_pts_i)),
    pts_j(move(_pts_j)), td_i(_td_i), td_j(_td_j) {
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;
}

bool ProjectionOneFrameTwoCamFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const {
    TicToc tic_toc;

    // set parameter
    Vector3d tic(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond qic(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d tic2(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond qic2(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_i = parameters[2][0];

    double td = parameters[3][0];

    // correspond point i j,
    // normalized camera point time difference correction, corresponding to the position at the acquisition time
    Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;

    // normalize camera point in left camera
    Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    // point in IMU coordinate from left camera coordinate
    Vector3d pts_imu_i = qic * pts_camera_i + tic;
    const Vector3d &pts_imu_j = pts_imu_i;
    // convert to camera point in right point
    Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
    Map<Vector2d> residual(residuals);

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();

    residual = sqrt_info * residual;

    // calculate visual reprojection residual jacobian
    if (jacobians) {
        Matrix3d ric = qic.toRotationMatrix();
        Matrix3d ric2 = qic2.toRotationMatrix();
        Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
        reduce = sqrt_info * reduce;

        // jacobian for left extrinsic param
        if (jacobians[0]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose(jacobians[0]);

            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose();
            jaco_ex.rightCols<3>() = ric2.transpose() * ric * -Utility::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // jacobian for right extrinsic param
        if (jacobians[1]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose1(jacobians[1]);

            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = -ric2.transpose();
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        // jacobian for inverse depth
        if (jacobians[2]) {
            Map<Vector2d> jacobian_feature(jacobians[2]);

            jacobian_feature = reduce * ric2.transpose() * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
        }
        // jacobian for td
        if (jacobians[3]) {
            Map<Vector2d> jacobian_td(jacobians[3]);

            jacobian_td = reduce * ric2.transpose() * ric * velocity_i / inv_dep_i * -1.0 + sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

ProjectionTwoFrameOneCamFactor::ProjectionTwoFrameOneCamFactor(
    Vector3d _pts_i, Vector3d _pts_j,
    const Vector2d &_velocity_i, const Vector2d &_velocity_j,
    const double _td_i, const double _td_j) :
    pts_i(move(_pts_i)),
    pts_j(move(_pts_j)), td_i(_td_i), td_j(_td_j) {
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;
}

bool ProjectionTwoFrameOneCamFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const {
    TicToc tic_toc;
    // set parameter
    Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    double td = parameters[4][0];

    Vector3d pts_i_td, pts_j_td;
    // matching point i & j,
    // normalized camera point time difference correction, corresponding to the position at the time of collection
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
    // convert to camera coordinate
    Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    // convert to IMU coordinate
    Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // convert to world coordinate
    Vector3d pts_w = Qi * pts_imu_i + Pi;
    // convert to IMU coordinate in point j
    Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // convert to camera coordinate in point j
    Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    Map<Vector2d> residual(residuals);

    // compute residual between two points in normalized camera plane
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();

    // multiply sqrt info
    residual = sqrt_info * residual;

    // compute jacobian
    if (jacobians) {
        Matrix3d Ri = Qi.toRotationMatrix();
        Matrix3d Rj = Qj.toRotationMatrix();
        Matrix3d ric = qic.toRotationMatrix();
        Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        reduce = sqrt_info * reduce;

        // jacobian for frame i
        if (jacobians[0]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_i(jacobians[0]);

            Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        // jacobian for frame j
        if (jacobians[1]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_j(jacobians[1]);

            Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        // jacobian for ric & tic
        if (jacobians[2]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose(jacobians[2]);

            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Matrix3d::Identity());
            Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() =
                -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) + Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // jacobian for inverse depth
        if (jacobians[3]) {
            Map<Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        // jacobian for td
        if (jacobians[4]) {
            Map<Vector2d> jacobian_td(jacobians[4]);
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0 + sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(
    Vector3d _pts_i, Vector3d _pts_j,
    const Vector2d &_velocity_i, const Vector2d &_velocity_j,
    const double _td_i, const double _td_j) :
    pts_i(move(_pts_i)),
    pts_j(move(_pts_j)), td_i(_td_i), td_j(_td_j) {
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;
}

bool ProjectionTwoFrameTwoCamFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const {
    TicToc tic_toc;
    // set parameter
    Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    double inv_dep_i = parameters[4][0];

    double td = parameters[5][0];

    // matching point i & j,
    // normalized camera point time difference correction, corresponding to the position at the time of collection
    Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;

    // normalized left camera point in the previous frame
    Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    // point in IMU coordinate in the previous frame
    Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // convert to world coordinate
    Vector3d pts_w = Qi * pts_imu_i + Pi;
    // convert to IMU coordinate in the current frame
    Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // convert to camera coordinate in the current frame
    Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
    Map<Vector2d> residual(residuals);

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();

    residual = sqrt_info * residual;

    if (jacobians) {
        Matrix3d Ri = Qi.toRotationMatrix();
        Matrix3d Rj = Qj.toRotationMatrix();
        Matrix3d ric = qic.toRotationMatrix();
        Matrix3d ric2 = qic2.toRotationMatrix();
        Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
        reduce = sqrt_info * reduce;

        // jacobian for frame i
        if (jacobians[0]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_i(jacobians[0]);

            Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        // jacobian for frame j
        if (jacobians[1]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_j(jacobians[1]);

            Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric2.transpose() * Utility::skewSymmetric(pts_imu_j);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        // jacobian for ric & tic
        if (jacobians[2]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose(jacobians[2]);

            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri;
            jaco_ex.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * ric * -Utility::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // jacobian for ric2 & tic2
        if (jacobians[3]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose1(jacobians[3]);

            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = -ric2.transpose();
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        // jacobian for inverse depth
        if (jacobians[4]) {
            Map<Vector2d> jacobian_feature(jacobians[4]);

            jacobian_feature = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        // jacobian for td
        if (jacobians[5]) {
            Map<Vector2d> jacobian_td(jacobians[5]);

            jacobian_td = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0 + sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

ProjectionRelocationFactor::ProjectionRelocationFactor(const Vector3d &_pts_i, const Vector3d &_pts_j) :
    pts_i(_pts_i), pts_j(_pts_j){};

bool ProjectionRelocationFactor::Evaluate(double const *const *parameters, double *residuals,
                                          double **jacobians) const {
    TicToc tic_toc;
    // set parameter
    Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    Vector3d pts_camera_i = pts_i / inv_dep_i;
    Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Vector3d pts_w = Qi * pts_imu_i + Pi;
    Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Map<Vector2d> residual(residuals);

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();

    residual = sqrt_info * residual;

    if (jacobians) {
        Matrix3d Ri = Qi.toRotationMatrix();
        Matrix3d Rj = Qj.toRotationMatrix();
        Matrix3d ric = qic.toRotationMatrix();
        Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        reduce = sqrt_info * reduce;
        // jacobian for frame i
        if (jacobians[0]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_i(jacobians[0]);

            Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        // jacobian for frame j
        if (jacobians[1]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_pose_j(jacobians[1]);

            Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        // jacobian for ric & tic
        if (jacobians[2]) {
            Map<Matrix<double, 2, 7, RowMajor>> jacobian_ex_pose(jacobians[2]);
            Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Matrix3d::Identity());
            Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) + Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // jacobian for inverse depth
        if (jacobians[3]) {
            Map<Vector2d> jacobian_feature(jacobians[3]);

            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

// ---------------------------------------------Marginalization Residual--------------------------------------------- //

void ResidualBlockInfo::Evaluate() {
    // resize to actual residual dimension
    residuals.resize(cost_function->num_residuals());
    // get parameter nums in each parameter block
    vector<int> block_sizes = cost_function->parameter_block_sizes();
    // get nums of parameter block
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    // set jacobian
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    // get residual and jacobian, because of the optimization before, the value of them usually small
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    // huber function, rewrite the jacobian and residual
    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;
        double sq_norm, rho[3];
        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo() {
    for (auto &it : parameter_block_data)
        delete it.second;

    for (auto &factor : factors) {
        delete[] factor->raw_jacobians;

        delete factor->cost_function;

        delete factor;
    }
}

void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info) {
    // add residual factor
    factors.emplace_back(residual_block_info);

    // get residual parameter blocks
    vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // get residual parameter blocks size
    vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // traverse each parameter block
    for (auto i = 0; i < residual_block_info->parameter_blocks.size(); i++) {
        // parameter block start address
        double *addr = parameter_blocks[i];
        // parameter block size
        int size = parameter_block_sizes[i];
        // set to parameter_block_size <variable block start address, variable block size>
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // traverse each parameter block to be droped out
    for (int i : residual_block_info->drop_set) {
        // parameter block start address
        double *addr = parameter_blocks[i];
        // set marg variable index to zero
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

void MarginalizationInfo::preMarginalize() {
    // traverse residual blocks related to the current marg frame
    for (auto it : factors) {
        // residual block evaluate, get jacobian and residual
        it->Evaluate();

        // traverse variable after optimization
        vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            // parameter block start address
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            // copy the parameter block in factor to parameter_block_data, parameter_block_data is the data of the entire optimization variable
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                auto *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

static void *threadsConstructA(void *threads_struct) {
    ThreadsStruct *p = ((ThreadsStruct *)threads_struct);
    // traverse residual block
    for (auto it : p->sub_factors) {
        // traverse variable block
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            // parameter block index
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            // parameter block size
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threads_struct;
}

void MarginalizationInfo::marginalize() {
    // traverse parameter block to be marged and calculate index
    int pos = 0;
    for (auto &it : parameter_block_idx) {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    // get the number of variable to be droped
    m = pos;

    // add reserved variables to parameter_block_idx and put the variables that need Marg to the front
    for (const auto &it : parameter_block_size) {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // get the number of variable to be retained
    n = pos - m;
    if (m == 0) {
        valid = false;
        cout << "no variable to be marginalized, unstable tracking..." << endl;
        return;
    }

    // matrix before marginalization
    MatrixXd A(pos, pos);
    VectorXd b(pos);
    A.setZero();
    b.setZero();

    // 4 threads, add residual items, construct information matrix A = J^T * J, b = J^T * r
    pthread_t tids[NUM_MARGIN_THREADS];
    ThreadsStruct threads_struct[NUM_MARGIN_THREADS];
    int i = 0;
    for (auto it : factors) {
        threads_struct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_MARGIN_THREADS;
    }
    for (int j = 0; j < NUM_MARGIN_THREADS; j++) {
        threads_struct[j].A = MatrixXd::Zero(pos, pos);
        threads_struct[j].b = VectorXd::Zero(pos);
        threads_struct[j].parameter_block_size = parameter_block_size;
        threads_struct[j].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create(&tids[j], nullptr, threadsConstructA, (void *)&(threads_struct[j]));
        if (ret != 0) {
            //ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for (int k = NUM_MARGIN_THREADS - 1; k >= 0; k--) {
        pthread_join(tids[k], nullptr);
        A += threads_struct[k].A;
        b += threads_struct[k].b;
    }

    // for numerical stable
    MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    // calculate Amm eigen value
    SelfAdjointEigenSolver<MatrixXd> saes(Amm);

    // get inverse of Amm
    MatrixXd Amm_inv =
        saes.eigenvectors() * VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

    // schur complication
    VectorXd bmm = b.segment(0, m);
    MatrixXd Amr = A.block(0, m, m, n);
    MatrixXd Arm = A.block(m, 0, n, m);
    MatrixXd Arr = A.block(m, m, n, n);
    VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    SelfAdjointEigenSolver<MatrixXd> saes2(A);
    // eigen values ​​greater than 0
    VectorXd S =
        VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    // inverse eigen values ​​greater than 0
    VectorXd S_inv =
        VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // get sqrt
    VectorXd S_sqrt = S.cwiseSqrt();
    VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // for ceres optimization
    // recover J from the information matrix A, A = J^T * J
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // recover the residual r from b, b = J^T r
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

vector<double *> MarginalizationInfo::getParameterBlocks(unordered_map<long, double *> &addr_shift) {
    // after Marg, some data of the variable is reserved, which is only used for the next optimization,
    // and the prior residual is added, which is not used for the next Marg
    vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    // traverse parameter_block_idx
    for (const auto &it : parameter_block_idx) {
        // variable which is keeped
        if (it.second >= m) {
            keep_block_size.push_back(parameter_block_size[it.first]);

            keep_block_idx.push_back(parameter_block_idx[it.first]);

            keep_block_data.push_back(parameter_block_data[it.first]);

            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) :
    marginalization_info(_marginalization_info) {
    // set retain parameters
    for (auto it : marginalization_info->keep_block_size) {
        mutable_parameter_block_sizes()->push_back(it);
    }

    // set prior residual dimension
    set_num_residuals(marginalization_info->n);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters,
                                     double *residuals,
                                     double **jacobians) const {
    // number of variables retained after the last Marg operation
    int n = marginalization_info->n;
    // number of variables deleted after the last Marg operation
    int m = marginalization_info->m;
    VectorXd dx(n);
    // traverse the parameter block reserved by this marginalization operation
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
        // size of variable block
        int size = marginalization_info->keep_block_size[i];
        // index of variable block, when saving before, m was not subtracted, and m is subtracted here, starting from 0
        int idx = marginalization_info->keep_block_idx[i] - m;
        // get variables value after the optimization and before the marginalization
        VectorXd x = Map<const VectorXd>(parameters[i], size);
        // variables value after the last marginalization
        VectorXd x0 = Map<const VectorXd>(marginalization_info->keep_block_data[i], size);

        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else {
            // quaternion compute
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) =
                2.0 * (Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if ((Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Quaterniond(x(6), x(3), x(4), x(5))).w() < 0) {
                dx.segment<3>(idx + 3) = 2.0 * -(Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // update residual, r = r0 + J0*dx taylor expansion
    Map<VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

    // calculate jacobian
    if (jacobians) {
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
            if (jacobians[i]) {
                int size = marginalization_info->keep_block_size[i];
                int local_size = localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                // fixed jacobian, no changed
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
} // namespace FLOW_VINS