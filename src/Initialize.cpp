/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: visual sfm initial
 */

#include "../include/Initialize.h"

namespace FLOW_VINS {
GlobalSFM::GlobalSFM() = default;

void GlobalSFM::triangulatePoint(Matrix<double, 3, 4> &Pose0,
                                 Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1,
                                 Vector3d &point_3d) const {
    Matrix4d design_matrix = Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_feature) const {
    vector<cv::Point2f> pts_2_vector;
    vector<cv::Point3f> pts_3_vector;
    // traverse feature points
    for (auto j = 0; j < m_feature_num; j++) {
        // skip sfm feature points which haven't been triangulated
        if (!sfm_feature[j].state)
            continue;
        Vector2d point2d;
        // traverse observation frame which include feature points
        for (auto k = 0; k < sfm_feature[j].observation.size(); k++) {
            // find i st frame in sfm_feature
            if (sfm_feature[j].observation[k].first == i) {
                // get 2d camera points and 3d world points(here is points in frame l coordinate)
                Vector2d img_pts = sfm_feature[j].observation[k].second;
                cv::Point2f pts_2(static_cast<float>(img_pts(0)),
                                  static_cast<float>(img_pts(1)));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(static_cast<float>(sfm_feature[j].position[0]),
                                  static_cast<float>(sfm_feature[j].position[1]),
                                  static_cast<float>(sfm_feature[j].position[2]));
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }
    // if feature points observed from the current frame is too few
    if (pts_2_vector.size() < 15) {
        cout << "unstable features tracking, please slowly move you device!" << endl;
        if (pts_2_vector.size() < 10)
            return false;
    }

    // transfer cv::solvePnP function to solve transformation between frame i and frame l
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true);
    if (!pnp_succ) {
        return false;
    }
    // convert r vector to r matrix by rodrigues method
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    // R_initial: R(c<--c0) P_initial: P(c<--c0)
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0, Matrix<double, 3, 4> &Pose0,
                                     int frame1, Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_feature) const {
    assert(frame0 != frame1);
    // traverse feature points
    for (auto j = 0; j < m_feature_num; j++) {
        // if point has been triangulated
        if (sfm_feature[j].state)
            continue;
        bool has_0 = false, has_1 = false;
        Vector2d point0;
        Vector2d point1;
        // find correspondent point between frame 0 and frame 1
        for (auto &k : sfm_feature[j].observation) {
            if (k.first == frame0) {
                point0 = k.second;
                has_0 = true;
            }
            if (k.first == frame1) {
                point1 = k.second;
                has_1 = true;
            }
        }
        // if feature point is observed by two frames
        if (has_0 && has_1) {
            // SVD method to calculate triangulate point
            Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_feature[j].state = true;
            // sfm feature add triangulate point
            sfm_feature[j].position[0] = point_3d(0);
            sfm_feature[j].position[1] = point_3d(1);
            sfm_feature[j].position[2] = point_3d(2);
        }
    }
}

bool GlobalSFM::construct(int frame_num, Quaterniond *Q, Vector3d *T, int l,
                          const Matrix3d &relative_R, const Vector3d &relative_T,
                          vector<SFMFeature> &sfm_feature,
                          map<int, Vector3d> &sfm_tracked_points) {
    m_feature_num = static_cast<int>(sfm_feature.size());
    // initial two view, set frame l to reference coordinate (world coordinate), Q[l] | T[l]: c0 frame, Q[frame-1] | T[frame-1]: R(c0<--ck), P(c0<--ck)
    Q[l].setIdentity();
    T[l].setZero();
    // the current frame
    Q[frame_num - 1] = Q[l] * Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

    // rotate to camera coordinate
    Matrix3d camera_rotation[frame_num];
    Vector3d camera_translation[frame_num];
    Quaterniond camera_quaternion[frame_num];
    // variable for ceres solver
    double camera_rotation_ba[frame_num][4];
    double camera_translation_ba[frame_num][3];
    Matrix<double, 3, 4> pose[frame_num];

    // Tcw in frame l, actually change nothing
    camera_quaternion[l] = Q[l].inverse();
    camera_rotation[l] = camera_quaternion[l].toRotationMatrix();
    camera_translation[l] = -1 * (camera_rotation[l] * T[l]);
    pose[l].block<3, 3>(0, 0) = camera_rotation[l];
    pose[l].block<3, 1>(0, 3) = camera_translation[l];

    // T(ck<--c0) in the current frame
    camera_quaternion[frame_num - 1] = Q[frame_num - 1].inverse();
    camera_rotation[frame_num - 1] = camera_quaternion[frame_num - 1].toRotationMatrix();
    camera_translation[frame_num - 1] = -1 * (camera_rotation[frame_num - 1] * T[frame_num - 1]);
    pose[frame_num - 1].block<3, 3>(0, 0) = camera_rotation[frame_num - 1];
    pose[frame_num - 1].block<3, 1>(0, 3) = camera_translation[frame_num - 1];

    // 1. triangulate between l, l+1,...,current frame - 1 and the current frame
    // 2. solve pnp l+1,...current frame - 1
    for (auto i = l; i < frame_num - 1; i++) {
        // solve pnp
        if (i > l) {
            // T(ci-->c0) in the previous frame
            Matrix3d R_initial = camera_rotation[i - 1];
            Vector3d P_initial = camera_translation[i - 1];
            // 3d-2d PnP calculate Twc in frame i
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_feature))
                return false;
            camera_rotation[i] = R_initial;
            camera_translation[i] = P_initial;
            camera_quaternion[i] = camera_rotation[i];
            pose[i].block<3, 3>(0, 0) = camera_rotation[i];
            pose[i].block<3, 1>(0, 3) = camera_translation[i];
        }

        // triangulate point between frame i and the current frame based on to solve pnp result
        triangulateTwoFrames(i, pose[i], frame_num - 1, pose[frame_num - 1], sfm_feature);
    }

    // 3. triangulate point between frame l+1,...,current frame and frame l
    for (auto i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, pose[l], i, pose[i], sfm_feature);

    // 4. solve pnp 0,..,l-1 and triangulate point between frame 0,..,,l-1 and frame l
    for (auto i = l - 1; i >= 0; i--) {
        // solve pnp
        Matrix3d R_initial = camera_rotation[i + 1];
        Vector3d P_initial = camera_translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_feature))
            return false;
        camera_rotation[i] = R_initial;
        camera_translation[i] = P_initial;
        camera_quaternion[i] = camera_rotation[i];
        pose[i].block<3, 3>(0, 0) = camera_rotation[i];
        pose[i].block<3, 1>(0, 3) = camera_translation[i];

        triangulateTwoFrames(i, pose[i], l, pose[l], sfm_feature);
    }
    // 5. triangulate all other untriangulated points
    for (auto j = 0; j < m_feature_num; j++) {
        if (sfm_feature[j].state)
            continue;
        // feature points observed frame number greater than 2
        if (sfm_feature[j].observation.size() >= 2) {
            Vector2d point0, point1;
            int frame_0 = sfm_feature[j].observation[0].first;
            point0 = sfm_feature[j].observation[0].second;
            int frame_1 = sfm_feature[j].observation.back().first;
            point1 = sfm_feature[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(pose[frame_0], pose[frame_1], point0, point1, point_3d);
            // save triangulate points
            sfm_feature[j].state = true;
            sfm_feature[j].position[0] = point_3d(0);
            sfm_feature[j].position[1] = point_3d(1);
            sfm_feature[j].position[2] = point_3d(2);
        }
    }

    // 6. full BA
    ceres::Problem problem;
    ceres::Manifold *local_parameterization = new ceres::QuaternionManifold();

    for (auto i = 0; i < frame_num; i++) {
        // double array for ceres
        camera_translation_ba[i][0] = camera_translation[i].x();
        camera_translation_ba[i][1] = camera_translation[i].y();
        camera_translation_ba[i][2] = camera_translation[i].z();
        camera_rotation_ba[i][0] = camera_quaternion[i].w();
        camera_rotation_ba[i][1] = camera_quaternion[i].x();
        camera_rotation_ba[i][2] = camera_quaternion[i].y();
        camera_rotation_ba[i][3] = camera_quaternion[i].z();
        problem.AddParameterBlock(camera_rotation_ba[i], 4, local_parameterization);
        problem.AddParameterBlock(camera_translation_ba[i], 3);
        // if frame i =l or the current frame, set parameter to constant
        if (i == l || i == frame_num - 1) {
            problem.SetParameterBlockConstant(camera_translation_ba[i]);
        }
    }

    // traverse feature points after triangulated
    for (auto i = 0; i < m_feature_num; i++) {
        // skip feature points which is not be triangulated
        if (!sfm_feature[i].state)
            continue;
        // traverse observe frame
        for (auto j = 0; j < sfm_feature[i].observation.size(); j++) {
            int k = sfm_feature[i].observation[j].first;

            // reprojection error, calculate point error with before triangulate and after triangulate points
            ceres::CostFunction *cost_function =
                ReprojectionError3D::Create(sfm_feature[i].observation[j].second.x(),
                                            sfm_feature[i].observation[j].second.y());

            // residual: R(c-->c0), t(c-->c0), 3d feature point in world (c0) coordinate
            problem.AddResidualBlock(cost_function, nullptr, camera_rotation_ba[k],
                                     camera_translation_ba[k], sfm_feature[i].position);
        }
    }

    // set ceres solver parameters
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost >= 5e-03) {
        return false;
    }
    // convert ceres parameter to Eigen
    for (auto i = 0; i < frame_num; i++) {
        Q[i].w() = camera_rotation_ba[i][0];
        Q[i].x() = camera_rotation_ba[i][1];
        Q[i].y() = camera_rotation_ba[i][2];
        Q[i].z() = camera_rotation_ba[i][3];
        // R(ci<--c0) --> R(c0<--ci)
        Q[i] = Q[i].inverse();
    }
    for (auto i = 0; i < frame_num; i++) {
        // T(ci<--c0) --> T(c0<--ci)
        T[i] = -1 * (Q[i] * Vector3d(camera_translation_ba[i][0], camera_translation_ba[i][1], camera_translation_ba[i][2]));
    }
    // save sfm track point (after ceres optimization), 3d point in frame l (c0) coordinate
    for (auto &feature : sfm_feature) {
        if (feature.state)
            sfm_tracked_points[feature.id] = Vector3d(feature.position[0], feature.position[1], feature.position[2]);
    }
    return true;
}

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs) {
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // rotation between frame i and frame j calculated by visual should be equal to pre-integration value
    // traverse from the first frame to the penultimate frame in sliding window
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
        frame_j = next(frame_i);

        MatrixXd tmp_A(3, 3);
        VectorXd tmp_b(3);
        tmp_A.setZero();
        tmp_b.setZero();
        // get rotation from frame j to frame i  |  R(c0<--ii)^-1 * R(c0<--ij) = R(ii<--ij)
        Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // tmp_A is sub jacobian matrix correspond to theta and gyro bias
        // tmp_b is 2 * Rji(pre-integration) * Rij(visual), extract the imaginary part of quaternion
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        // convert to positive definite so that could solve problem by cholesky decomposition
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    ROS_INFO_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // update gyro bias
    for (auto i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // recalculate pre-integration after updating gyro bias
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd tangentBasis(Vector3d &g0) {
    // convert g0 vector to a, a is a copy of g0 normalized
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    // schimidt orthogonalization to get vector b and calculate c through a X b to get a linearly independent set of basis in tangent space
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void refineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
    // g0 = normalized g_c0 * norm if g_world, to ensure norm of g_c0 is equal to G
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    int all_frame_count = static_cast<int>(all_image_frame.size());
    // optimize parameters: v0,...,vn, g_w1, g_w2, scale(no depth)
    int n_state = DEPTH ? (all_frame_count * 3 + 2) : (all_frame_count * 3 + 2 + 1);
    int state_dim = DEPTH ? 8 : 9;
    int gravity_pos = DEPTH ? 2 : 3;

    MatrixXd A(n_state, n_state);
    A.setZero();
    VectorXd b(n_state);
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // iterate 4 times to get optimized g vector
    for (auto k = 0; k < 4; k++) {
        MatrixXd lxly(3, 2);
        // get vector b1 and b2
        lxly = tangentBasis(g0);
        int i = 0;
        // traverse from the first frame to the penultimate frame in sliding window
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, state_dim);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            // time interval
            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            if (!DEPTH)
                tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();
            // convert to positive definite so that could solve problem by cholesky decomposition
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;
            // add state variable
            if (!DEPTH) {
                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();

                A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            } else {
                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
                b.tail<2>() += r_b.tail<2>();

                A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
                A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
            }
        }
        // multiply 1000 to ensure numerical stability
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - gravity_pos);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
    }
    // get refine gravity vector
    g = g0;
}

bool linearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
    int all_frame_count = static_cast<int>(all_image_frame.size());
    // optimize parameters: v0,...,vn, g_c0, scale
    int n_state = DEPTH ? (all_frame_count * 3 + 3) : (all_frame_count * 3 + 3 + 1);
    int state_dim = DEPTH ? 9 : 10;
    int gravity_pos = DEPTH ? 3 : 4;

    MatrixXd A(n_state, n_state);
    A.setZero();
    VectorXd b(n_state);
    b.setZero();
    // 1. initialize speed, gravity and scale factor
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // traverse from the first frame to the penultimate frame in sliding window
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, state_dim);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        // time interval
        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = 0.5 * frame_i->second.R.transpose() * dt * dt;
        // here divide 100 so scale factor should be divided 100
        if (!DEPTH)
            tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt;

        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();

        // convert to positive definite so that could solve problem by cholesky decomposition
        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // r_A(10 * 10)
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // r_b(10 * 1)

        // add state variable
        if (!DEPTH) {
            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
        } else {
            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
    }
    // multiply 1000 to ensure numerical stability
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    // get g_c0 factor
    g = x.segment<3>(n_state - gravity_pos);
    //INFO_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // numerical check
    if (fabs(g.norm() - G.norm()) > 0.5) {
        return false;
    }
    // 2. refine gravity vector to get more accuracy result
    refineGravity(all_image_frame, g, x);
    ROS_INFO_STREAM(" refine     " << g.norm() << " " << g.transpose());

    // get scale factor
    if (!DEPTH) {
        double s = (x.tail<1>())(0) / 100.0;
        ROS_DEBUG("estimated scale: %f", s);
        (x.tail<1>())(0) = s;
        if (s < 0.0)
            return false;
        else
            return true;
    }
    return true;
}

bool visualImuAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x) {
    // update gyro bias and update pre-integration
    solveGyroscopeBias(all_image_frame, Bgs);
    // initialize speed, gravity and scale factor
    if (linearAlignment(all_image_frame, g, x))
        return true;
    else
        return false;
}
} // namespace FLOW_VINS