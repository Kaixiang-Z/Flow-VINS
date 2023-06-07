/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: pose graph
 */

#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "../../thirdparty/DBoW/DBoW2.h"
#include "../../thirdparty/DVision/DVision.h"
#include "../../thirdparty/CameraModel/camera_factory.h"
#include "../../thirdparty/CameraModel/cata_camera.h"
#include "../../thirdparty/CameraModel/pinhole_camera.h"
#include "common.h"
#include "parameters.h"
#include "publisher.h"

using namespace std;
using namespace Eigen;
using namespace DVision;
using namespace DBoW2;

namespace FLOW_VINS {

/**
 * @brief: parameters for relocation
 */
class RelocationFrame {
public:
    // relocation state
    int index;
    double time;
    vector<Vector3d> relo_uv_id;
    Matrix3d relo_R;
    Vector3d relo_T;
};

/**
 * @brief: class for extract brief point
 */
class BriefExtractor {
public:
    /**
     * @brief: load brief pattern file
     */
    explicit BriefExtractor(const string &pattern_file);
    /**
     * @brief: compute brief descriptor with brief pattern file 
     */
    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;

    DVision::BRIEF m_brief;
};

/**
 * @brief: class for build new keyframe
 */
class KeyFrame {
public:
    /**
     * @brief: constructor function for create a new keyframe 
     */
    KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
             vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
             vector<double> &_point_id, int _sequence);

    /**
     * @brief: compute each brief descriptor of feature point in one image
     */
    void computeWindowBRIEFPoint();

    /**
     * @brief: extra extract feature points and compute brief descriptor for loop detect 
     */
    void computeBRIEFPoint();

    /**
     * @brief: compute hamming distance between two brief descriptor
     */
    static int hammingDistance(const BRIEF::bitset &a, const BRIEF::bitset &b);

    /**
     * @brief: the brief descriptor of a feature point in the key frame matches all the descriptors of the loop frame
     */
    static bool searchInAera(const BRIEF::bitset &window_descriptor,
                             const vector<BRIEF::bitset> &descriptors_old,
                             const vector<cv::KeyPoint> &keypoints_old,
                             const vector<cv::KeyPoint> &keypoints_old_norm,
                             cv::Point2f &best_match,
                             cv::Point2f &best_match_norm);

    /**
     * @brief: match the keyframe with the loopback frame for the BRIEF descriptor 
     */
    void searchByBRIEFDes(vector<cv::Point2f> &matched_2d_old,
                          vector<cv::Point2f> &matched_2d_old_norm,
                          vector<uchar> &status,
                          const vector<BRIEF::bitset> &descriptors_old,
                          const vector<cv::KeyPoint> &keypoints_old,
                          const vector<cv::KeyPoint> &keypoints_old_norm);

    /**
     * @brief: find and establish the matching relationship between the keyframe and the loopframe, return True to confirm the formation of the loop
     */
    bool findConnection(KeyFrame *old_kf, queue<RelocationFrame> &relo_frame_buf);

    /**
     * @brief: use pnp ransac method to solve R & T and remove outliers
     */
    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const vector<cv::Point3f> &matched_3d,
                   vector<uchar> &status,
                   Vector3d &PnP_T_old, Matrix3d &PnP_R_old);

    /**
     * @brief: get vio R & T 
     */
    void getVioPose(Vector3d &_T_w_i, Matrix3d &_R_w_i) const;

    /**
     * @brief: get loop fusion R & T 
     */
    void getPose(Vector3d &_T_w_i, Matrix3d &_R_w_i) const;

    /**
     * @brief: update vio R & T and set loop fusion R & T same as vio R & T 
     */
    void updateVioPose(const Vector3d &_T_w_i, const Matrix3d &_R_w_i);

    /**
     * @brief: update loop fusion R & T 
     */
    void updatePose(const Vector3d &_T_w_i, const Matrix3d &_R_w_i);

    /**
     * @brief: get loop fusion relative T 
     */
    Vector3d getLoopRelativeT();

    /**
     * @brief: get loop fusion relative R
     */
    Quaterniond getLoopRelativeQ();

    /**
     * @brief: get loop fusion relative yaw 
     */
    double getLoopRelativeYaw();

    // keyframe time stamp
    double time_stamp;
    // keyframe index
    int index;
    // keyframe local index
    int local_index;
    // vio pose
    Vector3d vio_T_w_i;
    Matrix3d vio_R_w_i;

    // loop fusion pose
    Vector3d T_w_i;
    Matrix3d R_w_i;

    // initial pose
    Vector3d origin_vio_T;
    Matrix3d origin_vio_R;
    // left image
    cv::Mat image;
    cv::Mat semantic_img;
    cv::Mat thumbnail;
    // keyframe world coordinate correspond to estimator node
    vector<cv::Point3f> point_3d;
    // keyframe pixel coordinate correspond to estimator node
    vector<cv::Point2f> point_2d_uv;
    // normalized camera coordinate correspond to estimator node
    vector<cv::Point2f> point_2d_norm;
    // map points id vector
    vector<double> point_id;
    // keypoint detected
    vector<cv::KeyPoint> keypoints;
    // keypoint in normalize camera coordinate
    vector<cv::KeyPoint> keypoints_norm;
    // feature point in current frame
    vector<cv::KeyPoint> window_keypoints;
    // extra brief descriptor vector of feature points
    vector<BRIEF::bitset> brief_descriptors;
    // brief descriptor vector of feature points in current frame
    vector<BRIEF::bitset> window_brief_descriptors;
    bool has_fast_point;
    // sequence number
    int sequence;
    // detect loop
    bool has_loop;
    // loop frame index
    int loop_index;
    // loop info (relative_T, relative_R, relative_yaw)
    Matrix<double, 8, 1> loop_info;
};

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
    void loadVocabulary(const string &voc_path);

    /**
     * @brief: confirm IMU is enabled, start 4 & 6 DOF optimize thread
     */
    void setIMUFlag(bool _use_imu);

    /**
     * @brief: add one keyframe
     */
    void addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);

    int sequence_cnt;
    // ros publish state
    nav_msgs::Path path[10];
    nav_msgs::Path base_path;
    // drift state
    Vector3d t_drift;
    Matrix3d r_drift;
    double yaw_drift;

    Vector3d w_t_vio;
    Matrix3d w_r_vio;

    // state
    Vector3d Ps{};
    Matrix3d Rs{};

    // relocation frame
    queue<RelocationFrame> relo_frame_buf;

private:
    /**
     * @brief: find old keyframe with index
     */
    KeyFrame *getKeyFrame(int index);
    /**
     * @brief: main process of loop detect 
     */
    int detectLoop(KeyFrame *keyframe, int frame_index);

    /**
     * @brief: add keyframe into vocabulary, only for visualization 
     */
    void addKeyFrameIntoVoc(KeyFrame *keyframe);

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

    queue<int> optimize_buf;
    list<KeyFrame *> keyframelist;
    vector<bool> sequence_loop;

    // index
    int global_index;
    int earliest_loop_index;
    bool use_imu;

    // brief descriptor
    BriefDatabase db;
    BriefVocabulary *voc{};
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
    bool Plus(const T *theta_radians, const T *delta_theta_radians, T *theta_radians_plus_delta) const {
        *theta_radians_plus_delta = Utility::normalizeAngle(*theta_radians + *delta_theta_radians);

        return true;
    }

    /**
     * @brief: implements minus operation for the manifold
     */
    template <typename T>
    bool Minus(const T *theta_radians, const T *delta_theta_radians, T *theta_radians_plus_delta) const {
        return true;
    }

    /**
     * @brief: create auto difference manifold, compute jacobian and residual
     */
    static ceres::Manifold *Create() {
        return (new ceres::AutoDiffManifold<AngleLocalParameterization, 1, 1>);
    }
};

/**
 * @brief: struct for 4 DoF cost function 
 */
struct FourDOFError {
    /**
     * @brief: constructor function, initial parameters
     */
    FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i) :
        t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
    }

    /**
     * @brief: compute residual
     */
    template <typename T>
    bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const {
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
    static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
                                       const double relative_yaw, const double pitch_i, const double roll_i) {
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
    SixDOFError(double t_x, double t_y, double t_z,
                double q_w, double q_x, double q_y, double q_z,
                double t_var, double q_var) :
        t_x(t_x),
        t_y(t_y), t_z(t_z),
        q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
        t_var(t_var), q_var(q_var) {
    }

    /**
     * @brief: compute residual
     */
    template <typename T>
    bool operator()(const T *const w_q_i, const T *ti, const T *w_q_j, const T *tj, T *residuals) const {
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
     * @brief: create cost function, residual dimension: 6 (x, y, z, yaw, pitch, roll), q_i (4), t_i(3), q_j (4), xyz_j(3) 
     */
    static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
                                       const double q_w, const double q_x, const double q_y, const double q_z,
                                       const double t_var, const double q_var) {
        return (new ceres::AutoDiffCostFunction<SixDOFError, 6, 4, 3, 4, 3>(
            new SixDOFError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
    }

    // state parameters
    double t_x, t_y, t_z, t_norm{};
    double q_w, q_x, q_y, q_z;
    double t_var, q_var;
};
} // namespace FLOW_VINS