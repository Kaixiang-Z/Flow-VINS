/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: key frame
 */

#pragma once

#include "../../thirdparty/cameramodels/camerafactory.h"
#include "../../thirdparty/cameramodels/catacamera.h"
#include "../../thirdparty/cameramodels/pinholecamera.h"
#include "../../thirdparty/dbow/dbow2.h"
#include "../../thirdparty/dvision/dvision.h"
#include "parameter.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <vector>

const int MIN_LOOP_NUM = 25;

extern CameraModel::CameraPtr m_camera;

namespace FLOW_VINS {

/**
 * @brief: class for extract brief point
 */
class BriefExtractor {
public:
	/**
	 * @brief: load brief pattern file
	 */
	explicit BriefExtractor(const std::string& pattern_file);
	/**
	 * @brief: compute brief descriptor with brief pattern file
	 */
	virtual void operator()(const cv::Mat& im, vector<cv::KeyPoint>& keys,
	                        vector<DVision::BRIEF::bitset>& descriptors) const;

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
	KeyFrame(double _time_stamp, int _index, Eigen::Vector3d& _vio_T_w_i, Eigen::Matrix3d& _vio_R_w_i, cv::Mat& _image,
	         vector<cv::Point3f>& _point_3d, vector<cv::Point2f>& _point_2d_uv, vector<cv::Point2f>& _point_2d_normal,
	         vector<double>& _point_id, int _sequence);

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
	static int hammingDistance(const DVision::BRIEF::bitset& a, const DVision::BRIEF::bitset& b);

	/**
	 * @brief: the brief descriptor of a feature point in the key frame matches all the descriptors of the loop frame
	 */
	static bool searchInAera(const DVision::BRIEF::bitset& window_descriptor,
	                         const std::vector<DVision::BRIEF::bitset>& descriptors_old,
	                         const std::vector<cv::KeyPoint>& keypoints_old,
	                         const std::vector<cv::KeyPoint>& keypoints_old_norm, cv::Point2f& best_match,
	                         cv::Point2f& best_match_norm);

	/**
	 * @brief: match the keyframe with the loopback frame for the BRIEF descriptor
	 */
	void searchByBRIEFDes(std::vector<cv::Point2f>& matched_2d_old, std::vector<cv::Point2f>& matched_2d_old_norm,
	                      std::vector<uchar>& status, const std::vector<DVision::BRIEF::bitset>& descriptors_old,
	                      const std::vector<cv::KeyPoint>& keypoints_old,
	                      const std::vector<cv::KeyPoint>& keypoints_old_norm);

	/**
	 * @brief: find and establish the matching relationship between the keyframe and the loopframe, return True to
	 * confirm the formation of the loop
	 */
	bool findConnection(KeyFrame* old_kf);

	/**
	 * @brief: use pnp ransac method to solve R & T and remove outliers
	 */
	void PnPRANSAC(const vector<cv::Point2f>& matched_2d_old_norm, const std::vector<cv::Point3f>& matched_3d,
	               std::vector<uchar>& status, Eigen::Vector3d& PnP_T_old, Eigen::Matrix3d& PnP_R_old);

	/**
	 * @brief: get vio R & T
	 */
	void getVioPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) const;

	/**
	 * @brief: get loop fusion R & T
	 */
	void getPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) const;

	/**
	 * @brief: update vio R & T and set loop fusion R & T same as vio R & T
	 */
	void updateVioPose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i);

	/**
	 * @brief: update loop fusion R & T
	 */
	void updatePose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i);

	/**
	 * @brief: get loop fusion relative T
	 */
	Eigen::Vector3d getLoopRelativeT();

	/**
	 * @brief: get loop fusion relative R
	 */
	Eigen::Quaterniond getLoopRelativeQ();

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
	Eigen::Vector3d vio_T_w_i;
	Eigen::Matrix3d vio_R_w_i;

	// loop fusion pose
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;

	// initial pose
	Eigen::Vector3d origin_vio_T;
	Eigen::Matrix3d origin_vio_R;
	// left image
	cv::Mat image;
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
	vector<DVision::BRIEF::bitset> brief_descriptors;
	// brief descriptor vector of feature points in current frame
	vector<DVision::BRIEF::bitset> window_brief_descriptors;
	bool has_fast_point;
	// sequence number
	int sequence;
	// detect loop
	bool has_loop;
	// loop frame index
	int loop_index;
	// loop info (relative_T, relative_R, relative_yaw)
	Eigen::Matrix<double, 8, 1> loop_info;
};

} // namespace FLOW_VINS