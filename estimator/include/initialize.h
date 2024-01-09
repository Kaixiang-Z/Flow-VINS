/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: visual sfm initial
 */

#pragma once
#include "factor.h"
#include "feature.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace FLOW_VINS {

/**
 * @brief: class for sfm feature
 */
struct SFMFeature {
	bool state;                                               // sfm state
	int id;                                                   // feature id
	std::vector<std::pair<int, Eigen::Vector2d>> observation; // normalize camera points in x frame
	double position[3];                                       // triangulate points
};

/**
 * @brief: class for calculate reprojection error
 */
struct ReprojectionError3D {
	ReprojectionError3D(double observed_u, double observed_v)
	    : observed_u(observed_u)
	    , observed_v(observed_v) {}

	/**
	 * @brief: calaulate reprojection error
	 */
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const {
		T p[3];
		// p = camera_R * point + camera_T
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0];
		p[1] += camera_T[1];
		p[2] += camera_T[2];
		// normalize camera coordinate
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];
		// calculate residual, xp & yp: 3d point correspont to feature point project to 2d, observed_u & observed_v:
		// initial 2d point
		residuals[0] = xp - T(observed_u);
		residuals[1] = yp - T(observed_v);
		return true;
	}
	/**
	 * @brief: create cost function, residual dimension: 2 (u, v), rotation dimention: 4, translation dimention: 3,
	 * feature point in world coordinate: 3
	 */
	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
		    new ReprojectionError3D(observed_x, observed_y)));
	}

	double observed_u;
	double observed_v;
};

/**
 * @brief: class for global sfm
 */
class GlobalSFM {
public:
	GlobalSFM();

	/**
	 * @brief: sfm problem construct
	 */
	bool construct(int frame_num, Eigen::Quaterniond* Q, Eigen::Vector3d* T, int l, const Eigen::Matrix3d& relative_R,
	               const Eigen::Vector3d& relative_T, std::vector<SFMFeature>& sfm_feature,
	               std::map<int, Eigen::Vector3d>& sfm_tracked_points);

private:
	/**
	 * @brief: 3d-2d PnP method to calculate pose in i frame
	 */
	bool solveFrameByPnP(Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial, int i,
	                     std::vector<SFMFeature>& sfm_feature) const;

	/**
	 * @brief: SVD method to calculate triangulate points
	 */
	void triangulatePoint(Eigen::Matrix<double, 3, 4>& Pose0, Eigen::Matrix<double, 3, 4>& Pose1,
	                      Eigen::Vector2d& point0, Eigen::Vector2d& point1, Eigen::Vector3d& point_3d) const;

	/**
	 * @brief: calculate triangulate points and put into sfm feature positions
	 */
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4>& Pose0, int frame1,
	                          Eigen::Matrix<double, 3, 4>& Pose1, std::vector<SFMFeature>& sfm_feature) const;

	int m_feature_num{};
};

/**
 * @brief: class for process image frame, include frame pre-integration, R, T, and feature points
 */
class ImageFrame {
public:
	/**
	 * @brief: constructor for ImageFrame, parameters initialize
	 */
	ImageFrame() = default;
	ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t)
	    : points(_points)
	    , t(_t)
	    , is_key_frame(false){};

	std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;
	double t{};
	Eigen::Matrix3d R;
	Eigen::Vector3d T;
	IntegrationBase* pre_integration{};
	bool is_key_frame{};
};

/**
 * @brief: use schimidt orthogonalization method to find a linearly independent set of basis in tangent space
 */
Eigen::MatrixXd tangentBasis(Eigen::Vector3d& g0);

/**
 * @brief: calculate gyro bias by ldlt solver and update IMU pre-integration
 */
void solveGyroscopeBias(std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs);

/**
 * @brief: correct gravity vector after linear alignment
 */
void refineGravity(std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d& g, Eigen::VectorXd& x);

/**
 * @brief: initialize speed, gravity and scale factor and refine gravity and scale factor
 */
bool linearAlignment(std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d& g, Eigen::VectorXd& x);

/**
 * @brief: main process of visual and Imu aligenment, include gyro bias update and initialize speed, gravity and scale
 * factor
 */
bool visualImuAlignment(std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs, Eigen::Vector3d& g,
                        Eigen::VectorXd& x);

} // namespace FLOW_VINS