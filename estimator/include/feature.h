/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: front end
 */

#pragma once

#include "../../thirdparty/cameramodels/camerafactory.h"
#include "../../thirdparty/cameramodels/catacamera.h"
#include "../../thirdparty/cameramodels/pinholecamera.h"
#include "parameter.h"
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <execinfo.h>
#include <iostream>
#include <list>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

#ifdef GPU_MODE
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaoptflow.hpp>
#endif

namespace FLOW_VINS {

typedef Eigen::Matrix<double, 7, 1> TrackFeatureNoId;
typedef std::pair<int, TrackFeatureNoId> TrackFeature;
typedef std::vector<TrackFeature> FeatureFramenoId;
typedef std::map<int, FeatureFramenoId> FeatureFrame;

/**
 * @brief: feature point level
 */
enum FeatureLevel { REMOVE = 1, DYNAMIC, UN_INITIAL, OPTIMIZE };

class FeatureTracker {
public:
	/**
	 * @brief: constructor function, reset state parameters
	 */
	FeatureTracker();

	/**
	 * @brief: set camera model from calib file
	 */
	void readIntrinsicParameter(const std::vector<std::string>& calib_file);

	/**
	 * @brief: draw a circle in feature point, and sort feature point vector based on track times
	 */
	void setMask();

	/**
	 * @brief: input semantic mask (only for rgb-d camera)
	 */
	void setSemanticMask(const cv::Mat& semantic);

	/**
	 * @brief: check if point is in semantic mask
	 */
	bool inSemantic(const cv::Point& p);

	/**
	 * @brief: check if point is in image border
	 */
	bool inBorder(const cv::Point2f& pt) const;

	/**
	 * @brief: get distance between two points
	 */
	static double distance(cv::Point2f& pt1, cv::Point2f& pt2);

	/**
	 * @brief: main process of feature tracking
	 */
	FeatureFrame trackImage(double _cur_time, const cv::Mat& img_0, const cv::Mat& img_1 = cv::Mat());

	/**
	 * @brief: get normalized camera plane points calculated from pixels, with distortion correction
	 */
	static std::vector<cv::Point2f> undistortedPts(std::vector<cv::Point2f>& pts, const CameraModel::CameraPtr& cam);

	/**
	 * @brief: calculate the moving speed of the current frame normalized camera plane feature points in the x and y
	 * directions
	 */
	std::vector<cv::Point2f> ptsVelocity(std::vector<int>& ids, std::vector<cv::Point2f>& pts,
	                                     std::map<int, cv::Point2f>& cur_id_pts,
	                                     std::map<int, cv::Point2f>& prev_id_pts);

	/**
	 * @brief: calculate fundamental matrix and further eliminate outliers
	 */
	void rejectWithF();

	/**
	 * @brief: estimate the position of feature points in the current frame using the motion of the previous frame
	 */
	void setPrediction(std::map<int, Eigen::Vector3d>& _predict_pts);

	/**
	 * @brief: specify the outlier point, and delete
	 */
	void removeOutliers(std::set<int>& remove_pts_ids);

	/**
	 * @brief: for visualization, the feature points in the left image are tracked by color (less red, more blue), green
	 * point in the right image
	 */
	void drawTrack(const cv::Mat& img_left, const cv::Mat& img_right, std::vector<int>& _cur_left_ids,
	               std::vector<cv::Point2f>& _cur_left_pts, std::vector<cv::Point2f>& _cur_right_pts,
	               std::map<int, cv::Point2f>& _prev_left_pts_map);

	/**
	 * @brief: get tracking image for rviz display
	 */
	cv::Mat getTrackImage();

	/**
	 * @brief: set feature point level
	 */
	void setFeatureStatus(int feature_id, int status);

	/**
	 * @brief: get feature point level
	 */
	std::map<int, int> getFeatureStatus();

private:
	// image row and col
	int row{}, col{};
	// tracking image (with feature point)
	cv::Mat img_track;
	// for optical flow tracking min distance
	cv::Mat mask;
	// right image for binocular
	cv::Mat right_img;
	// segment image
	cv::Mat semantic_img;
	// image for optical flow
	cv::Mat prev_img, cur_img;
	// corner point vector
	std::vector<cv::Point2f> n_pts;
	// predict point
	std::vector<cv::Point2f> predict_pts;
	// feature point (previous frame, current frame, right frame)
	std::vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
	// feature point with distortion correction
	std::vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
	// point velocity in normalized camera coordinate
	std::vector<cv::Point2f> pts_velocity, right_pts_velocity;
	// feature point id vector
	std::vector<int> ids, ids_right;
	// tracking number
	std::vector<int> track_cnt;
	// map with feature point id and position, for calculate velocity
	std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
	std::map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
	// map in previous frame, only for visualization
	std::map<int, cv::Point2f> prev_left_pts_map;
	// camera model
	std::vector<CameraModel::CameraPtr> camera;
	// feature point id
	int n_id;
	// time stamp
	double cur_time{};
	double prev_time{};
	// state flag
	bool use_stereo_cam;
	bool use_rgbd_cam;
	bool has_prediction;

	// feature level manager
	std::map<int, int> pts_status;
	std::set<int> removed_pts;
};

/**
 * @brief: class for each feature point information in one frame
 */
class FeaturePerFrame {
public:
	/**
	 * @brief: constructor function, build point information in one frame
	 */
	FeaturePerFrame(const TrackFeatureNoId& _point, double td) {
		point.x() = _point(0);
		point.y() = _point(1);
		point.z() = _point(2);
		uv.x() = _point(3);
		uv.y() = _point(4);
		velocity.x() = _point(5);
		velocity.y() = _point(6);
		cur_td = td;
		depth = 0;
		is_stereo = false;
		is_depth = false;
	}

	/**
	 * @brief: if camera is stereo, add right camera observation
	 */
	void rightObservation(const TrackFeatureNoId& _point) {
		point_right.x() = _point(0);
		point_right.y() = _point(1);
		point_right.z() = _point(2);
		uv_right.x() = _point(3);
		uv_right.y() = _point(4);
		velocity_right.x() = _point(5);
		velocity_right.y() = _point(6);
		is_stereo = true;
	}

	/**
	 * @brief: if camera is rgb-d, add depth information
	 */
	void inputDepth(double depth_) {
		depth = depth_;
		is_depth = true;
	}

	double cur_td;
	// normalized camera point
	Eigen::Vector3d point, point_right;
	// pixel point
	Eigen::Vector2d uv, uv_right;
	// normalized camera coordinate velocity
	Eigen::Vector2d velocity, velocity_right;
	// feature point level

	bool is_stereo;
	bool is_depth;
	double depth;
};

/**
 * @brief: class for each feature point observed by multiple consecutive frame
 */
class FeaturePerId {
public:
	/**
	 * @brief: constructor function, initial parameters
	 */
	FeaturePerId(int _feature_id, int _start_frame)
	    : feature_id(_feature_id)
	    , start_frame(_start_frame)
	    , used_num(0)
	    , estimated_depth(-1.0)
	    , solve_flag(0)
	    , estimate_flag(0) {}

	FeaturePerId() = default;

	/**
	 * @brief: get end frame index
	 */
	int endFrame() const;

	const int feature_id = -1;
	int start_frame = -1;
	int used_num = 0;
	// vector for observe frame in one feature point
	std::vector<FeaturePerFrame> feature_per_frame;
	double estimated_depth = -1;
	// 0 haven't solve yet; 1 solve successful; 2 solve fail;
	int solve_flag = 0;
	// 0 initial; 1 by depth image; 2 by triangulate
	int estimate_flag = 0;
};

/**
 * @brief: class for feature manager
 */
class FeatureManager {
public:
	/**
	 * @brief: constructor function, initial parameters
	 */
	FeatureManager();

	/**
	 * @brief: clear feature point
	 */
	void clearState();

	/**
	 * @brief: set extrinsic matrix from camera to imu
	 */
	void setRic(Eigen::Matrix3d _ric[]);

	/**
	 * @brief: input depth image (only rgb-d camera used)
	 */
	void setDepthImage(const cv::Mat& depth);

	/**
	 * @brief: returns the number of feature points with more than 2 observation frames
	 */
	int getFeatureCount();

	/**
	 * @brief: add one feature frame and check if it is key frame
	 */
	bool addFeatureCheckParallax(int frame_count, FeatureFrame& image, double td);

	/**
	 * @brief: extract the matching points of two frames, requiring these points to be tracked between the two frames
	 */
	std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

	/**
	 * @brief: set the depth value after feature point optimization, if it is less than 0, mark solve_flag as 2 (solve
	 * is failed)
	 */
	void setDepth(const std::map<int, double>& deps);

	/**
	 * @brief: delete feature points with negative depth values ​​after optimization
	 */
	void removeFailures();

	/**
	 * @brief: reset depth as -1
	 */
	void clearDepth();

	/**
	 * @brief: get the inverse depth of feature points in the current frame, limited to feature points with more than 2
	 * observation frames
	 */
	std::map<int, double> getDepthVector();

	/**
	 * @brief: calculate triangulate point by SVD method
	 */
	static void triangulatePoint(Eigen::Matrix<double, 3, 4>& Pose0, Eigen::Matrix<double, 3, 4>& Pose1,
	                             Eigen::Vector2d& point0, Eigen::Vector2d& point1, Eigen::Vector3d& point_3d);

	/**
	 * @brief: solve pose with pnp method
	 */
	static bool solvePoseByPnP(Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial, std::vector<cv::Point2f>& pts2D,
	                           std::vector<cv::Point3f>& pts3D);

	/**
	 * @brief: 3d-2d pnp solves the current frame pose
	 */
	void initFramePoseByPnP(int frame_cnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[],
	                        Eigen::Matrix3d ric[]);

	/**
	 * @brief: triangulate current frame feature points
	 */
	void triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

	/**
	 * @brief: delete outlier feature points, calculate the re-projection error after sliding window optimization, and
	 * remove points with excessive errors
	 */
	void removeOutlier(std::set<int>& outlier_index);

	/**
	 * @brief: after marginalizing the first frame, delete the frame from the observation frame set of the feature
	 * point, update depth
	 */
	void removeBackShiftDepth(const Eigen::Matrix3d& marg_R, const Eigen::Vector3d& marg_P,
	                          const Eigen::Matrix3d& new_R, const Eigen::Vector3d& new_P);

	/**
	 * @brief: after marginalizing the first frame, delete the frame from the observation frame set of the feature point
	 */
	void removeBack();

	/**
	 * @brief: after marginalizing the previous frame of the current frame, delete the frame from the observation frame
	 * set of the feature point.
	 */
	void removeFront(int frame_count);

	/**
	 * @brief: calculate the distance of the current feature point in the last 3 and 2 frames, and normalize the
	 * distance on the camera plane
	 */
	static double compensatedParallax2(const FeaturePerId& it_per_id, int frame_count);

	int last_track_num{};
	int new_feature_num{};
	int long_track_num{};

	// list<FeaturePerId> feature;
	std::map<int, FeaturePerId> feature;

	FeatureTracker ft;
	cv::Mat depth_img;
	Eigen::Matrix3d ric[2];

	// healthy check
	double stable_scale;
};
} // namespace FLOW_VINS