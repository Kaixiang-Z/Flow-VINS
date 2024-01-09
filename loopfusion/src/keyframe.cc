/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: key frame
 */
#include "../include/keyframe.h"

CameraModel::CameraPtr m_camera;

namespace FLOW_VINS {

/**
 * @brief: delete variables in vector which value is equal to zero
 */
template <typename Derived> static void reduceVector(vector<Derived>& v, vector<uchar> status) {
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

BriefExtractor::BriefExtractor(const std::string& pattern_file) {
	// loads the pattern
	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
		throw string("Could not open file ") + pattern_file;

	vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator()(const cv::Mat& im, vector<cv::KeyPoint>& keys,
                                vector<DVision::BRIEF::bitset>& descriptors) const {
	m_brief.compute(im, keys, descriptors);
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Eigen::Vector3d& _vio_T_w_i, Eigen::Matrix3d& _vio_R_w_i,
                   cv::Mat& _image, vector<cv::Point3f>& _point_3d, vector<cv::Point2f>& _point_2d_uv,
                   vector<cv::Point2f>& _point_2d_norm, vector<double>& _point_id, int _sequence) {
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;

	computeWindowBRIEFPoint();
	computeBRIEFPoint();

	image.release();
}

void KeyFrame::computeWindowBRIEFPoint() {
	// build brief extractor
	BriefExtractor extractor(BRIEF_PATTERN_FILE);
	// convert to cv keyframe format
	for (const auto& i : point_2d_uv) {
		cv::KeyPoint key;
		key.pt = i;
		window_keypoints.push_back(key);
	}
	// compute brief descriptors
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint() {
	// build brief extractor
	BriefExtractor extractor(BRIEF_PATTERN_FILE);
	const int fast_th = 20; // corner detector response threshold
	// detect extra fast feature point
	cv::FAST(image, keypoints, fast_th, true);

	extractor(image, keypoints, brief_descriptors);
	// compute normalize camera coordinate and push to keypoints_norm
	for (auto& keypoint : keypoints) {
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(static_cast<float>(tmp_p.x() / tmp_p.z()), static_cast<float>(tmp_p.y() / tmp_p.z()));
		keypoints_norm.push_back(tmp_norm);
	}
}

int KeyFrame::hammingDistance(const DVision::BRIEF::bitset& a, const DVision::BRIEF::bitset& b) {
	// the Hamming distance is the number of different position values ​​after converting to binary
	DVision::BRIEF::bitset xor_of_bitset = a ^ b;
	int dis = static_cast<int>(xor_of_bitset.count());
	return dis;
}

bool KeyFrame::searchInAera(const DVision::BRIEF::bitset& window_descriptor,
                            const std::vector<DVision::BRIEF::bitset>& descriptors_old,
                            const std::vector<cv::KeyPoint>& keypoints_old,
                            const std::vector<cv::KeyPoint>& keypoints_old_norm, cv::Point2f& best_match,
                            cv::Point2f& best_match_norm) {
	int bestDist = 128;
	int bestIndex = -1;
	// find shortest distance between current descriptor and old descriptors
	for (int i = 0; i < (int)descriptors_old.size(); i++) {
		int dis = hammingDistance(window_descriptor, descriptors_old[i]);
		if (dis < bestDist) {
			bestDist = dis;
			bestIndex = i;
		}
	}
	// if find descriptor
	if (bestIndex != -1 && bestDist < 80) {
		best_match = keypoints_old[bestIndex].pt;
		best_match_norm = keypoints_old_norm[bestIndex].pt;
		return true;
	} else
		return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f>& matched_2d_old, std::vector<cv::Point2f>& matched_2d_old_norm,
                                std::vector<uchar>& status, const std::vector<DVision::BRIEF::bitset>& descriptors_old,
                                const std::vector<cv::KeyPoint>& keypoints_old,
                                const std::vector<cv::KeyPoint>& keypoints_old_norm) {
	// traverse brief descriptors in current frame
	for (const auto& window_brief_descriptor : window_brief_descriptors) {
		cv::Point2f pt(0.f, 0.f);
		cv::Point2f pt_norm(0.f, 0.f);
		if (searchInAera(window_brief_descriptor, descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
			status.push_back(1);
		else
			status.push_back(0);
		// get matched feature points
		matched_2d_old.push_back(pt);
		matched_2d_old_norm.push_back(pt_norm);
	}
}

bool KeyFrame::findConnection(KeyFrame* old_kf) {
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	// find relationship between current frame and loop frame, then reduce vector
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints,
	                 old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);

	status.clear();

	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Eigen::Quaterniond relative_q;
	double relative_yaw;
	// if matched point nums > MIN_LOOP_NUM, use pnp ransac delete mis-matched points
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_id, status);
	}
	// after remove mis-matched point, point num still greater than MIN_LOOP_NUM
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		relative_q = PnP_R_old.transpose() * origin_vio_R;
		relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());

		// detect loop
		if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {
			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
			    relative_q.z(), relative_yaw;
			return true;
		}
	}
	return false;
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f>& matched_2d_old_norm, const std::vector<cv::Point3f>& matched_3d,
                         std::vector<uchar>& status, Eigen::Vector3d& PnP_T_old, Eigen::Matrix3d& PnP_R_old) {
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Eigen::Matrix3d R_inital;
	Eigen::Vector3d P_inital;
	Eigen::Matrix3d R_w_c = origin_vio_R * LOOP_QIC;
	Eigen::Vector3d T_w_c = origin_vio_T + origin_vio_R * LOOP_TIC;

	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;

	cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

	// remove outlier
	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
		status.push_back(0);

	for (int i = 0; i < inliers.rows; i++) {
		int n = inliers.at<int>(i);
		status[n] = 1;
	}

	cv::Rodrigues(rvec, r);
	Eigen::Matrix3d R_pnp, R_w_c_old;
	cv::cv2eigen(r, R_pnp);
	R_w_c_old = R_pnp.transpose();
	Eigen::Vector3d T_pnp, T_w_c_old;
	cv::cv2eigen(t, T_pnp);
	T_w_c_old = R_w_c_old * (-T_pnp);

	PnP_R_old = R_w_c_old * LOOP_QIC.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * LOOP_TIC;
}

void KeyFrame::getVioPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) const {
	_T_w_i = vio_T_w_i;
	_R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) const {
	_T_w_i = T_w_i;
	_R_w_i = R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i) {
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i) {
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT() { return Eigen::Vector3d{loop_info(0), loop_info(1), loop_info(2)}; }

Eigen::Quaterniond KeyFrame::getLoopRelativeQ() {
	return Eigen::Quaterniond{loop_info(3), loop_info(4), loop_info(5), loop_info(6)};
}

double KeyFrame::getLoopRelativeYaw() { return loop_info(7); }

} // namespace FLOW_VINS
