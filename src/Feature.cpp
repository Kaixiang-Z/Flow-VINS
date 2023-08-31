/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: feature extractor
 */

#include "../include/Feature.h"

namespace FLOW_VINS {

/**
 * @brief: delete variables in vector which value is equal to zero
 */
template <typename T>
static void reduceVector(vector<T> &v, vector<uchar> status) {
    int j = 0;
    for (auto i = 0; i < v.size(); ++i) {
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

FeatureTracker::FeatureTracker() {
    use_stereo_cam = false;
    use_rgbd_cam = false;
    n_id = 0;
    has_prediction = false;
}

void FeatureTracker::setMask() {
    // build a new mask
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (auto i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], make_pair(cur_pts[i], ids[i]));

    // sort feature point on the current frame, based on tracking times
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a,
            const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    // clear vector
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    // traverse feature point, reinput to point vector
    for (auto &it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
            // tracking long times is in the front
            cur_pts.emplace_back(it.second.first);
            ids.emplace_back(it.second.second);
            track_cnt.emplace_back(it.first);
            // draw a circle in feature points, for optical flow min distance
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::setSemanticMask(const cv::Mat &semantic) {
    semantic_img = semantic;
}

bool FeatureTracker::inSemantic(const cv::Point &p) {
    if (!USE_SEGMENTATION) return false;
    if (semantic_img.empty()) return false;
    uchar pixel_value = semantic_img.at<uchar>(p);

    if ((int)pixel_value == 255)
        return true;
    else
        return false;
}

bool FeatureTracker::inBorder(const cv::Point2f &pt) const {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file) {
    for (const auto &i : calib_file) {
        cout << "reading parameter of camera: " << i.c_str() << endl;
        CameraModel::CameraPtr _camera =
            CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(i);
        camera.emplace_back(_camera);
    }
    if (STEREO) use_stereo_cam = true;
    if (DEPTH) use_rgbd_cam = true;
    cout << "USE STEREO: " << use_stereo_cam << " USE RGBD: " << use_rgbd_cam << endl;
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

FeatureFrame FeatureTracker::trackImage(double _cur_time, const cv::Mat &img_0, const cv::Mat &img_1) {
    // initial parameters
    cur_time = _cur_time;
    cur_img = img_0;
    row = cur_img.rows;
    col = cur_img.cols;
    right_img = img_1;
    cur_pts.clear();

    // if it is not first frame
    if (!prev_pts.empty()) {
        // opencv optical flow tracking parameters
        vector<uchar> status;
        vector<float> err;
        // if has predict in estimator before
        if (has_prediction) {
            cur_pts = predict_pts;
            // set pyramid level as 1
            cv::calcOpticalFlowPyrLK(
                prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

            // the number of tracking points
            int succ_num = 0;
            for (unsigned char statu : status) {
                if (statu)
                    succ_num++;
            }
            // if feature point num is too few, set pyramid level as 3 and re-optical flow
            if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status,
                                         err, cv::Size(21, 21), 3);
        }
        // if has no prediction
        else
            // set pyramid level as 3 and optical flow
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status,
                                     err, cv::Size(21, 21), 3);

        // reverse LK optical flow calculate once
        if (FLOW_BACK) {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(
                cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err,
                cv::Size(21, 21), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

            // both the forward and reverse optical flow are matched and the distance from the original point is not more than 0.5 pixels
            for (auto i = 0; i < status.size(); i++) {
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else
                    status[i] = 0;
            }
        }
        ROS_DEBUG("cur_pts size: %ld", cur_pts.size());
        // remove feature point in image border and semantic mask
        for (auto i = 0; i < cur_pts.size(); i++) {
            if (inSemantic(cur_pts[i])) {
                setFeatureStatus(ids[i], FeatureLevel::DYNAMIC);
            } else {
                if (track_cnt[ids[i]] > 3)
                    setFeatureStatus(ids[i], FeatureLevel::OPTIMIZE);
                else {
                    setFeatureStatus(ids[i], FeatureLevel::UN_INITIAL);
                }
            }
            if (status[i] && !inBorder(cur_pts[i])) {
                status[i] = 0;
            }
        }

        // remove feature point which is tracking lost
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("after reduce cur_pts size: %ld", cur_pts.size());
    }

    // add 1 to tracking times of the remaining feature points
    for (auto &n : track_cnt)
        n++;

    // calculate fundamental matrix and further eliminate outliers
    rejectWithF();

    //  draw a circle in feature point, and sort feature point vector based on track times
    setMask();

    // track up to MAX_CNT feature points.
    // If the current frame does not have so many feature points, the rest will be filled by corner points
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    ROS_DEBUG("n_max_cnt: %d", n_max_cnt);
    if (n_max_cnt > 0) {
        if (mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        // extract corner points
        cv::goodFeaturesToTrack(cur_img, n_pts,
                                MAX_CNT - static_cast<int>(cur_pts.size()), 0.01,
                                MIN_DIST, mask);
    }
    // if the number of feature point tracked by optical flow is meet the requirements, clear the n_pts vector
    else
        n_pts.clear();

    // fill in corner points
    for (auto &p : n_pts) {
        if (inSemantic(p))
            setFeatureStatus(n_id, FeatureLevel::DYNAMIC);
        else
            setFeatureStatus(n_id, FeatureLevel::UN_INITIAL);
        cur_pts.emplace_back(p);
        ids.emplace_back(n_id++);
        track_cnt.emplace_back(1);
    }

    // get normalized camera plane points calculated from pixels, with distortion correction
    cur_un_pts = undistortedPts(cur_pts, camera[0]);
    // calculate the moving speed of the current frame normalized camera plane feature points in the x and y directions
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // if camera is stereo
    if (!img_1.empty() && use_stereo_cam) {
        // almost the same as before
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if (!cur_pts.empty()) {
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(cur_img, right_img, cur_pts, cur_right_pts,
                                     status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            if (FLOW_BACK) {
                cv::calcOpticalFlowPyrLK(right_img, cur_img, cur_right_pts,
                                         reverseLeftPts, statusRightLeft, err,
                                         cv::Size(21, 21), 3);
                for (auto i = 0; i < status.size(); i++) {
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }
            // only delete the feature points lost on the right side
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);

            // calculate the moving speed of the feature points on the right image on the normalized camera plane
            cur_un_right_pts = undistortedPts(cur_right_pts, camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map,
                                             prev_un_right_pts_map);
        }
        // update state
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    // for visualization
    if (SHOW_TRACK)
        drawTrack(cur_img, right_img, ids, cur_pts, cur_right_pts, prev_left_pts_map);

    // update state
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    has_prediction = false;
    prev_left_pts_map.clear();
    for (auto i = 0; i < cur_pts.size(); i++)
        prev_left_pts_map[ids[i]] = cur_pts[i];

    // add current frame feature points
    FeatureFrame feature_frame;
    for (auto i = 0; i < ids.size(); i++) {
        // feature id
        int feature_id = ids[i];
        // normalized camera coordinate
        double x, y, z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        // pixel coordinate
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        // camera id
        int camera_id = 0;
        // normalized camera coordinate feature point speed
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        // set feature frame
        Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    // if stereo
    if (!img_1.empty() && use_stereo_cam) {
        for (size_t i = 0; i < ids_right.size(); i++) {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            // use camera id to distinguish
            feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }
    return feature_frame;
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts,
                                                   const CameraModel::CameraPtr &cam) {
    vector<cv::Point2f> un_pts;
    for (auto &pt : pts) {
        // feature point pixel coodinate
        Vector2d a(pt.x, pt.y);
        Vector3d b;
        // calculate normalized camera coordinate by pixel point with distortion correction
        cam->liftProjective(a, b);
        // normalized camera coordinate
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return un_pts;
}

vector<cv::Point2f>
FeatureTracker::ptsVelocity(vector<int> &ids_, vector<cv::Point2f> &pts,
                            map<int, cv::Point2f> &cur_id_pts,
                            map<int, cv::Point2f> &prev_id_pts) {
    vector<cv::Point2f> pts_velocity_;
    cur_id_pts.clear();
    for (auto i = 0; i < ids_.size(); i++) {
        cur_id_pts.insert(make_pair(ids_[i], pts[i]));
    }

    // calculate points velocity
    if (!prev_id_pts.empty()) {
        double dt = cur_time - prev_time;

        // traverse normalized camera coordinate feature point on the current frame
        for (auto i = 0; i < pts.size(); i++) {
            map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids_[i]);
            if (it != prev_id_pts.end()) {
                // calculate velocity in normalized camera coordinate
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity_.emplace_back(v_x, v_y);
            } else
                pts_velocity_.emplace_back(0, 0);
        }
    } else {
        for (auto i = 0; i < cur_pts.size(); i++) {
            pts_velocity_.emplace_back(0, 0);
        }
    }
    return pts_velocity_;
}

void FeatureTracker::rejectWithF() {
    if (cur_pts.size() >= 8) {
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            Vector3d tmp_p;
            camera[0]->liftProjective(Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            camera[0]->liftProjective(Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
    }
}

void FeatureTracker::setPrediction(map<int, Vector3d> &_predict_pts) {
    has_prediction = true;
    predict_pts.clear();
    map<int, Vector3d>::iterator predict_it;
    // predict point position
    for (auto i = 0; i < ids.size(); i++) {
        int id = ids[i];
        predict_it = _predict_pts.find(id);
        if (predict_it != _predict_pts.end()) {
            Vector2d tmp_uv;
            camera[0]->spaceToPlane(predict_it->second, tmp_uv);
            predict_pts.emplace_back(tmp_uv.x(), tmp_uv.y());
        } else
            predict_pts.emplace_back(prev_pts[i]);
    }
}

void FeatureTracker::removeOutliers(set<int> &remove_pts_ids) {
    set<int>::iterator set_it;
    vector<uchar> status;
    // find feature point with same id in remove set
    for (int id : ids) {
        set_it = remove_pts_ids.find(id);

        if (set_it != remove_pts_ids.end()) {
            status.emplace_back(0);
            pts_status.emplace(id, FeatureLevel::REMOVE);
        } else {
            status.emplace_back(1);
            pts_status.emplace(id, FeatureLevel::OPTIMIZE);
        }
    }
    // reduce
    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

void FeatureTracker::drawTrack(const cv::Mat &img_left, const cv::Mat &img_right,
                               vector<int> &_cur_left_ids,
                               vector<cv::Point2f> &_cur_left_pts,
                               vector<cv::Point2f> &_cur_right_pts,
                               map<int, cv::Point2f> &_prev_left_pts_map) {
    int cols = img_left.cols;
    // use the left image for monocular, put the left and right images together for binocular
    if (!img_right.empty() && use_stereo_cam)
        cv::hconcat(img_left, img_right, img_track);
    else
        img_track = img_left.clone();
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);

    // draw a circle for the feature points in the left picture, the number of red tracking is less, and the number of blue tracking is more
    for (auto j = 0; j < _cur_left_pts.size(); j++) {
        double len = min(1.0, 1.0 * track_cnt[j] / 20);
        int status = 0;
        if (pts_status.find(ids[j]) != pts_status.end()) {
            status = pts_status[ids[j]];
        }
        if (status == FeatureLevel::DYNAMIC) {
            cv::circle(img_track, _cur_left_pts[j], 2, cv::Scalar(0, 255, 255), 2);
            //cv::putText(img_track, to_string(ids[j]), _cur_left_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1.8);
        } else if (status == FeatureLevel::UN_INITIAL) {
            cv::circle(img_track, _cur_left_pts[j], 2, cv::Scalar(0, 0, 255), 2);
            //cv::putText(img_track, to_string(ids[j]), _cur_left_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1.8);
        } else {
            //cv::circle(img_track, _cur_left_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            cv::circle(img_track, _cur_left_pts[j], 2, cv::Scalar(255, 0, 0), 2);
            //cv::putText(img_track, to_string(ids[j]), _cur_left_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255 * (1 - len), 0, 255 * len), 1.8);
        }
    }
    // draw a green circle for the feature points in the right picture
    if (!img_right.empty() && use_stereo_cam) {
        for (auto rightPt : _cur_right_pts) {
            rightPt.x += static_cast<float>(cols);
            cv::circle(img_track, rightPt, 2, cv::Scalar(0, 255, 0), 2);
        }
    }

    //draw an arrow for feature points in the left image and aim to the position of feature points in the previous frame
    map<int, cv::Point2f>::iterator map_it;
    for (auto i = 0; i < _cur_left_ids.size(); i++) {
        int id = _cur_left_ids[i];
        map_it = _prev_left_pts_map.find(id);
        if (map_it != _prev_left_pts_map.end()) {
            cv::arrowedLine(img_track, _cur_left_pts[i], map_it->second,
                            cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

cv::Mat FeatureTracker::getTrackImage() {
    return img_track;
}

void FeatureTracker::setFeatureStatus(int feature_id, int status) {
    pts_status[feature_id] = status;
    if (status < 0) {
        removed_pts.insert(feature_id);
    }
}

map<int, int> FeatureTracker::getFeatureStatus() {
    return pts_status;
}

int FeaturePerId::endFrame() const {
    return start_frame + static_cast<int>(feature_per_frame.size()) - 1;
}

FeatureManager::FeatureManager() {
    for (auto i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::clearState() {
    feature.clear();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
    for (auto i = 0; i < NUM_OF_CAM; i++) {
        ric[i] = _ric[i];
    }
}

int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (auto &_it : feature) {
        auto &it = _it.second;
        it.used_num = static_cast<int>(it.feature_per_frame.size());
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
            cnt++;
        }
    }
    return cnt;
}

void FeatureManager::setDepthImage(const cv::Mat &depth) {
    depth_img = depth;
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, FeatureFrame &image, double td) {
    ROS_DEBUG("input feature: %zu", image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    new_feature_num = 0;
    long_track_num = 0;
    // traverse feature points
    for (auto iter = image.begin(), iter_next = image.begin(); iter != image.end(); iter = iter_next) {
        ++iter_next;

        // feature point correspond to left camera
        FeaturePerFrame feature_per_frame(iter->second[0].second, td);
        assert(iter->second[0].first == 0);
        // if vio system run in stereo mode
        if (iter->second.size() == 2 && STEREO) {
            // feature point correspond to right camera
            feature_per_frame.rightObservation(iter->second[1].second);
            assert(iter->second[1].first == 1);
        }
        // if vio system run in rgb-d mode
        if (DEPTH) {
            // get feature point depth from depth image
            auto pt_depth_mm = depth_img.at<unsigned short>((int)iter->second[0].second(4), (int)iter->second[0].second(3));
            double pt_depth_m = pt_depth_mm / 1000.0;
            if (0 < pt_depth_m && pt_depth_m < DEPTH_MIN_DIST) {
                image.erase(iter);
                continue;
            }
            // add depth information to feature per frame
            feature_per_frame.inputDepth(pt_depth_m);
        }

        // feature point id
        int feature_id = iter->first;

        // new feature
        if (feature.find(feature_id) == feature.end()) {
            // record feature point id and frame index
            feature.emplace(feature_id, FeaturePerId(feature_id, frame_count));
            // add feature point message to feature buffer
            feature[feature_id].feature_per_frame.push_back(feature_per_frame);
            // new feature point number on the current frame
            new_feature_num++;
        }
        // old feature
        else {
            feature[feature_id].feature_per_frame.push_back(feature_per_frame);
            // old feature point number on the current frame
            last_track_num++;
            if (feature[feature_id].feature_per_frame.size() >= 4)
                // number of the feature point which been tracked greater than 4 frame
                long_track_num++;
        }
    }

    // check if it is keyframe
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    // traverse feature points
    for (auto &_it : feature) {
        auto &it_per_id = _it.second;
        // if the feature points two frames ago have been tracked until the current frame is still
        if (it_per_id.start_frame <= frame_count - 2 && it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
            // calculate the distance of the current feature point on the normalized camera plane in the first two frames
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // check again if it is keyframe
    if (parallax_num == 0) {
        return true;
    } else {
        // check whether the disparity between the first two frames is large enough (10 pixels in the pixel coordinate system)
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
    vector<pair<Vector3d, Vector3d>> corres;
    // traverse feature points
    for (auto &it : feature) {
        auto &it_per_id = it.second;
        // the observation frame of the feature point covers l and r
        if (it_per_id.start_frame <= frame_count_l && it_per_id.endFrame() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it_per_id.start_frame;
            int idx_r = frame_count_r - it_per_id.start_frame;

            a = it_per_id.feature_per_frame[idx_l].point;
            b = it_per_id.feature_per_frame[idx_r].point;

            corres.emplace_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(map<int, double> deps) {
    for (auto &it : deps) {
        int _id = it.first;
        double depth = it.second;
        auto &it_per_id = feature[_id];

        it_per_id.estimated_depth = 1.0 / depth;

        if (it_per_id.estimated_depth < 0) {
            it_per_id.solve_flag = 2;
        } else {
            it_per_id.solve_flag = 1;
        }
    }
}

void FeatureManager::removeFailures() {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;
        auto &_it = it->second;
        if (_it.solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth() {
    for (auto &_it : feature) {
        auto &it_per_id = _it.second;
        it_per_id.estimated_depth = -1;
    }
}

map<int, double> FeatureManager::getDepthVector() {
    //This function gives actually points for solving; We only use oldest max_solve_cnt point, oldest pts has good track
    //As for some feature point not solve all the time; we do re triangulate on it
    map<int, double> dep_vec;
    auto pts_status = ft.getFeatureStatus();
    int total_feature_num = 0, dynamic_feature_num = 0;
    for (auto &_it : feature) {
        auto &it_per_id = _it.second;
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        if (dep_vec.size() < MAX_SOLVE_CNT && pts_status[it_per_id.feature_id] == FeatureLevel::OPTIMIZE
            && it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2) {
            dep_vec[it_per_id.feature_id] = 1. / it_per_id.estimated_depth;
        }

        if (it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2) {
            total_feature_num++;
            if (pts_status[it_per_id.feature_id] == FeatureLevel::DYNAMIC) dynamic_feature_num++;
        }
    }
    ROS_INFO("total feature num: %d", total_feature_num);
    ROS_INFO("dynamic feature num: %d", dynamic_feature_num);
    ROS_INFO("ratio: %lf", (double)dynamic_feature_num / total_feature_num);

    if (dep_vec.size() < MAX_SOLVE_CNT) {
        for (auto &_it : feature) {
            auto &it_per_id = _it.second;
            it_per_id.used_num = it_per_id.feature_per_frame.size();

            if (dep_vec.size() < MAX_SOLVE_CNT && pts_status[it_per_id.feature_id] == FeatureLevel::UN_INITIAL
                && it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2) {
                dep_vec[it_per_id.feature_id] = 1. / it_per_id.estimated_depth;
            }
        }
    }

    return dep_vec;
}

void FeatureManager::triangulatePoint(Matrix<double, 3, 4> &Pose0,
                                      Matrix<double, 3, 4> &Pose1,
                                      Vector2d &point0,
                                      Vector2d &point1,
                                      Vector3d &point_3d) {
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

bool FeatureManager::solvePoseByPnP(Matrix3d &R, Vector3d &P,
                                    vector<cv::Point2f> &pts2D,
                                    vector<cv::Point3f> &pts3D) {
    Matrix3d R_initial;
    Vector3d P_initial;

    // w<-cam ---> cam<-w
    // R_initial: Rcw
    // P_initial: Pcw
    R_initial = R.inverse();
    P_initial = -(R_initial * P);
    // feature point is too less
    if (int(pts2D.size()) < 4) {
        cout << "feature tracking not enough, please slowly move you device!" << endl;
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    // 3d-2d Pnp solve pose
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, true);

    if (!pnp_succ) {
        cout << "pnp failed !" << endl;
        return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);
    return true;
}

void FeatureManager::initFramePoseByPnP(int frame_cnt, Vector3d Ps_[],
                                        Matrix3d Rs_[], Vector3d tic_[],
                                        Matrix3d ric_[]) {
    if (frame_cnt > 0) {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        // traverse feature points in the current frame
        for (auto &it : feature) {
            auto &it_per_id = it.second;
            // check depth > 0
            if (it_per_id.estimated_depth > 0) {
                int index = frame_cnt - it_per_id.start_frame;
                // feature point is stable tracked
                if ((int)it_per_id.feature_per_frame.size() >= index + 1) {
                    // point in IMU coordinate in the first frame
                    Vector3d pts_in_body = ric_[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic_[0];
                    // convert to world coordinate from the first frame IMU coordinate
                    Vector3d pts_in_world = Rs_[it_per_id.start_frame] * pts_in_body + Ps_[it_per_id.start_frame];

                    cv::Point3f point3d(static_cast<float>(pts_in_world.x()),
                                        static_cast<float>(pts_in_world.y()),
                                        static_cast<float>(pts_in_world.z()));
                    cv::Point2f point2d(static_cast<float>(it_per_id.feature_per_frame[index].point.x()),
                                        static_cast<float>(it_per_id.feature_per_frame[index].point.y()));
                    // push back 3d point
                    pts3D.emplace_back(point3d);
                    // push back 2d point
                    pts2D.emplace_back(point2d);
                }
            }
        }
        Matrix3d R_camera;
        Vector3d P_camera;
        // trans to w_T_cam
        // pose in the previous frame Twc = Twi * Tic
        // R_camera: Rwc
        // P_camera: Pwc
        R_camera = Rs_[frame_cnt - 1] * ric_[0];
        P_camera = Rs_[frame_cnt - 1] * tic_[0] + Ps_[frame_cnt - 1];

        // 3d-2d solve pose by pnp
        if (solvePoseByPnP(R_camera, P_camera, pts2D, pts3D)) {
            // convert to IMU coordinate
            Rs_[frame_cnt] = R_camera * ric_[0].transpose();
            Ps_[frame_cnt] = -R_camera * ric_[0].transpose() * tic_[0] + P_camera;

            Quaterniond Q(Rs_[frame_cnt]);
        }
    }
}

void FeatureManager::triangulate(Vector3d Ps_[], Matrix3d Rs_[], Vector3d tic_[],
                                 Matrix3d ric_[]) {
    // traverse feature points in the current frame
    for (auto &it : feature) {
        auto &it_per_id = it.second;

        // if has get depth
        if (it_per_id.estimated_depth > 0)
            continue;
        // if it is rgb-d camera
        if (DEPTH) {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            int imu_i = it_per_id.start_frame;

            // tr: Pwc   Rr: Rwc
            Vector3d tr = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
            Matrix3d Rr = Rs_[imu_i] * ric_[0];

            vector<double> verified_depths;
            int no_depth_num = 0;

            vector<double> rough_depths;
            // perform deep cross-validation on the same id feature point,
            // and average the depth on the initial frame of the point whose reprojection error is less than the threshold as the estimated depth
            for (int k = 0; k < (int)it_per_id.feature_per_frame.size(); k++) {
                if (it_per_id.feature_per_frame[k].depth == 0) {
                    no_depth_num++;
                    continue;
                }
                Vector3d t0 = Ps_[imu_i + k] + Rs_[imu_i + k] * tic_[0];
                Matrix3d R0 = Rs_[imu_i + k] * ric_[0];
                //point0: Pc'
                Vector3d point0(it_per_id.feature_per_frame[k].point * it_per_id.feature_per_frame[k].depth);
                // transform to reference frame
                // t2r: tcc' R2r: Rcc'
                Vector3d t2r = Rr.transpose() * (t0 - tr);
                Matrix3d R2r = Rr.transpose() * R0;

                for (int j = 0; j < (int)it_per_id.feature_per_frame.size(); j++) {
                    if (k == j)
                        continue;
                    Vector3d t1 = Ps_[imu_i + j] + Rs_[imu_i + j] * tic_[0];
                    Matrix3d R1 = Rs_[imu_i + j] * ric_[0];
                    // t20: tc'c"  R20: Rc'c"
                    Vector3d t20 = R0.transpose() * (t1 - t0);
                    Matrix3d R20 = R0.transpose() * R1;
                    // point1_projected: Pc"
                    Vector3d point1_projected = R20.transpose() * point0 - R20.transpose() * t20;
                    Vector2d point1_2d(it_per_id.feature_per_frame[j].point.x(), it_per_id.feature_per_frame[j].point.y());

                    Vector2d residual = point1_2d - Vector2d(point1_projected.x() / point1_projected.z(), point1_projected.y() / point1_projected.z());
                    if (residual.norm() < 10.0 / 460) {
                        // point_r: Pc
                        Vector3d point_r = R2r * point0 + t2r;
                        if (it_per_id.feature_per_frame[k].depth > DEPTH_MAX_DIST) {
                            rough_depths.emplace_back(point_r.z());
                        } else {
                            verified_depths.emplace_back(point_r.z());
                        }
                    }
                }
            }
            if (verified_depths.empty()) {
                if (rough_depths.empty()) {
                    if (no_depth_num == it_per_id.feature_per_frame.size()) {
                        // the initial observation frame pose Tcw
                        int imu_i = it_per_id.start_frame;
                        Matrix<double, 3, 4> left_pose;
                        Vector3d t0 = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
                        Matrix3d R0 = Rs_[imu_i] * ric_[0];
                        left_pose.leftCols<3>() = R0.transpose();
                        left_pose.rightCols<1>() = -R0.transpose() * t0;

                        // the second observation frame pose Tcw
                        imu_i++;
                        Matrix<double, 3, 4> right_pose;
                        Vector3d t1 = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
                        Matrix3d R1 = Rs_[imu_i] * ric_[0];
                        right_pose.leftCols<3>() = R1.transpose();
                        right_pose.rightCols<1>() = -R1.transpose() * t1;

                        // take the normalized camera plane points corresponding to the two frames
                        Vector2d point0, point1;
                        Vector3d point3d;
                        point0 = it_per_id.feature_per_frame[0].point.head(2);
                        point1 = it_per_id.feature_per_frame[1].point.head(2);

                        // SVD calculate triangulate point
                        triangulatePoint(left_pose, right_pose, point0, point1, point3d);
                        // camera point
                        Vector3d local_point;
                        local_point = left_pose.leftCols<3>() * point3d + left_pose.rightCols<1>();
                        // set depth
                        double depth = local_point.z();
                        if (depth >= DEPTH_MIN_DIST) {
                            it_per_id.estimated_depth = depth;
                            it_per_id.estimate_flag = 2;
                        } else {
                            it_per_id.estimated_depth = INIT_DEPTH;
                            it_per_id.estimate_flag = 0;
                        }

                    } else {
                        continue;
                    }
                } else {
                    double depth_sum = accumulate(begin(rough_depths), end(rough_depths), 0.0);
                    double depth_ave = depth_sum / rough_depths.size();
                    it_per_id.estimated_depth = depth_ave;
                    it_per_id.estimate_flag = 0;
                }
            } else {
                double depth_sum = accumulate(begin(verified_depths), end(verified_depths), 0.0);
                double depth_ave = depth_sum / verified_depths.size();
                it_per_id.estimated_depth = depth_ave;
                it_per_id.estimate_flag = 1;
            }

            if (it_per_id.estimated_depth < DEPTH_MIN_DIST) {
                it_per_id.estimated_depth = INIT_DEPTH;
                it_per_id.estimate_flag = 0;
            }
        }
        // if it is stereo camera or monocular camera
        else {
            // stereo triangulate in one frame
            if (STEREO && it_per_id.feature_per_frame[0].is_stereo) {
                // the initial observation frame pose Tcw
                int imu_i = it_per_id.start_frame;
                Matrix<double, 3, 4> left_pose;
                Vector3d t0 = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
                Matrix3d R0 = Rs_[imu_i] * ric_[0];
                left_pose.leftCols<3>() = R0.transpose();
                left_pose.rightCols<1>() = -R0.transpose() * t0;

                // the initial observation frame pose，right camera Tcw
                Matrix<double, 3, 4> right_pose;
                Vector3d t1 = Ps_[imu_i] + Rs_[imu_i] * tic_[1];
                Matrix3d R1 = Rs_[imu_i] * ric_[1];
                right_pose.leftCols<3>() = R1.transpose();
                right_pose.rightCols<1>() = -R1.transpose() * t1;

                // take the normalized camera plane points corresponding to the left and right camera
                Vector2d point0, point1;
                Vector3d point3d;
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[0].point_right.head(2);

                // SVD calculate triangulate point
                triangulatePoint(left_pose, right_pose, point0, point1, point3d);
                // camera point
                Vector3d local_point;
                local_point = left_pose.leftCols<3>() * point3d + left_pose.rightCols<1>();
                // set depth
                double depth = local_point.z();
                if (depth > 0) {
                    it_per_id.estimated_depth = depth;
                    it_per_id.estimate_flag = 2;
                } else {
                    it_per_id.estimated_depth = INIT_DEPTH;
                    it_per_id.estimate_flag = 0;
                }
                continue;
            }
            // for monocular triangulation, the observation frame must be at least 2 frames
            else if (it_per_id.feature_per_frame.size() > 1) {
                // the initial observation frame pose Tcw
                int imu_i = it_per_id.start_frame;
                Matrix<double, 3, 4> left_pose;
                Vector3d t0 = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
                Matrix3d R0 = Rs_[imu_i] * ric_[0];
                left_pose.leftCols<3>() = R0.transpose();
                left_pose.rightCols<1>() = -R0.transpose() * t0;

                // the second observation frame pose Tcw
                imu_i++;
                Matrix<double, 3, 4> right_pose;
                Vector3d t1 = Ps_[imu_i] + Rs_[imu_i] * tic_[0];
                Matrix3d R1 = Rs_[imu_i] * ric_[0];
                right_pose.leftCols<3>() = R1.transpose();
                right_pose.rightCols<1>() = -R1.transpose() * t1;

                // take the normalized camera plane points corresponding to the two frames
                Vector2d point0, point1;
                Vector3d point3d;
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[1].point.head(2);

                // SVD calculate triangulate point
                triangulatePoint(left_pose, right_pose, point0, point1, point3d);
                // camera point
                Vector3d local_point;
                local_point = left_pose.leftCols<3>() * point3d + left_pose.rightCols<1>();
                // set depth
                double depth = local_point.z();
                if (depth > 0) {
                    it_per_id.estimated_depth = depth;
                    it_per_id.estimate_flag = 2;
                } else {
                    it_per_id.estimated_depth = INIT_DEPTH;
                    it_per_id.estimate_flag = 0;
                }
                continue;
            }
        }
    }
}

void FeatureManager::removeOutlier(set<int> &outlier_index) {
    set<int>::iterator set_it;
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        int index = it->first;
        set_it = outlier_index.find(index);
        if (set_it != outlier_index.end()) {
            ft.setFeatureStatus(it->second.feature_id, FeatureLevel::REMOVE);
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(const Matrix3d &marg_R, const Vector3d &marg_P,
                                          const Matrix3d &new_R, const Vector3d &new_P) {
    // traverse feature points
    for (auto _it = feature.begin(), it_next = feature.begin(); _it != feature.end(); _it = it_next) {
        it_next++;

        auto &it = _it->second;
        if (it.start_frame != 0)
            it.start_frame--;
        else {
            // the normalized camera point of the first observed frame
            Vector3d uv_i = it.feature_per_frame[0].point;
            // delete the observation frame that is currently marg off
            it.feature_per_frame.erase(it.feature_per_frame.begin());
            // if there are less than 2 observation frames, delete the point
            if (it.feature_per_frame.size() < 2) {
                feature.erase(_it);
                continue;
            } else {
                // estimated_depth is the depth value under the observation frame of the first frame, and now it is updated to the next frame
                Vector3d pts_i = uv_i * it.estimated_depth;
                Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it.estimated_depth = dep_j;
                else
                    it.estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::removeBack() {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->second.start_frame != 0)
            it->second.start_frame--;
        else {
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin());
            if (it->second.feature_per_frame.empty()) {
                feature.erase(it);
                ft.setFeatureStatus(it->second.feature_id, FeatureLevel::REMOVE);
            }
        }
    }
}

void FeatureManager::removeFront(int frame_count) {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->second.start_frame == frame_count) {
            it->second.start_frame--;
        } else {
            // delete the frame dropped by marg
            int j = WINDOW_SIZE - 1 - it->second.start_frame;
            if (it->second.endFrame() < frame_count - 1)
                continue;
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin() + j);
            if (it->second.feature_per_frame.empty()) {
                feature.erase(it);
                ft.setFeatureStatus(it->second.feature_id, FeatureLevel::REMOVE);
            }
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
    // check the second last frame is keyframe or not
    // parallax between second last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // computes the distance between two points on the normalized camera plane
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

} // namespace FLOW_VINS