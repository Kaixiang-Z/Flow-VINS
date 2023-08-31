/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: pose graph
 */

#include "../include/LoopFusion.h"

namespace FLOW_VINS {

    static CameraModel::CameraPtr m_camera;

/**
 * @brief: delete variables in vector which value is equal to zero
 */
    template<typename Derived>
    static void reduceVector(vector<Derived> &v, vector<uchar> status) {
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

    BriefExtractor::BriefExtractor(const string &pattern_file) {
        // loads the pattern
        cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
        if (!fs.isOpened()) throw string("Could not open file ") + pattern_file;

        vector<int> x1, y1, x2, y2;
        fs["x1"] >> x1;
        fs["x2"] >> x2;
        fs["y1"] >> y1;
        fs["y2"] >> y2;

        m_brief.importPairs(x1, y1, x2, y2);
    }

    void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys,
                                    vector<BRIEF::bitset> &descriptors) const {
        m_brief.compute(im, keys, descriptors);
    }

    KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                       vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv,
                       vector<cv::Point2f> &_point_2d_norm,
                       vector<double> &_point_id, int _sequence) {
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
        for (const auto &i: point_2d_uv) {
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
        for (auto &keypoint: keypoints) {
            Vector3d tmp_p;
            m_camera->liftProjective(Vector2d(keypoint.pt.x, keypoint.pt.y), tmp_p);
            cv::KeyPoint tmp_norm;
            tmp_norm.pt = cv::Point2f(static_cast<float>(tmp_p.x() / tmp_p.z()),
                                      static_cast<float>(tmp_p.y() / tmp_p.z()));
            keypoints_norm.push_back(tmp_norm);
        }
    }

    int KeyFrame::hammingDistance(const BRIEF::bitset &a, const BRIEF::bitset &b) {
        // the Hamming distance is the number of different position values ​​after converting to binary
        BRIEF::bitset xor_of_bitset = a ^ b;
        int dis = static_cast<int>(xor_of_bitset.count());
        return dis;
    }

    bool KeyFrame::searchInAera(const BRIEF::bitset &window_descriptor,
                                const vector<BRIEF::bitset> &descriptors_old,
                                const vector<cv::KeyPoint> &keypoints_old,
                                const vector<cv::KeyPoint> &keypoints_old_norm,
                                cv::Point2f &best_match,
                                cv::Point2f &best_match_norm) {
        int bestDist = 128;
        int bestIndex = -1;
        // find shortest distance between current descriptor and old descriptors
        for (int i = 0; i < (int) descriptors_old.size(); i++) {
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

    void KeyFrame::searchByBRIEFDes(vector<cv::Point2f> &matched_2d_old,
                                    vector<cv::Point2f> &matched_2d_old_norm,
                                    vector<uchar> &status,
                                    const vector<BRIEF::bitset> &descriptors_old,
                                    const vector<cv::KeyPoint> &keypoints_old,
                                    const vector<cv::KeyPoint> &keypoints_old_norm) {
        // traverse brief descriptors in current frame
        for (const auto &window_brief_descriptor: window_brief_descriptors) {
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

    bool KeyFrame::findConnection(KeyFrame *old_kf, queue<RelocationFrame> &relo_frame_buf) {
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

        Vector3d PnP_T_old;
        Matrix3d PnP_R_old;
        Vector3d relative_t;
        Quaterniond relative_q;
        double relative_yaw;
        // if matched point nums > MIN_LOOP_NUM, use pnp ransac delete mis-matched points
        if ((int) matched_2d_cur.size() > MIN_LOOP_NUM) {
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
        if ((int) matched_2d_cur.size() > MIN_LOOP_NUM) {
            relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
            relative_q = PnP_R_old.transpose() * origin_vio_R;
            relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());

            // detect loop
            if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {
                has_loop = true;
                loop_index = old_kf->index;
                loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                        relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                        relative_yaw;

                // publish match points
                RelocationFrame relo_frame;
                relo_frame.time = time_stamp;
                for (int i = 0; i < (int) matched_2d_old_norm.size(); i++) {
                    relo_frame.relo_uv_id.emplace_back(
                            Vector3d(matched_2d_old_norm[i].x, matched_2d_old_norm[i].y, matched_id[i]));
                }
                relo_frame.relo_T = old_kf->T_w_i;
                relo_frame.relo_R = old_kf->R_w_i;
                relo_frame.index = index;
                mutex_relocation.lock();
                relo_frame_buf.push(relo_frame);
                mutex_relocation.unlock();

                return true;
            }
        }

        return false;
    }

    void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                             const vector<cv::Point3f> &matched_3d,
                             vector<uchar> &status,
                             Vector3d &PnP_T_old, Matrix3d &PnP_R_old) {
        cv::Mat r, rvec, t, D, tmp_r;
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
        Matrix3d R_inital;
        Vector3d P_inital;
        Matrix3d R_w_c = origin_vio_R * LOOP_QIC;
        Vector3d T_w_c = origin_vio_T + origin_vio_R * LOOP_TIC;

        R_inital = R_w_c.inverse();
        P_inital = -(R_inital * T_w_c);

        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        cv::Mat inliers;

        cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

        // remove outlier
        for (int i = 0; i < (int) matched_2d_old_norm.size(); i++)
            status.push_back(0);

        for (int i = 0; i < inliers.rows; i++) {
            int n = inliers.at<int>(i);
            status[n] = 1;
        }

        cv::Rodrigues(rvec, r);
        Matrix3d R_pnp, R_w_c_old;
        cv::cv2eigen(r, R_pnp);
        R_w_c_old = R_pnp.transpose();
        Vector3d T_pnp, T_w_c_old;
        cv::cv2eigen(t, T_pnp);
        T_w_c_old = R_w_c_old * (-T_pnp);

        PnP_R_old = R_w_c_old * LOOP_QIC.transpose();
        PnP_T_old = T_w_c_old - PnP_R_old * LOOP_TIC;
    }

    void KeyFrame::getVioPose(Vector3d &_T_w_i, Matrix3d &_R_w_i) const {
        _T_w_i = vio_T_w_i;
        _R_w_i = vio_R_w_i;
    }

    void KeyFrame::getPose(Vector3d &_T_w_i, Matrix3d &_R_w_i) const {
        _T_w_i = T_w_i;
        _R_w_i = R_w_i;
    }

    void KeyFrame::updateVioPose(const Vector3d &_T_w_i, const Matrix3d &_R_w_i) {
        vio_T_w_i = _T_w_i;
        vio_R_w_i = _R_w_i;
        T_w_i = vio_T_w_i;
        R_w_i = vio_R_w_i;
    }

    void KeyFrame::updatePose(const Vector3d &_T_w_i, const Matrix3d &_R_w_i) {
        T_w_i = _T_w_i;
        R_w_i = _R_w_i;
    }

    Vector3d KeyFrame::getLoopRelativeT() {
        return Vector3d{loop_info(0), loop_info(1), loop_info(2)};
    }

    Quaterniond KeyFrame::getLoopRelativeQ() {
        return Quaterniond{loop_info(3), loop_info(4), loop_info(5), loop_info(6)};
    }

    double KeyFrame::getLoopRelativeYaw() {
        return loop_info(7);
    }

    PoseGraph::PoseGraph() {
    }

    PoseGraph::~PoseGraph() {
        thread_optimize.detach();
    }

    void PoseGraph::setParameter() {
        // set parameters
        t_drift = Vector3d::Zero();
        r_drift = Matrix3d::Identity();
        yaw_drift = 0;
        w_t_vio = Vector3d::Zero();
        w_r_vio = Matrix3d::Identity();
        earliest_loop_index = -1;
        global_index = 0;
        sequence_cnt = 0;
        sequence_loop.push_back(false);
        use_imu = false;
        m_camera = CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[0]);
        // load vocabulary
        loadVocabulary(VOCABULARY_PATH);
        // confirm IMU enable, start optimize thread
        setIMUFlag(USE_IMU);
    }

    void PoseGraph::loadVocabulary(const string &voc_path) {
        voc = new BriefVocabulary(voc_path);
        db.setVocabulary(*voc, false, 0);
    }

    void PoseGraph::setIMUFlag(bool _use_imu) {
        use_imu = _use_imu;
        if (use_imu) {
            cout << "VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization" << endl;
            thread_optimize = thread(&PoseGraph::optimize4DoF, this);
        } else {
            cout << "VO input, perfrom 6 DoF pose graph optimization" << endl;
            thread_optimize = thread(&PoseGraph::optimize6DoF, this);
        }
    }

    void PoseGraph::addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop) {
        // shift to base frame
        Vector3d vio_P_cur;
        Matrix3d vio_R_cur;
        if (sequence_cnt != cur_kf->sequence) {
            sequence_cnt++;
            sequence_loop.push_back(false);
            // if is uncontinued, reset state
            w_t_vio = Vector3d::Zero();
            w_r_vio = Matrix3d::Identity();
            mutex_drift.lock();
            t_drift = Vector3d::Zero();
            r_drift = Matrix3d::Identity();
            mutex_drift.unlock();
        }
        // update current keyframe vio R & T
        cur_kf->getVioPose(vio_P_cur, vio_R_cur);
        // update the pose after the loop correction to eliminate the accumulated error
        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
        vio_R_cur = w_r_vio * vio_R_cur;
        cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
        // update index
        cur_kf->index = global_index;
        global_index++;
        int loop_index = -1;
        // if enable loop correctiom, find the earliest keyframe index with high score
        if (flag_detect_loop) {
            loop_index = detectLoop(cur_kf, cur_kf->index);
        } else {
            addKeyFrameIntoVoc(cur_kf);
        }
        // detect loop closure
        if (loop_index != -1) {
            // get old keyframe with loop index find before
            KeyFrame *old_kf = getKeyFrame(loop_index);

            // descriptor matching between the current frame and the loopback candidate frame
            if (cur_kf->findConnection(old_kf, relo_frame_buf)) {
                if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                    earliest_loop_index = loop_index;

                // calculate the relative pose of the current frame and the loopback frame, and correct the pose of the current frame
                Vector3d w_P_old, w_P_cur, vio_P_cur;
                Matrix3d w_R_old, w_R_cur, vio_R_cur;
                old_kf->getVioPose(w_P_old, w_R_old);
                cur_kf->getVioPose(vio_P_cur, vio_R_cur);

                Vector3d relative_t;
                Quaterniond relative_q;
                relative_t = cur_kf->getLoopRelativeT();
                relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
                w_P_cur = w_R_old * relative_t + w_P_old;
                w_R_cur = w_R_old * relative_q;

                // the shift between the pose obtained by loopback and the VIO pose
                double shift_yaw;
                Matrix3d shift_r;
                Vector3d shift_t;
                if (use_imu) {
                    shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
                    shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
                } else
                    shift_r = w_R_cur * vio_R_cur.transpose();
                shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;

                // shift vio pose of whole sequence to the world frame
                if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
                    w_r_vio = shift_r;
                    w_t_vio = shift_t;
                    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                    vio_R_cur = w_r_vio * vio_R_cur;
                    cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
                    auto it = keyframelist.begin();
                    for (; it != keyframelist.end(); it++) {
                        if ((*it)->sequence == cur_kf->sequence) {
                            Vector3d vio_P_cur;
                            Matrix3d vio_R_cur;
                            (*it)->getVioPose(vio_P_cur, vio_R_cur);
                            vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                            vio_R_cur = w_r_vio * vio_R_cur;
                            (*it)->updateVioPose(vio_P_cur, vio_R_cur);
                        }
                    }
                    sequence_loop[cur_kf->sequence] = true;
                }
                // put the current frame in the optimization queue
                mutex_optimize.lock();
                optimize_buf.push(cur_kf->index);
                mutex_optimize.unlock();
            }
        }
        mutex_keyframe.lock();
        // get the pose R & T of the current frame of the VIO, and calculate the actual pose according to the offset, then update the pose
        Vector3d P;
        Matrix3d R;
        cur_kf->getVioPose(P, R);
        P = r_drift * P + t_drift;
        R = r_drift * R;
        cur_kf->updatePose(P, R);

        // publisher state update
        Quaterniond Q{R};
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = P.x();
        pose_stamped.pose.position.y = P.y();
        pose_stamped.pose.position.z = P.z();
        pose_stamped.pose.orientation.x = Q.x();
        pose_stamped.pose.orientation.y = Q.y();
        pose_stamped.pose.orientation.z = Q.z();
        pose_stamped.pose.orientation.w = Q.w();
        path[sequence_cnt].poses.push_back(pose_stamped);
        path[sequence_cnt].header = pose_stamped.header;

        // save loop result file into VINS_RESULT_PATH
        ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
        double time_stamp = cur_kf->time_stamp;
        loop_path_file.setf(ios::fixed, ios::floatfield);
        loop_path_file << time_stamp << " ";
        loop_path_file.precision(5);
        loop_path_file << P.x() << " "
                       << P.y() << " "
                       << P.z() << " "
                       << Q.x() << " "
                       << Q.y() << " "
                       << Q.z() << " "
                       << Q.w() << endl;
        loop_path_file.close();

        keyframelist.push_back(cur_kf);

        // publish topics
        pubPoseGraph(*this);
        mutex_keyframe.unlock();
    }

    KeyFrame *PoseGraph::getKeyFrame(int index) {
        auto it = keyframelist.begin();
        // traverse keyframe list
        for (; it != keyframelist.end(); it++) {
            if ((*it)->index == index)
                break;
        }
        if (it != keyframelist.end())
            return *it;
        else
            return nullptr;
    }

    int PoseGraph::detectLoop(KeyFrame *keyframe, int frame_index) {
        // first query; then add this frame into database!
        QueryResults ret;
        // query the dictionary database to get the similarity score ret with each frame, first 50 frames is not in database
        db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);
        // add keyframe brief descriptors into database
        db.add(keyframe->brief_descriptors);

        // ret[0] is the nearest neighbour's score. threshold change with neighour score
        bool find_loop = false;

        // ensure good similarity scores to neighboring frames
        if (!ret.empty() && ret[0].Score > 0.05)
            for (unsigned int i = 1; i < ret.size(); i++) {
                if (ret[i].Score > 0.015) {
                    find_loop = true;
                    int tmp_index = static_cast<int>(ret[i].Id);
                }
            }
        // find the earliest keyframe index with a score greater than 0.015
        if (find_loop && frame_index > 50) {
            int min_index = -1;
            for (unsigned int i = 0; i < ret.size(); i++) {
                if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                    min_index = static_cast<int>(ret[i].Id);
            }
            return min_index;
        } else
            return -1;
    }

    void PoseGraph::addKeyFrameIntoVoc(KeyFrame *keyframe) {
        db.add(keyframe->brief_descriptors);
    }

    void PoseGraph::optimize4DoF() {
        // thread start
        while (true) {
            // 1. get keyframe index from optimize queue
            int cur_index = -1;
            int first_looped_index = -1;
            mutex_optimize.lock();
            while (!optimize_buf.empty()) {
                cur_index = optimize_buf.front();
                first_looped_index = earliest_loop_index;
                optimize_buf.pop();
            }
            mutex_optimize.unlock();
            // 2. find cur index, start optimize
            if (cur_index != -1) {
                ROS_DEBUG("optimize pose graph");
                mutex_keyframe.lock();
                // get cur keyframe
                KeyFrame *cur_kf = getKeyFrame(cur_index);

                int max_length = cur_index + 1;

                // ceres optimize parameters Rwb, Twb
                double t_array[max_length][3];
                Quaterniond q_array[max_length];
                double euler_array[max_length][3];
                double sequence_array[max_length];

                // ceres problem config
                ceres::Problem problem;
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.max_num_iterations = 5;
                ceres::Solver::Summary summary;
                ceres::LossFunction *loss_function;
                loss_function = new ceres::HuberLoss(0.1);

                ceres::Manifold *angle_local_parameterization = AngleLocalParameterization::Create();

                list<KeyFrame *>::iterator it;
                // traverse keyframe list
                int i = 0;
                for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
                    if ((*it)->index < first_looped_index)
                        continue;
                    (*it)->local_index = i;
                    Quaterniond tmp_q;
                    Matrix3d tmp_r;
                    Vector3d tmp_t;
                    (*it)->getVioPose(tmp_t, tmp_r);
                    tmp_q = tmp_r;
                    t_array[i][0] = tmp_t(0);
                    t_array[i][1] = tmp_t(1);
                    t_array[i][2] = tmp_t(2);
                    q_array[i] = tmp_q;

                    Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                    euler_array[i][0] = euler_angle.x();
                    euler_array[i][1] = euler_angle.y();
                    euler_array[i][2] = euler_angle.z();

                    sequence_array[i] = (*it)->sequence;
                    // add parameter block
                    problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
                    problem.AddParameterBlock(t_array[i], 3);
                    // set first loop frame fixed
                    if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
                        problem.SetParameterBlockConstant(euler_array[i]);
                        problem.SetParameterBlockConstant(t_array[i]);
                    }

                    //add residual block
                    for (int j = 1; j < 5; j++) {
                        if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
                            Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
                            Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
                                                t_array[i][2] - t_array[i - j][2]);
                            relative_t = q_array[i - j].inverse() * relative_t;
                            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                            ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(),
                                                                                      relative_t.y(),
                                                                                      relative_t.z(),
                                                                                      relative_yaw,
                                                                                      euler_conncected.y(),
                                                                                      euler_conncected.z());
                            problem.AddResidualBlock(cost_function, nullptr, euler_array[i - j],
                                                     t_array[i - j],
                                                     euler_array[i],
                                                     t_array[i]);
                        }
                    }

                    //add loop edge
                    if ((*it)->has_loop) {
                        assert((*it)->loop_index >= first_looped_index);
                        int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                        Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                        Vector3d relative_t;
                        relative_t = (*it)->getLoopRelativeT();
                        double relative_yaw = (*it)->getLoopRelativeYaw();
                        ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(),
                                                                                  relative_t.y(),
                                                                                  relative_t.z(),
                                                                                  relative_yaw,
                                                                                  euler_conncected.y(),
                                                                                  euler_conncected.z());
                        problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                                 t_array[connected_index],
                                                 euler_array[i],
                                                 t_array[i]);
                    }

                    if ((*it)->index == cur_index)
                        break;
                    i++;
                }
                mutex_keyframe.unlock();

                ceres::Solve(options, &problem, &summary);

                mutex_keyframe.lock();
                // convert array to Eigen, update state
                i = 0;
                for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
                    if ((*it)->index < first_looped_index)
                        continue;
                    Quaterniond tmp_q;
                    tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                    Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                    Matrix3d tmp_r = tmp_q.toRotationMatrix();
                    (*it)->updatePose(tmp_t, tmp_r);

                    if ((*it)->index == cur_index)
                        break;
                    i++;
                }

                Vector3d cur_t, vio_t;
                Matrix3d cur_r, vio_r;
                cur_kf->getPose(cur_t, cur_r);
                cur_kf->getVioPose(vio_t, vio_r);

                // update drift
                mutex_drift.lock();
                yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
                r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
                t_drift = cur_t - r_drift * vio_t;
                mutex_drift.unlock();

                it++;
                for (; it != keyframelist.end(); it++) {
                    Vector3d P;
                    Matrix3d R;
                    (*it)->getVioPose(P, R);
                    P = r_drift * P + t_drift;
                    R = r_drift * R;
                    (*it)->updatePose(P, R);
                }
                mutex_keyframe.unlock();
                updatePath();
            }

            // sleep 2000ms
            chrono::milliseconds dura(2000);
            this_thread::sleep_for(dura);
        }
    }

    void PoseGraph::optimize6DoF() {
        // thread start
        while (true) {
            // 1. get keyframe index from optimize queue
            int cur_index = -1;
            int first_looped_index = -1;
            mutex_optimize.lock();
            while (!optimize_buf.empty()) {
                cur_index = optimize_buf.front();
                first_looped_index = earliest_loop_index;
                optimize_buf.pop();
            }
            mutex_optimize.unlock();
            // 2. find cur index, start optimize
            if (cur_index != -1) {
                ROS_DEBUG("optimize pose graph");
                mutex_keyframe.lock();
                // get cur keyframe
                KeyFrame *cur_kf = getKeyFrame(cur_index);

                int max_length = cur_index + 1;

                // ceres optimize parameters Rwb, Twb
                double t_array[max_length][3];
                double q_array[max_length][4];
                double sequence_array[max_length];

                // ceres problem config
                ceres::Problem problem;
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.max_num_iterations = 5;
                ceres::Solver::Summary summary;
                ceres::LossFunction *loss_function;
                loss_function = new ceres::HuberLoss(0.1);
                ceres::Manifold *local_parameterization = new ceres::QuaternionManifold();

                list<KeyFrame *>::iterator it;
                // traverse keyframe list
                int i = 0;
                for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
                    if ((*it)->index < first_looped_index)
                        continue;
                    (*it)->local_index = i;
                    Quaterniond tmp_q;
                    Matrix3d tmp_r;
                    Vector3d tmp_t;
                    (*it)->getVioPose(tmp_t, tmp_r);
                    tmp_q = tmp_r;
                    t_array[i][0] = tmp_t(0);
                    t_array[i][1] = tmp_t(1);
                    t_array[i][2] = tmp_t(2);
                    q_array[i][0] = tmp_q.w();
                    q_array[i][1] = tmp_q.x();
                    q_array[i][2] = tmp_q.y();
                    q_array[i][3] = tmp_q.z();

                    sequence_array[i] = (*it)->sequence;

                    // add parameter block
                    problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                    problem.AddParameterBlock(t_array[i], 3);
                    // set first loop frame fixed
                    if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
                        problem.SetParameterBlockConstant(q_array[i]);
                        problem.SetParameterBlockConstant(t_array[i]);
                    }

                    //add residual block
                    for (int j = 1; j < 5; j++) {
                        if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
                            Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
                                                t_array[i][2] - t_array[i - j][2]);
                            Quaterniond q_i_j = Quaterniond(q_array[i - j][0], q_array[i - j][1], q_array[i - j][2],
                                                            q_array[i - j][3]);
                            Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                            relative_t = q_i_j.inverse() * relative_t;
                            Quaterniond relative_q = q_i_j.inverse() * q_i;
                            ceres::CostFunction *vo_function = SixDOFError::Create(relative_t.x(), relative_t.y(),
                                                                                   relative_t.z(),
                                                                                   relative_q.w(), relative_q.x(),
                                                                                   relative_q.y(), relative_q.z(),
                                                                                   0.1, 0.01);
                            problem.AddResidualBlock(vo_function, nullptr, q_array[i - j], t_array[i - j], q_array[i],
                                                     t_array[i]);
                        }
                    }

                    //add loop edge

                    if ((*it)->has_loop) {
                        assert((*it)->loop_index >= first_looped_index);
                        int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                        Vector3d relative_t;
                        relative_t = (*it)->getLoopRelativeT();
                        Quaterniond relative_q;
                        relative_q = (*it)->getLoopRelativeQ();
                        ceres::CostFunction *loop_function = SixDOFError::Create(relative_t.x(), relative_t.y(),
                                                                                 relative_t.z(),
                                                                                 relative_q.w(), relative_q.x(),
                                                                                 relative_q.y(), relative_q.z(),
                                                                                 0.1, 0.01);
                        problem.AddResidualBlock(loop_function, loss_function, q_array[connected_index],
                                                 t_array[connected_index], q_array[i], t_array[i]);
                    }

                    if ((*it)->index == cur_index)
                        break;
                    i++;
                }
                mutex_keyframe.unlock();

                ceres::Solve(options, &problem, &summary);

                mutex_keyframe.lock();
                // convert array to Eigen, update state
                i = 0;
                for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
                    if ((*it)->index < first_looped_index)
                        continue;
                    Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                    Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                    Matrix3d tmp_r = tmp_q.toRotationMatrix();
                    (*it)->updatePose(tmp_t, tmp_r);

                    if ((*it)->index == cur_index)
                        break;
                    i++;
                }

                Vector3d cur_t, vio_t;
                Matrix3d cur_r, vio_r;
                cur_kf->getPose(cur_t, cur_r);
                cur_kf->getVioPose(vio_t, vio_r);

                // update drift
                mutex_drift.lock();
                r_drift = cur_r * vio_r.transpose();
                t_drift = cur_t - r_drift * vio_t;
                mutex_drift.unlock();

                it++;
                for (; it != keyframelist.end(); it++) {
                    Vector3d P;
                    Matrix3d R;
                    (*it)->getVioPose(P, R);
                    P = r_drift * P + t_drift;
                    R = r_drift * R;
                    (*it)->updatePose(P, R);
                }
                mutex_keyframe.unlock();
                updatePath();
            }
            // sleep 2000ms
            chrono::milliseconds dura(2000);
            this_thread::sleep_for(dura);
        }
    }

    void PoseGraph::updatePath() {
        mutex_keyframe.lock();
        list<KeyFrame *>::iterator it;
        for (int i = 1; i <= sequence_cnt; i++) {
            path[i].poses.clear();
        }
        base_path.poses.clear();

        // update all keyframe state
        for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
            Vector3d P;
            Matrix3d R;
            (*it)->getPose(P, R);
            Quaterniond Q{R};

            // publisher state update
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose.position.x = P.x();
            pose_stamped.pose.position.y = P.y();
            pose_stamped.pose.position.z = P.z();
            pose_stamped.pose.orientation.x = Q.x();
            pose_stamped.pose.orientation.y = Q.y();
            pose_stamped.pose.orientation.z = Q.z();
            pose_stamped.pose.orientation.w = Q.w();
            if ((*it)->sequence == 0) {
                base_path.poses.push_back(pose_stamped);
                base_path.header = pose_stamped.header;
            } else {
                path[(*it)->sequence].poses.push_back(pose_stamped);
                path[(*it)->sequence].header = pose_stamped.header;
            }
        }
        // publish topics
        pubPoseGraph(*this);
        mutex_keyframe.unlock();
    }

} // namespace FLOW_VINS