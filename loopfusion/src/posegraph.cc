/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: pose graph
 */

#include "../include/posegraph.h"

namespace FLOW_VINS {
PoseGraph::PoseGraph() {}

PoseGraph::~PoseGraph() { optimization_thread.detach(); }

void PoseGraph::setParameter() {
	// set parameters
	t_drift = Eigen::Vector3d::Zero();
	r_drift = Eigen::Matrix3d::Identity();
	yaw_drift = 0;
	w_t_vio = Eigen::Vector3d::Zero();
	w_r_vio = Eigen::Matrix3d::Identity();
	earliest_loop_index = -1;
	global_index = 0;
	sequence_cnt = 0;
	sequence_loop.push_back(false);
	use_imu = false;
	m_camera = CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(CAMERA_PATH);
	// load vocabulary
	loadVocabulary(VOCABULARY_PATH);
	// confirm IMU enable, start optimize thread
	setIMUFlag(USE_IMU);
}

void PoseGraph::loadVocabulary(const std::string& voc_path) {
	voc = new BriefVocabulary(voc_path);
	db.setVocabulary(*voc, false, 0);
}

void PoseGraph::setIMUFlag(bool _use_imu) {
	use_imu = _use_imu;
	if (use_imu) {
		LOGGER_INFO("VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization");
		optimization_thread = std::thread(&PoseGraph::optimize4DoF, this);
	} else {
		LOGGER_INFO("VO input, perfrom 6 DoF pose graph optimization");
		optimization_thread = std::thread(&PoseGraph::optimize6DoF, this);
	}
}

void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop) {
	// shift to base frame
	Eigen::Vector3d vio_P_cur;
	Eigen::Matrix3d vio_R_cur;
	if (sequence_cnt != cur_kf->sequence) {
		sequence_cnt++;
		sequence_loop.push_back(false);
		// if is uncontinued, reset state
		w_t_vio = Eigen::Vector3d::Zero();
		w_r_vio = Eigen::Matrix3d::Identity();
		m_drift.lock();
		t_drift = Eigen::Vector3d::Zero();
		r_drift = Eigen::Matrix3d::Identity();
		m_drift.unlock();
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
	//  detect loop closure
	if (loop_index != -1) {
		// get old keyframe with loop index find before
		KeyFrame* old_kf = getKeyFrame(loop_index);

		// descriptor matching between the current frame and the loopback candidate frame
		if (cur_kf->findConnection(old_kf)) {
			if (earliest_loop_index > loop_index || earliest_loop_index == -1)
				earliest_loop_index = loop_index;

			// calculate the relative pose of the current frame and the loopback frame, and correct the pose of the
			// current frame
			Eigen::Vector3d w_P_old, w_P_cur, vio_P_cur;
			Eigen::Matrix3d w_R_old, w_R_cur, vio_R_cur;
			old_kf->getVioPose(w_P_old, w_R_old);
			cur_kf->getVioPose(vio_P_cur, vio_R_cur);

			Eigen::Vector3d relative_t;
			Eigen::Quaterniond relative_q;
			relative_t = cur_kf->getLoopRelativeT();
			relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
			w_P_cur = w_R_old * relative_t + w_P_old;
			w_R_cur = w_R_old * relative_q;

			// the shift between the pose obtained by loopback and the VIO pose
			double shift_yaw;
			Eigen::Matrix3d shift_r;
			Eigen::Vector3d shift_t;
			if (use_imu) {
				shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
				shift_r = Utility::ypr2R(Eigen::Vector3d(shift_yaw, 0, 0));
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
						Eigen::Vector3d vio_P_cur;
						Eigen::Matrix3d vio_R_cur;
						(*it)->getVioPose(vio_P_cur, vio_R_cur);
						vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
						vio_R_cur = w_r_vio * vio_R_cur;
						(*it)->updateVioPose(vio_P_cur, vio_R_cur);
					}
				}
				sequence_loop[cur_kf->sequence] = true;
			}
			// put the current frame in the optimization queue
			m_optimize_buf.lock();
			optimize_buf.push(cur_kf->index);
			m_optimize_buf.unlock();
		}
	}
	m_keyframelist.lock();
	// get the pose R & T of the current frame of the VIO, and calculate the actual pose according to the offset,then
	// update the pose
	Eigen::Vector3d P;
	Eigen::Matrix3d R;
	cur_kf->getVioPose(P, R);
	P = r_drift * P + t_drift;
	R = r_drift * R;
	cur_kf->updatePose(P, R);

	// publisher state update
	Eigen::Quaterniond Q{R};
	geometry_msgs::msg::PoseStamped pose_stamped;
	int sec_ts = (int)cur_kf->time_stamp;
	uint nsec_ts = (uint)((cur_kf->time_stamp - sec_ts) * 1e9);
	pose_stamped.header.stamp.sec = sec_ts;
	pose_stamped.header.stamp.nanosec = nsec_ts;

	pose_stamped.header.frame_id = "world";
	pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
	pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
	pose_stamped.pose.position.z = P.z();
	pose_stamped.pose.orientation.x = Q.x();
	pose_stamped.pose.orientation.y = Q.y();
	pose_stamped.pose.orientation.z = Q.z();
	pose_stamped.pose.orientation.w = Q.w();
	path[sequence_cnt].poses.push_back(pose_stamped);
	path[sequence_cnt].header = pose_stamped.header;

	// save loop result file into VINS_RESULT_PATH
	ofstream loop_path_file(LOOP_FUSION_RESULT_PATH, ios::app);
	double time_stamp = cur_kf->time_stamp;
	loop_path_file.setf(ios::fixed, ios::floatfield);
	loop_path_file << time_stamp << " ";
	loop_path_file << P.x() << " " << P.y() << " " << P.z() << " " << Q.x() << " " << Q.y() << " " << Q.z() << " "
	               << Q.w() << endl;
	loop_path_file.close();

	keyframelist.push_back(cur_kf);

	// publish topics
	pubPoseGraph(*this);
	m_keyframelist.unlock();
}

KeyFrame* PoseGraph::getKeyFrame(int index) {
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

int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index) {
	// first query; then add this frame into database!
	DBoW2::QueryResults ret;
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

void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe) { db.add(keyframe->brief_descriptors); }

void PoseGraph::optimize4DoF() {
	// thread start
	while (true) {
		//  1. get keyframe index from optimize queue
		int cur_index = -1;
		int first_looped_index = -1;
		// ROS_INFO("optimize_buf size: %d", optimize_buf.size());
		m_optimize_buf.lock();
		while (!optimize_buf.empty()) {
			cur_index = optimize_buf.front();
			first_looped_index = earliest_loop_index;
			optimize_buf.pop();
		}
		m_optimize_buf.unlock();
		// ROS_INFO("cur_index: %d", cur_index);
		//  2. find cur index, start optimize
		if (cur_index != -1) {
			LOGGER_INFO("optimize pose graph");
			m_keyframelist.lock();
			// get cur keyframe
			KeyFrame* cur_kf = getKeyFrame(cur_index);

			int max_length = cur_index + 1;

			// ceres optimize parameters Rwb, Twb
			double t_array[max_length][3];
			Eigen::Quaterniond q_array[max_length];
			double euler_array[max_length][3];
			double sequence_array[max_length];

			// ceres problem config
			ceres::Problem problem;
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
			options.max_num_iterations = 5;
			ceres::Solver::Summary summary;
			ceres::LossFunction* loss_function;
			loss_function = new ceres::HuberLoss(0.1);

			ceres::Manifold* angle_local_parameterization = AngleLocalParameterization::Create();

			list<KeyFrame*>::iterator it;
			// traverse keyframe list
			int i = 0;
			for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
				if ((*it)->index < first_looped_index)
					continue;
				(*it)->local_index = i;
				Eigen::Quaterniond tmp_q;
				Eigen::Matrix3d tmp_r;
				Eigen::Vector3d tmp_t;
				(*it)->getVioPose(tmp_t, tmp_r);
				tmp_q = tmp_r;
				t_array[i][0] = tmp_t(0);
				t_array[i][1] = tmp_t(1);
				t_array[i][2] = tmp_t(2);
				q_array[i] = tmp_q;

				Eigen::Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
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

				// add residual block
				for (int j = 1; j < 5; j++) {
					if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
						Eigen::Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
						Eigen::Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
						                           t_array[i][2] - t_array[i - j][2]);
						relative_t = q_array[i - j].inverse() * relative_t;
						double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
						ceres::CostFunction* cost_function =
						    FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
						                         euler_conncected.y(), euler_conncected.z());
						problem.AddResidualBlock(cost_function, nullptr, euler_array[i - j], t_array[i - j],
						                         euler_array[i], t_array[i]);
					}
				}

				// add loop edge
				if ((*it)->has_loop) {
					assert((*it)->loop_index >= first_looped_index);
					int connected_index = getKeyFrame((*it)->loop_index)->local_index;
					Eigen::Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
					Eigen::Vector3d relative_t;
					relative_t = (*it)->getLoopRelativeT();
					double relative_yaw = (*it)->getLoopRelativeYaw();
					ceres::CostFunction* cost_function =
					    FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
					                         euler_conncected.y(), euler_conncected.z());
					problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
					                         t_array[connected_index], euler_array[i], t_array[i]);
				}

				if ((*it)->index == cur_index)
					break;
				i++;
			}
			m_keyframelist.unlock();

			ceres::Solve(options, &problem, &summary);

			m_keyframelist.lock();
			// convert array to Eigen, update state
			i = 0;
			for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
				if ((*it)->index < first_looped_index)
					continue;
				Eigen::Quaterniond tmp_q;
				tmp_q = Utility::ypr2R(Eigen::Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
				Eigen::Vector3d tmp_t = Eigen::Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
				Eigen::Matrix3d tmp_r = tmp_q.toRotationMatrix();
				(*it)->updatePose(tmp_t, tmp_r);

				if ((*it)->index == cur_index)
					break;
				i++;
			}

			Eigen::Vector3d cur_t, vio_t;
			Eigen::Matrix3d cur_r, vio_r;
			cur_kf->getPose(cur_t, cur_r);
			cur_kf->getVioPose(vio_t, vio_r);

			// update drift
			m_drift.lock();
			yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
			r_drift = Utility::ypr2R(Eigen::Vector3d(yaw_drift, 0, 0));
			t_drift = cur_t - r_drift * vio_t;
			m_drift.unlock();

			it++;
			for (; it != keyframelist.end(); it++) {
				Eigen::Vector3d P;
				Eigen::Matrix3d R;
				(*it)->getVioPose(P, R);
				P = r_drift * P + t_drift;
				R = r_drift * R;
				(*it)->updatePose(P, R);
			}
			m_keyframelist.unlock();
			updatePath();
		}

		// sleep 2000ms
		std::chrono::milliseconds dura(2000);
		std::this_thread::sleep_for(dura);
	}
}

void PoseGraph::optimize6DoF() {
	// thread start
	while (true) {
		//  1. get keyframe index from optimize queue
		int cur_index = -1;
		int first_looped_index = -1;
		m_optimize_buf.lock();
		while (!optimize_buf.empty()) {
			cur_index = optimize_buf.front();
			first_looped_index = earliest_loop_index;
			optimize_buf.pop();
		}
		m_optimize_buf.unlock();
		//  2. find cur index, start optimize
		if (cur_index != -1) {
			LOGGER_INFO("optimize pose graph");
			m_keyframelist.lock();
			// get cur keyframe
			KeyFrame* cur_kf = getKeyFrame(cur_index);

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
			ceres::LossFunction* loss_function;
			loss_function = new ceres::HuberLoss(0.1);
			ceres::Manifold* local_parameterization = new ceres::QuaternionManifold();

			list<KeyFrame*>::iterator it;
			// traverse keyframe list
			int i = 0;
			for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
				if ((*it)->index < first_looped_index)
					continue;
				(*it)->local_index = i;
				Eigen::Quaterniond tmp_q;
				Eigen::Matrix3d tmp_r;
				Eigen::Vector3d tmp_t;
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

				// add residual block
				for (int j = 1; j < 5; j++) {
					if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
						Eigen::Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
						                           t_array[i][2] - t_array[i - j][2]);
						Eigen::Quaterniond q_i_j = Eigen::Quaterniond(q_array[i - j][0], q_array[i - j][1],
						                                              q_array[i - j][2], q_array[i - j][3]);
						Eigen::Quaterniond q_i =
						    Eigen::Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
						relative_t = q_i_j.inverse() * relative_t;
						Eigen::Quaterniond relative_q = q_i_j.inverse() * q_i;
						ceres::CostFunction* vo_function =
						    SixDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(),
						                        relative_q.x(), relative_q.y(), relative_q.z(), 0.1, 0.01);
						problem.AddResidualBlock(vo_function, nullptr, q_array[i - j], t_array[i - j], q_array[i],
						                         t_array[i]);
					}
				}

				// add loop edge

				if ((*it)->has_loop) {
					assert((*it)->loop_index >= first_looped_index);
					int connected_index = getKeyFrame((*it)->loop_index)->local_index;
					Eigen::Vector3d relative_t;
					relative_t = (*it)->getLoopRelativeT();
					Eigen::Quaterniond relative_q;
					relative_q = (*it)->getLoopRelativeQ();
					ceres::CostFunction* loop_function =
					    SixDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(),
					                        relative_q.x(), relative_q.y(), relative_q.z(), 0.1, 0.01);
					problem.AddResidualBlock(loop_function, loss_function, q_array[connected_index],
					                         t_array[connected_index], q_array[i], t_array[i]);
				}

				if ((*it)->index == cur_index)
					break;
				i++;
			}
			m_keyframelist.unlock();

			ceres::Solve(options, &problem, &summary);

			m_keyframelist.lock();
			// convert array to Eigen, update state
			i = 0;
			for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
				if ((*it)->index < first_looped_index)
					continue;
				Eigen::Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
				Eigen::Vector3d tmp_t = Eigen::Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
				Eigen::Matrix3d tmp_r = tmp_q.toRotationMatrix();
				(*it)->updatePose(tmp_t, tmp_r);

				if ((*it)->index == cur_index)
					break;
				i++;
			}

			Eigen::Vector3d cur_t, vio_t;
			Eigen::Matrix3d cur_r, vio_r;
			cur_kf->getPose(cur_t, cur_r);
			cur_kf->getVioPose(vio_t, vio_r);

			// update drift
			m_drift.lock();
			r_drift = cur_r * vio_r.transpose();
			t_drift = cur_t - r_drift * vio_t;
			m_drift.unlock();

			it++;
			for (; it != keyframelist.end(); it++) {
				Eigen::Vector3d P;
				Eigen::Matrix3d R;
				(*it)->getVioPose(P, R);
				P = r_drift * P + t_drift;
				R = r_drift * R;
				(*it)->updatePose(P, R);
			}
			m_keyframelist.unlock();
			updatePath();
		}
		// sleep 2000ms
		std::chrono::milliseconds dura(2000);
		std::this_thread::sleep_for(dura);
	}
}

void PoseGraph::updatePath() {
	m_keyframelist.lock();
	list<KeyFrame*>::iterator it;
	for (int i = 1; i <= sequence_cnt; i++) {
		path[i].poses.clear();
	}
	base_path.poses.clear();

	// update all keyframe state
	for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
		Eigen::Vector3d P;
		Eigen::Matrix3d R;
		(*it)->getPose(P, R);
		Eigen::Quaterniond Q{R};

		// publisher state update
		geometry_msgs::msg::PoseStamped pose_stamped;
		int sec_ts = (int)(*it)->time_stamp;
		uint nsec_ts = (uint)(((*it)->time_stamp - sec_ts) * 1e9);
		pose_stamped.header.stamp.sec = sec_ts;
		pose_stamped.header.stamp.nanosec = nsec_ts;

		pose_stamped.header.frame_id = "world";
		pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
		pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
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
	m_keyframelist.unlock();
}

} // namespace FLOW_VINS