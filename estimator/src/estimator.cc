/**
 * @Author: Zhang Kaixiang
 * @Date: 2022-12-21 17:55:38
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Description: 位姿估计
 */

#include "../include/estimator.h"

namespace FLOW_VINS {

bool ManifoldParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const {
	// plus pose and quaternion
	Eigen::Map<const Eigen::Vector3d> _p(x);
	Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
	Eigen::Map<const Eigen::Vector3d> dp(delta);

	Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
	Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
	Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

	p = _p + dp;
	q = (_q * dq).normalized();

	return true;
}

bool ManifoldParameterization::PlusJacobian(const double* x, double* jacobian) const {
	// remap double array to Matrix, preference row
	Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
	// set the jacobian for variable corresponding to the location with respect to itself
	j.topRows<6>().setIdentity();
	j.bottomRows<1>().setZero();
	return true;
}

Estimator::Estimator()
    : init_thread_flag(false)
    , eskf_estimator()
    , visual_residual_weight(1.0)
    , imu_residual_weight(1.0)
    , mag_residual_weight(1.0) {
	LOGGER_INFO("init begins");
	clearState();
}

Estimator::~Estimator() {
	if (MULTIPLE_THREAD) {
		// waiting for process thread released
		process_thread.join();
		LOGGER_INFO("join thread. ");
	}
}

void Estimator::clearState() {
	mutex_process.lock();
	while (!acc_buf.empty())
		acc_buf.pop();
	while (!gyr_buf.empty())
		gyr_buf.pop();
	while (!feature_buf.empty())
		feature_buf.pop();
	mutex_process.unlock();

	solver_flag = INITIAL;
	marginalization_flag = MARGIN_OLD;
	prev_time = -1;
	cur_time = 0;
	image_count = 0;
	open_ex_estimation = false;
	init_first_pose_flag = false;

	for (int i = 0; i < WINDOW_SIZE + 1; i++) {
		Rs[i].setIdentity();
		Ps[i].setZero();
		Vs[i].setZero();
		Bas[i].setZero();
		Bgs[i].setZero();
		dt_buf[i].clear();
		linear_acceleration_buf[i].clear();
		angular_velocity_buf[i].clear();

		if (pre_integrations[i] != nullptr) {
			delete pre_integrations[i];
		}
		pre_integrations[i] = nullptr;
	}

	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = Eigen::Vector3d::Zero();
		ric[i] = Eigen::Matrix3d::Identity();
	}

	first_imu_flag = false;
	frame_count = 0;
	initial_timestamp = 0;
	all_image_frame.clear();
	delete tmp_pre_integration;
	delete last_marginalization_info;
	last_marginalization_info = nullptr;
	last_marginalization_parameter_blocks.clear();
	tmp_pre_integration = nullptr;
	feature_manager.clearState();
	failure_occur_flag = false;
}

void Estimator::setParameter() {
	// 1. set estimator parameters
	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = TIC[i];
		ric[i] = RIC[i];
		LOGGER_INFO(" extrinsic cam ", i, "\n", ric[i], "\n", tic[i].transpose());
	}
	feature_manager.setRic(ric);
	ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
	ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
	ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();

	td = TD;
	g = G;
	LOGGER_INFO("set g: ", g.transpose());

	feature_manager.ft.readIntrinsicParameter(CAM_NAMES);

	// 2. start solver thread
	LOGGER_INFO("MULTIPLE_THREAD is ", MULTIPLE_THREAD);
	if (MULTIPLE_THREAD && !init_thread_flag) {
		init_thread_flag = true;
		process_thread = std::thread(&Estimator::processMeasurements, this);
	}
}

void Estimator::inputImu(double t, const Eigen::Vector3d& linear_acceleration,
                         const Eigen::Vector3d& angular_velocity) {
	// IMU data input
	mutex_buf.lock();
	acc_buf.emplace(t, linear_acceleration);
	gyr_buf.emplace(t, angular_velocity);
	mutex_buf.unlock();
}

void Estimator::inputAhrs(double t, const Eigen::Vector3d& linear_acceleration, const Eigen::Vector3d& angular_velocity,
                          const Eigen::Vector3d& magnetometer) {
	// AHRS data input
	mutex_buf.lock();
	acc_buf.emplace(t, linear_acceleration);
	gyr_buf.emplace(t, angular_velocity);
	mag_buf.emplace(t, magnetometer);
	mutex_buf.unlock();
}

void Estimator::inputImage(double t, const cv::Mat& _img, const cv::Mat& _img1, const cv::Mat& _mask) {
	// 1. feature tracker input image
	// feature_frame construct: feature id | camera id | feature ( x y z u v vx vy )
	image_count++;
	FeatureFrame feature_frame;
	if (_img1.empty())
		feature_frame = feature_manager.ft.trackImage(t, _img);
	else {
		if (DEPTH) {
			LOGGER_DEBUG("input depth image");
			feature_frame = feature_manager.ft.trackImage(t, _img);
			feature_manager.setDepthImage(_img1);
		} else {
			LOGGER_DEBUG("input right image");
			feature_frame = feature_manager.ft.trackImage(t, _img, _img1);
		}
		if (USE_SEGMENTATION)
			feature_manager.ft.setSemanticMask(_mask);
	}

	// 2. add feature points to feature buffer, if system run in single thread, process it immediately
	if (MULTIPLE_THREAD) {
		if (image_count % FREQ == 0) {
			mutex_buf.lock();
			feature_buf.emplace(t, feature_frame);
			mutex_buf.unlock();
		}

	} else {
		mutex_buf.lock();
		feature_buf.emplace(t, feature_frame);
		mutex_buf.unlock();
		processMeasurements();
	}

	// 3. publish tracking image
	if (SHOW_TRACK)
		pubTrackImage(feature_manager.ft.getTrackImage(), t);
}

void Estimator::processMeasurements() {
	while (true) {
		// get feature and IMU data from buffer
		std::pair<double, FeatureFrame> feature;
		std::vector<std::pair<double, Eigen::Vector3d>> acc_vector, gyr_vector, mag_vector;

		if (!feature_buf.empty()) {
			TicToc t_vio;
			// current frame feature points
			feature = feature_buf.front();
			cur_time = feature.first + td;
			// check if IMU data is available
			while (true) {
				if (!USE_IMU || ImuAvailable(feature.first + td))
					break;
				else {
					if (!MULTIPLE_THREAD)
						return;
					LOGGER_DEBUG("waiting for IMU data");
					std::chrono::milliseconds dura(5);
					std::this_thread::sleep_for(dura);
				}
			}
			// mutex lock
			mutex_buf.lock();
			if (USE_IMU) {
				if (!USE_MAGNETOMETER)
					// get IMU data interval between the newest frame and last frame
					getImuInterval(prev_time, cur_time, acc_vector, gyr_vector);
				else
					// get AHRS data interval between the newest frame and last frame
					getAhrsInterval(prev_time, cur_time, acc_vector, gyr_vector, mag_vector);
			}

			feature_buf.pop();
			mutex_buf.unlock();

			if (USE_IMU) {
				// initialize first frame IMU pose, make acceleration z axis point to gravity
				if (!init_first_pose_flag) {
					if (!USE_MAGNETOMETER)
						initFirstImuPose(acc_vector);
					else
						initFirstAhrsPose(acc_vector, mag_vector);
				}

				// process IMU and compute IMU variance
				int n = (int)acc_vector.size();
				// get last 50 frame for variance compute
				int var_size = n > 50 ? 50 : n;

				Eigen::Vector3d sum_g = {0, 0, 0};
				for (auto i = 0; i < n; ++i) {
					double dt, var_dt;
					if (i == 0)
						dt = acc_vector[i].first - prev_time;
					else if (i == n - 1)
						dt = cur_time - acc_vector[i - 1].first;
					else
						dt = acc_vector[i].first - acc_vector[i - 1].first;

					if (n - var_size + i == 0)
						var_dt = acc_vector[n - var_size + i].first - prev_time;
					else if (n - var_size + i == n - 1)
						var_dt = cur_time - acc_vector[i - 1].first;
					else
						var_dt = acc_vector[n - var_size + i].first - acc_vector[n - var_size + i - 1].first;
					processIMU(acc_vector[i].first, dt, acc_vector[i].second, gyr_vector[i].second);

					// calculate the standard deviation of the IMU acceleration in sliding window to judge the speed of
					// movement
					Eigen::Vector3d tmp_g = acc_vector[n - var_size + i].second / var_dt;
					sum_g += tmp_g;

					// use ESKF to estimator pose
					if (USE_MAGNETOMETER) {
						Eigen::Matrix<double, 10, 1> measurement;
						measurement << acc_vector[i].second, gyr_vector[i].second, mag_vector[i].second,
						    acc_vector[i].first;
						Eigen::Quaterniond q = eskf_estimator.run(measurement);
						q.normalize();

						Eigen::Vector3d euler; //= quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

						euler(0) = atan2(2 * (q.y() * q.z() + q.w() * q.x()),
						                 (q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z()));
						euler(1) = asin(-2 * q.x() * q.z() + 2 * q.w() * q.y());
						euler(2) = atan2(2 * (q.x() * q.y() + q.w() * q.z()),
						                 (q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z()));
						euler *= 180 / M_PI;
					}
				}
				// average of acceleration
				Eigen::Vector3d aver_g;
				aver_g = sum_g * 1.0 / (var_size - 1);
				// variance of acceleration
				double var = 0;
				for (auto i = n - var_size; i < n; i++) {
					double dt;
					if (i == 0)
						dt = acc_vector[i].first - prev_time;
					else if (i == n - 1)
						dt = cur_time - acc_vector[i - 1].first;
					else
						dt = acc_vector[i].first - acc_vector[i - 1].first;

					Eigen::Vector3d tmp_g = acc_vector[i].second / dt;
					var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
				}
				// calculate the standard deviation of the IMU acceleration
				var = sqrt(var / (n - 1));
				LOGGER_WARN("Imu Variance: ", var);
			}

			prev_time = cur_time;
			// process image
			processImage(feature.second, feature.first);

			printStatistics(*this, t_vio.toc());

			// publish topic data
			std_msgs::msg::Header header;
			header.frame_id = "world";
			int sec_ts = (int)feature.first;
			uint nsec_ts = (uint)((feature.first - sec_ts) * 1e9);
			header.stamp.sec = sec_ts;
			header.stamp.nanosec = nsec_ts;

			pubOdometry(*this, header);
			pubKeyPoses(*this, header);
			pubCameraPose(*this, header);
			pubPointCloud(*this, header);
			pubKeyframe(*this);
			pubTF(*this, header);
		}

		if (!MULTIPLE_THREAD)
			break;

		std::chrono::milliseconds dura(2);
		std::this_thread::sleep_for(dura);
	}
}

void Estimator::processIMU(double t, double dt, const Eigen::Vector3d& linear_acceleration,
                           const Eigen::Vector3d& angular_velocity) {
	// if input data is the first data, set flag and init state
	if (!first_imu_flag) {
		first_imu_flag = true;
		acc_0 = linear_acceleration;
		gyr_0 = angular_velocity;
	}

	// IMU pre-integration initialize at frame count position
	if (!pre_integrations[frame_count])
		pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

	if (frame_count != 0) {
		// IMU pre-integration process at frame count position
		pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
		tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

		// cache IMU data
		dt_buf[frame_count].emplace_back(dt);
		linear_acceleration_buf[frame_count].emplace_back(linear_acceleration);
		angular_velocity_buf[frame_count].emplace_back(angular_velocity);

		// midian integration process
		int i = frame_count;
		// acceleration at the previous moment
		Eigen::Vector3d un_acc_0 = Rs[i] * (acc_0 - Bas[i]) - g;
		// gyro median integration
		Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[i];
		// get current moment Q
		Rs[i] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
		// acceleration at the current moment
		Eigen::Vector3d un_acc_1 = Rs[i] * (linear_acceleration - Bas[i]) - g;
		// acceleration median integration
		Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
		// get current moment P
		Ps[i] += dt * Vs[i] + 0.5 * dt * dt * un_acc;
		// get current moment V
		Vs[i] += dt * un_acc;
	}
	// update acc_0 and gyr_0 with current data
	acc_0 = linear_acceleration;
	gyr_0 = angular_velocity;
}

void Estimator::processImage(FeatureFrame& image, double header) {
	LOGGER_DEBUG("new image coming --------------------------------------");
	LOGGER_DEBUG("Adding feature points ", image.size());
	// 1. add feature point records, check whether the current frame is keyframe
	if (feature_manager.addFeatureCheckParallax(frame_count, image, td)) {
		marginalization_flag = MARGIN_OLD;
	} else {
		marginalization_flag = MARGIN_SECOND_NEW;
	}
	LOGGER_DEBUG(marginalization_flag ? "Non-keyframe" : "Keyframe");
	LOGGER_DEBUG("Solving ", frame_count);
	LOGGER_DEBUG("number of feature: ", feature_manager.getFeatureCount());
	headers[frame_count] = header;

	ImageFrame imageframe(image, header);
	imageframe.pre_integration = tmp_pre_integration;
	all_image_frame.insert(std::make_pair(header, imageframe));
	// reset tmp pre-integration
	tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

	// 2. main process of vio system
	if (solver_flag == INITIAL) {
		initialize(header);
	} else {
		backend();
	}
	LOGGER_DEBUG("process image finish");
}

void Estimator::initialize(const double& header) {
	// monocular + IMU initialization
	if (!(STEREO || DEPTH) && USE_IMU) {
		// require sliding window is filled
		if (frame_count == WINDOW_SIZE) {
			bool result = false;
			// if last initialization is failed, require time interval is greater than 0.1s
			if ((header - initial_timestamp) > 0.1) {
				result = initialStructure();
				initial_timestamp = header;
			}
			// if initialization is success, implement optimize and sliding window, otherwise only sliding window
			if (result) {
				solver_flag = NON_LINEAR;
				optimization();
				slideWindow();
				LOGGER_DEBUG("Initialization finish!");
			} else {
				slideWindow();
			}
		}
	}
	// stereo + IMU initilization
	if (STEREO && USE_IMU) {
		// solve pnp and triangulate
		feature_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		feature_manager.triangulate(Ps, Rs, tic, ric);

		if (frame_count == WINDOW_SIZE) {
			int i = 0;
			std::map<double, ImageFrame>::iterator frame_it;
			for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
				frame_it->second.R = Rs[i];
				frame_it->second.T = Ps[i];
				i++;
			}
			bool result = false;
			if (header - initial_timestamp > 0.1) {
				result = true;
				initial_timestamp = header;
			}
			// initialize IMU bias
			if (result) {
				solver_flag = NON_LINEAR;
				solveGyroscopeBias(all_image_frame, Bgs);
				for (int j = 0; j <= WINDOW_SIZE; j++) {
					pre_integrations[j]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
				}
				optimization();
				slideWindow();
				LOGGER_DEBUG("Initialization finish!");
			} else {
				slideWindow();
			}
		}
	}

	// stereo initilization
	if (STEREO && !USE_IMU) {
		feature_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		feature_manager.triangulate(Ps, Rs, tic, ric);

		if (frame_count == WINDOW_SIZE) {
			solver_flag = NON_LINEAR;
			optimization();
			slideWindow();
			LOGGER_DEBUG("Initialization finish!");
		}
	}

	if (DEPTH) {
		// RGB_D with IMU direct initilization
		feature_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		feature_manager.triangulate(Ps, Rs, tic, ric);

		if (USE_IMU) {
			if (frame_count == WINDOW_SIZE) {
				std::map<double, ImageFrame>::iterator frame_it;
				int i = 0;
				for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
					frame_it->second.R = Rs[i];
					frame_it->second.T = Ps[i];
					i++;
				}
				bool result = false;
				if (header - initial_timestamp > 0.1) {
					result = true;
					initial_timestamp = header;
				}
				// initialize IMU bias
				if (result) {
					solver_flag = NON_LINEAR;
					solveGyroscopeBias(all_image_frame, Bgs);
					for (int j = 0; j <= WINDOW_SIZE; j++) {
						pre_integrations[j]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
					}
					optimization();
					slideWindow();
					LOGGER_DEBUG("Initialization finish!");

				} else {
					slideWindow();
				}
			}
		} else {
			if (frame_count == WINDOW_SIZE) {
				solver_flag = NON_LINEAR;
				optimization();
				slideWindow();
				LOGGER_DEBUG("Initialization finish!");
			}
		}
		// without imu
	}

	if (frame_count < WINDOW_SIZE) {
		frame_count++;
		int prev_frame = frame_count - 1;
		Ps[frame_count] = Ps[prev_frame];
		Vs[frame_count] = Vs[prev_frame];
		Rs[frame_count] = Rs[prev_frame];
		Bas[frame_count] = Bas[prev_frame];
		Bgs[frame_count] = Bgs[prev_frame];
	}
}

bool Estimator::initialStructure() {
	// 1.check imu observability
	bool is_imu_excited = checkImuObservibility();

	// 2.build sfm features which are used in initialization
	std::vector<SFMFeature> sfm_feature;
	buildSfmFeature(sfm_feature);

	// 3.find frame l in sliding window, whose average parallax is above threshold
	// and find the relative R and T(l < -11) between frame l and the newest frame(No.11)
	Eigen::Matrix3d relative_R;
	Eigen::Vector3d relative_T;
	int l;
	if (!relativePose(relative_R, relative_T, l)) {
		LOGGER_WARN("Not enough features or parallax; Move device around");
		return false;
	}

	// 4.solve SFM problems to get the rotation and translation for all frames and landmarks in frame l coordinate
	// system
	GlobalSFM sfm;
	// Q: Rwc, set frame l as reference frame, T is same as Q
	Eigen::Quaterniond Q[frame_count + 1];
	Eigen::Vector3d T[frame_count + 1];
	std::map<int, Eigen::Vector3d> sfm_tracked_points;
	if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_feature, sfm_tracked_points)) {
		LOGGER_WARN("global SFM failed!");
		// if sfm triangulate failed, marginalize the oldest frame
		marginalization_flag = MARGIN_OLD;
		return false;
	}

	// 5.solve pnp for all frame
	if (!solvePnPForAllFrame(Q, T, sfm_tracked_points)) {
		LOGGER_WARN("solve PnP for all frame failed!");
		return false;
	}

	// 6.align visual and IMU for initialization

	if (visualInitialAlign()) {
		if (!is_imu_excited) {
			// estimate Bas by average of acceleration
			Eigen::Vector3d sum_a(0, 0, 0);
			std::map<double, ImageFrame>::iterator frame_it;
			for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
				double dt = frame_it->second.pre_integration->sum_dt;
				Eigen::Vector3d tmp_a = frame_it->second.pre_integration->delta_v / dt;
				sum_a += tmp_a;
			}
			Eigen::Vector3d avg_a;
			avg_a = sum_a * 1.0 / ((int)all_image_frame.size() - 1);

			Eigen::Vector3d tmp_Bas = avg_a - Utility::g2R(avg_a).inverse() * G;
			LOGGER_WARN("accelerator bias initial calibration ", tmp_Bas.transpose());
			for (int i = 0; i <= WINDOW_SIZE; i++) {
				Bas[i] = tmp_Bas;
			}
		}
		return true;
	} else {
		LOGGER_WARN("misalign visual structure with IMU");
		return false;
	}
}

void Estimator::initFirstImuPose(std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector) {
	LOGGER_INFO("init first imu pose");
	init_first_pose_flag = true;
	// calculate average acceleration
	Eigen::Vector3d average_acc(0, 0, 0);
	int n = (int)acc_vector.size();
	for (auto& i : acc_vector) {
		average_acc = average_acc + i.second;
	}
	average_acc = average_acc / n;
	LOGGER_INFO("averge acc", average_acc.transpose());

	// calculate initial rotation for IMU axis z alignment to gravity
	Eigen::Matrix3d R0 = Utility::g2R(average_acc);
	Rs[0] = R0;
	LOGGER_INFO("init R0 ", "\n", Rs[0]);
}

void Estimator::initFirstAhrsPose(std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
                                  std::vector<std::pair<double, Eigen::Vector3d>>& mag_vector) {
	LOGGER_INFO("init first ahrs pose");
	init_first_pose_flag = true;
	// calculate average acceleration
	Eigen::Vector3d average_acc(0, 0, 0);
	int n = (int)acc_vector.size();
	for (auto& i : acc_vector) {
		average_acc = average_acc + i.second;
	}
	average_acc = average_acc / n;
	LOGGER_INFO("averge acc", average_acc.transpose());

	// calculate average magnetometer
	Eigen::Vector3d average_mag(0, 0, 0);
	n = (int)mag_vector.size();
	for (auto& i : mag_vector) {
		average_mag = average_mag + i.second;
	}
	average_mag = average_mag / n;
	LOGGER_INFO("averge mag", average_mag.transpose());

	double mag_norm = average_mag.norm();

	// calculate initial rotation for IMU axis z alignment to gravity
	Eigen::Matrix3d R0 = Utility::g2R(average_acc);
	Rs[0] = R0;
	LOGGER_INFO("init R0 ", "\n", Rs[0]);
}

bool Estimator::ImuAvailable(double t) {
	if (!acc_buf.empty() && t <= acc_buf.back().first)
		return true;
	else
		return false;
}

bool Estimator::getImuInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
                               std::vector<std::pair<double, Eigen::Vector3d>>& gyr_vector) {
	if (acc_buf.empty()) {
		LOGGER_WARN("not receive imu");
		return false;
	}
	// extract the data of the time period (t0, t1)
	if (t1 <= acc_buf.back().first) {
		while (acc_buf.front().first <= t0) {
			acc_buf.pop();
			gyr_buf.pop();
		}
		while (acc_buf.front().first < t1) {
			acc_vector.emplace_back(acc_buf.front());
			acc_buf.pop();
			gyr_vector.emplace_back(gyr_buf.front());
			gyr_buf.pop();
		}
		acc_vector.emplace_back(acc_buf.front());
		gyr_vector.emplace_back(gyr_buf.front());
		return true;
	} else {
		return false;
	}
}

bool Estimator::getAhrsInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>>& acc_vector,
                                std::vector<std::pair<double, Eigen::Vector3d>>& gyr_vector,
                                std::vector<std::pair<double, Eigen::Vector3d>>& mag_vector) {
	if (acc_buf.empty()) {
		LOGGER_WARN("not receive ahrs");
		return false;
	}
	// extract the data of the time period (t0, t1)
	if (t1 <= acc_buf.back().first) {
		while (acc_buf.front().first <= t0) {
			acc_buf.pop();
			gyr_buf.pop();
			mag_buf.pop();
		}
		while (acc_buf.front().first < t1) {
			acc_vector.emplace_back(acc_buf.front());
			acc_buf.pop();
			gyr_vector.emplace_back(gyr_buf.front());
			gyr_buf.pop();
			mag_vector.emplace_back(mag_buf.front());
			mag_buf.pop();
		}
		acc_vector.emplace_back(acc_buf.front());
		gyr_vector.emplace_back(gyr_buf.front());
		mag_vector.emplace_back(mag_buf.front());
		return true;
	} else {
		return false;
	}
}

bool Estimator::checkImuObservibility() {
	// calculate the standard deviation of the IMU acceleration in sliding window to judge the speed of movement
	std::map<double, ImageFrame>::iterator frame_it;
	Eigen::Vector3d sum_g;
	// accumulate the acceleration of each frame from the second frame
	for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
		double dt = frame_it->second.pre_integration->sum_dt;
		Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
		sum_g += tmp_g;
	}
	// average of acceleration
	Eigen::Vector3d aver_g;
	aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
	// variance of acceleration
	double var = 0;
	for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
		double dt = frame_it->second.pre_integration->sum_dt;
		Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
		var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
	}
	// calculate the standard deviation of the IMU acceleration
	var = sqrt(var / ((int)all_image_frame.size() - 1));
	LOGGER_INFO("IMU variation ", var, "!");
	if (var < 0.5) {
		LOGGER_WARN("IMU excitation not enough!");
		return false;
	} else
		return true;
}

void Estimator::buildSfmFeature(std::vector<SFMFeature>& sfm_feature) {
	// traverse feature points in the current frame
	for (auto& it : feature_manager.feature) {
		auto& it_per_id = it.second;
		int imu_j = it_per_id.start_frame - 1;
		SFMFeature tmp_feature;
		tmp_feature.state = false;
		tmp_feature.id = it_per_id.feature_id;
		// traverse frames which include feature points
		for (auto& it_per_frame : it_per_id.feature_per_frame) {
			imu_j++;
			// feature point normalized camera plane point
			Eigen::Vector3d pts_j = it_per_frame.point;
			tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
		}
		sfm_feature.push_back(tmp_feature);
	}
}

bool Estimator::relativePose(Eigen::Matrix3d& relative_R, Eigen::Vector3d& relative_T, int& l) {
	// find previous frame which contains enough correspondence and parallax with the newest frame
	for (int i = 0; i < WINDOW_SIZE; i++) {
		// extract the matching points between each frame in sliding window and the current frame
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
		corres = feature_manager.getCorresponding(i, WINDOW_SIZE);
		// if correspond points number is greater than 20
		if (corres.size() > 20) {
			// calculate the cumulate average parallax as the parallax between current two frames
			double sum_parallax = 0;
			double average_parallax;
			for (auto& corre : corres) {
				Eigen::Vector2d pts_0(corre.first(0), corre.first(1));
				Eigen::Vector2d pts_1(corre.second(0), corre.second(1));
				double parallax = (pts_0 - pts_1).norm();
				sum_parallax = sum_parallax + parallax;
			}
			average_parallax = 1.0 * sum_parallax / int(corres.size());
			// the parallax exceeds 30 pixels, matching points of two frames calculate the essential matrix E and
			// restore R t
			if (average_parallax * 460 > 30 && solveRelativeRT(corres, relative_R, relative_T)) {
				l = i;
				LOGGER_INFO("average_parallax", average_parallax * 460, " choose l ", l,
				            " and newest frame to triangulate the whole structure");
				return true;
			}
		}
	}
	return false;
}

bool Estimator::solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& corres,
                                Eigen::Matrix3d& rotation, Eigen::Vector3d& translation) {
	if (corres.size() >= 15) {
		// traverse correspondence points and normalize point in camera coordinate
		std::vector<cv::Point2f> ll, rr;
		for (const auto& corre : corres) {
			ll.emplace_back(corre.first(0), corre.first(1));
			rr.emplace_back(corre.second(0), corre.second(1));
		}
		// calculate essential matrix E
		cv::Mat mask;
		cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
		cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		cv::Mat rot, trans;
		// restore R and t from essential matrix | R(r <-- l) t(r <-- l)
		int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

		// convert cv::Mat R & T to Eigen format
		Eigen::Matrix3d R;
		Eigen::Vector3d T;
		for (auto i = 0; i < 3; i++) {
			T(i) = trans.at<double>(i, 0);
			for (auto j = 0; j < 3; j++)
				R(i, j) = rot.at<double>(i, j);
		}

		// transformation from the current frame to the previous frame | R(l <-- r) t(l <-- r)
		rotation = R.transpose();
		translation = -R.transpose() * T;
		if (inlier_cnt > 12)
			return true;
		else
			return false;
	}
	return false;
}

bool Estimator::solvePnPForAllFrame(Eigen::Quaterniond Q[], Eigen::Vector3d T[],
                                    std::map<int, Eigen::Vector3d>& sfm_tracked_points) {
	std::map<double, ImageFrame>::iterator frame_it;
	std::map<int, Eigen::Vector3d>::iterator it;
	frame_it = all_image_frame.begin();
	for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
		// provide initial guess
		cv::Mat r, rvec, t, D, tmp_r;
		if (frame_it->first == headers[i]) {
			frame_it->second.is_key_frame = true;
			// R: R(c0<--ci) * Rci = R(c0<--ii), T: T(c0<--ci)
			frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
			frame_it->second.T = T[i];
			i++;
			continue;
		}
		if (frame_it->first > headers[i]) {
			i++;
		}
		// convert R(c0<--ci) to R(ci<--c0), T(c0<--ci) to T(ci<--c0)
		Eigen::Matrix3d R_initial = (Q[i].inverse()).toRotationMatrix();
		Eigen::Vector3d P_initial = -R_initial * T[i];
		cv::eigen2cv(R_initial, tmp_r);
		cv::Rodrigues(tmp_r, rvec);
		cv::eigen2cv(P_initial, t);

		frame_it->second.is_key_frame = false;
		std::vector<cv::Point3f> pts_3_vector;
		std::vector<cv::Point2f> pts_2_vector;
		// traverse feature points in the current frame
		for (auto& id_pts : frame_it->second.points) {
			int feature_id = id_pts.first;
			// traverse observe frame include feature point and extract pixel points and world points
			for (auto& i_p : id_pts.second) {
				it = sfm_tracked_points.find(feature_id);
				if (it != sfm_tracked_points.end()) {
					Eigen::Vector3d world_pts = it->second;
					cv::Point3f pts_3(static_cast<float>(world_pts(0)), static_cast<float>(world_pts(1)),
					                  static_cast<float>(world_pts(2)));
					pts_3_vector.push_back(pts_3);
					Eigen::Vector2d img_pts = i_p.second.head<2>();
					cv::Point2f pts_2(static_cast<float>(img_pts(0)), static_cast<float>(img_pts(1)));
					pts_2_vector.push_back(pts_2);
				}
			}
		}
		cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		// 3d point number check
		if (pts_3_vector.size() < 6) {
			LOGGER_WARN("pts_3_vector size ", pts_3_vector.size());
			LOGGER_WARN("Not enough points for solve pnp !");
			return false;
		}
		// 3d-2d PnP solve transformation
		if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true)) {
			LOGGER_WARN("solve pnp fail!");
			return false;
		}
		// R_pnp: R(c0<--ci), T_pnp: T(c0<--ci)
		cv::Rodrigues(rvec, r);
		Eigen::MatrixXd R_pnp, tmp_R_pnp;
		cv::cv2eigen(r, tmp_R_pnp);
		R_pnp = tmp_R_pnp.transpose();
		Eigen::MatrixXd T_pnp;
		cv::cv2eigen(t, T_pnp);
		T_pnp = R_pnp * (-T_pnp);
		// R: R(c0<--ci) * Rci = R(c0<--ii), T: T(c0<--ci), for visual and IMU alignment compute
		frame_it->second.R = R_pnp * RIC[0].transpose();
		frame_it->second.T = T_pnp;
	}
	return true;
}

bool Estimator::visualInitialAlign() {
	Eigen::VectorXd x;
	// 1. alignment with IMU and visual
	bool result = false;
	result = visualImuAlignment(all_image_frame, Bgs, g, x);
	if (!result) {
		LOGGER_WARN("solve g failed!");
		return false;
	}

	// 2. get all image frame pose and set it to keyframe, Rs: R(c0<--bi), Ps: T(c0<--ci)
	for (auto i = 0; i <= frame_count; i++) {
		Eigen::Matrix3d Ri = all_image_frame[headers[i]].R;
		Eigen::Vector3d Pi = all_image_frame[headers[i]].T;
		Ps[i] = Pi;
		Rs[i] = Ri;
		all_image_frame[headers[i]].is_key_frame = true;
	}

	// 3. update pre-integration
	for (int i = 0; i <= WINDOW_SIZE; i++) {
		pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
	}

	// 4. convert reference coordinate from frame l to IMU frame 0
	double s = (x.tail<1>())(0);
	// convert Ps: T(c0<--ci) to Ps: T(c0<--bi), is different with source code
	for (int i = frame_count; i >= 0; i--)
		Ps[i] = s * Ps[i] - Rs[i] * TIC[0];

	// 5. update speed
	int kv = -1;
	std::map<double, ImageFrame>::iterator frame_i;
	for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
		if (frame_i->second.is_key_frame) {
			kv++;
			// Vs: V(c0<--bi)
			Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
		}
	}

	// 6. rotate gravity vector into axis z and get rotation matrix, then convert all state from c0 to world coordinate
	Eigen::Matrix3d R0 = Utility::g2R(g);
	double yaw = Utility::R2ypr(R0 * Rs[0]).x();
	R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
	g = R0 * g;
	Eigen::Matrix3d rot_diff = R0;
	// P Q V: (w <-- bi) --> Twi
	for (int i = 0; i <= frame_count; i++) {
		Ps[i] = rot_diff * Ps[i];
		Rs[i] = rot_diff * Rs[i];
		Vs[i] = rot_diff * Vs[i];
	}
	LOGGER_INFO("g0     ", g.transpose());
	LOGGER_INFO("my R0  ", Utility::R2ypr(Rs[0]).transpose());

	// 7. triangulate to update depth
	feature_manager.clearDepth();
	feature_manager.triangulate(Ps, Rs, tic, ric);

	return true;
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d& T) {
	// convert R (3*3) and P (3*1) to Twi (4*4)
	T = Eigen::Matrix4d::Identity();
	T.block<3, 3>(0, 0) = Rs[index];
	T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame() {
	if (frame_count < 2)
		return;
	// predict next pose. Assume constant velocity motion
	// get last frame Twi and current frame Twi
	Eigen::Matrix4d curT, prevT, nextT;
	getPoseInWorldFrame(frame_count, curT);
	getPoseInWorldFrame(frame_count - 1, prevT);
	// predict next frame Twi with transformation between last frame and current frame
	/* formula derivation:
	 * Pw = Twi1 * Pi1, Pw = Twi2 * Pi2, Pi2 = T21 * Pi1   -->   T21 = Twi2^-1 * Twi1 = T32(predict)
	 * Pw = Twi3 * Pi3, Pi3 = T32 * Pi2                    -->   Twi2 * Pi2 = Twi3 * T32 * Pi2
	 * Twi3(predict) = Twi2 * Twi1^-1 * Twi2
	 */
	nextT = curT * (prevT.inverse() * curT);
	std::map<int, Eigen::Vector3d> predictPts;

	// traverse feature points
	for (auto& it : feature_manager.feature) {
		auto& it_per_id = it.second;
		if (it_per_id.estimated_depth > 0) {
			int firstIndex = it_per_id.start_frame;
			int lastIndex = it_per_id.start_frame + static_cast<int>(it_per_id.feature_per_frame.size()) - 1;
			if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) {
				double depth = it_per_id.estimated_depth;
				int ptsIndex = it_per_id.feature_id;
				// body coordinate of the feature points in the first observation frame
				Eigen::Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
				// convert body coordinate to world coordinate
				Eigen::Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
				// convert to next frame in body coordinate
				Eigen::Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
				// convert to camera coordinate
				Eigen::Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
				// set predict point
				predictPts[ptsIndex] = pts_cam;
			}
		}
	}
	// set initial positioin tracking points in next frame
	feature_manager.ft.setPrediction(predictPts);
}

void Estimator::backend() {
	// 1. solve odometry with optimization and marginalization
	solveOdometry();

	// 2. moving consistency check, calculate reprojection error and delete outliers
	std::set<int> remove_index;
	movingConsistencyCheck(remove_index);
	feature_manager.removeOutlier(remove_index);

	// 3. predict feature point position in next frame, assumption of constant velocity motion
	// predictPtsInNextFrame();

	// 4. failure detect, if system failure, clear estimator parameters and reboot vio system
	if (failureDetection()) {
		LOGGER_WARN("failure detection!");
		failure_occur_flag = true;
		clearState();
		setParameter();
		LOGGER_WARN("system reboot!");
		return;
	}

	// 5. sliding window, update frame set and index, delete feature points which are not in observed frame set
	slideWindow();

	// 6. remove feature points which depth is nagetive after ceres solver
	feature_manager.removeFailures();

	// 7. state update
	key_poses.clear();
	for (int i = 0; i <= WINDOW_SIZE; i++)
		key_poses.push_back(Ps[i]);

	last_R = Rs[WINDOW_SIZE];
	last_P = Ps[WINDOW_SIZE];
	last_R0 = Rs[0];
	last_P0 = Ps[0];
}

void Estimator::solveOdometry() {
	// only optimize when the sliding window is full and initialization is finish
	if (frame_count < WINDOW_SIZE)
		return;

	if (solver_flag == NON_LINEAR) {
		// 1. frontend process: pnp and triangulate to calculate initial pose guess
		if (!USE_IMU)
			feature_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);

		feature_manager.triangulate(Ps, Rs, tic, ric);
		// 2. backend: ceres solver to optimize state parameters and marginalize
		optimization();
	}
}

void Estimator::optimization() {
	// create ceres solver problem and loss function
	ceres::Problem problem;
	ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
	// ceres nonlinear optimization
	nonLinearOptimization(problem, loss_function);
	// marginalization operation
	if (marginalization_flag == MARGIN_OLD)
		// if the second frame is key frame, marginalize the oldest frame
		margOld(loss_function);
	else
		// else throw away the second frame directly
		margNew();
}

void Estimator::nonLinearOptimization(ceres::Problem& problem, ceres::LossFunction* loss_function) {
	// 1. convert state parameters in sliding window to array for ceres solver
	vector2double();
	ceres::LossFunction* visual_loss_function =
	    new ceres::ScaledLoss(loss_function, visual_residual_weight, ceres::TAKE_OWNERSHIP);
	ceres::LossFunction* imu_loss_function = new ceres::ScaledLoss(nullptr, imu_residual_weight, ceres::TAKE_OWNERSHIP);
	ceres::LossFunction* mag_loss_function = new ceres::ScaledLoss(nullptr, mag_residual_weight, ceres::TAKE_OWNERSHIP);

	// 2. add parameter block and set point to constant
	// traverse sliding window to add pose and speed bias parameters
	for (int i = 0; i < frame_count + 1; i++) {
		ceres::Manifold* local_parameterization = new ManifoldParameterization();
		problem.AddParameterBlock(para_pose[i], SIZE_POSE, local_parameterization);
		if (USE_IMU)
			problem.AddParameterBlock(para_speed_bias[i], SIZE_SPEEDBIAS);
	}
	// if IMU is unused, set the first frame pose to constant
	if (!USE_IMU)
		problem.SetParameterBlockConstant(para_pose[0]);

	// add camera and IMU extrinsic parameters
	for (int i = 0; i < NUM_OF_CAM; i++) {
		ceres::Manifold* local_parameterization = new ManifoldParameterization();
		problem.AddParameterBlock(para_ex_pose[i], SIZE_POSE, local_parameterization);
		// determine whether to estimate extrinsic parameters
		if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || open_ex_estimation) {
			open_ex_estimation = true;
		} else {
			problem.SetParameterBlockConstant(para_ex_pose[i]);
		}
	}
	// add camera and IMU time difference parameter
	problem.AddParameterBlock(para_td[0], 1);
	// determine whether to estimate time difference parameters
	if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
		problem.SetParameterBlockConstant(para_td[0]);

	// 3. add residual block
	// add priority residual, through Schur complement operation, superimpose the information of marginalization part on
	// the retained variables
	if (last_marginalization_info && last_marginalization_info->valid) {
		// construct new marginalization_factor
		auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
		problem.AddResidualBlock(marginalization_factor, nullptr, last_marginalization_parameter_blocks);
	}
	// add IMU residual
	if (USE_IMU) {
		for (int i = 0; i < frame_count; i++) {
			int j = i + 1;
			if (pre_integrations[j]->sum_dt > 10.0)
				continue;
			// build IMU residual between two frames
			auto* imu_factor = new IMUFactor(pre_integrations[j]);
			// state parameter dimension correspond to IMU factor dimension
			problem.AddResidualBlock(imu_factor, imu_loss_function, para_pose[i], para_speed_bias[i], para_pose[j],
			                         para_speed_bias[j]);
		}
	}

	// add visual reprojection residual

	// traverse feature points
	for (int _id : param_feature_id) {
		auto& it_per_id = feature_manager.feature[_id];

		int feature_index = param_feature_id_to_index[it_per_id.feature_id];

		int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

		// get normalize camera point in the first frame
		Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

		// traverse frames which observed feature point choosen
		for (auto& it_per_frame : it_per_id.feature_per_frame) {
			imu_j++;
			// get reprojection residual between two frame in one camera
			// only extract frames after the first frame
			if (imu_i != imu_j) {
				// get current frame normalize camera point
				Eigen::Vector3d pts_j = it_per_frame.point;
				// build reprojection residual between the first frame and the current frame
				auto* f_td = new ProjectionTwoFrameOneCamFactor(
				    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
				    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
				// optimize parameters: pose in the first frame and current frame, extrinsic parameters, reverse depth
				// in feature point and td between camera and IMU
				problem.AddResidualBlock(f_td, visual_loss_function, para_pose[imu_i], para_pose[imu_j],
				                         para_ex_pose[0], para_feature[feature_index], para_td[0]);
				// if depth guess is accuracy and set depth fixed, set reverse depth parameter constant
				if (it_per_id.estimate_flag == 1)
					problem.SetParameterBlockConstant(para_feature[feature_index]);
				// prevent reverse depth is too small in distant feature points
				else if (it_per_id.estimate_flag == 2) {
					problem.SetParameterUpperBound(para_feature[feature_index], 0, 2 / DEPTH_MAX_DIST);
				}
			}

			// build stereo reprojection residual
			if ((STEREO) && it_per_frame.is_stereo) {
				Eigen::Vector3d pts_j_right = it_per_frame.point_right;
				// if not same frame get reprojection residual between two camera in two frame
				if (imu_i != imu_j) {
					// build reprojection residual between last frame in left camera and current frame in right camera
					auto* f = new ProjectionTwoFrameTwoCamFactor(
					    pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity_right,
					    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
					// optimize parameters: pose in the first frame and current frame, extrinsic parameters, reverse
					// depth in feature point and td between camera and IMU
					problem.AddResidualBlock(f, visual_loss_function, para_pose[imu_i], para_pose[imu_j],
					                         para_ex_pose[0], para_ex_pose[1], para_feature[feature_index], para_td[0]);
				} else {
					// if same frame get reprojection residual between two camera in one frame
					auto* f = new ProjectionOneFrameTwoCamFactor(
					    pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity_right,
					    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
					// optimize parameters: extrinsic parameters for left and right camera, reverse depth in feature
					// point and td between camera and IMU
					problem.AddResidualBlock(f, visual_loss_function, para_ex_pose[0], para_ex_pose[1],
					                         para_feature[feature_index], para_td[0]);
				}
			}
		}
	}

	// 4. set ceres solver parameters and optimization
	ceres::Solver::Options options;

	if (!USE_GPU_ACC)
		options.linear_solver_type = ceres::DENSE_SCHUR;
	else
		options.dense_linear_algebra_library_type = ceres::CUDA;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.max_num_iterations = NUM_ITERATIONS;

	if (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD)
		options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
	else
		options.max_solver_time_in_seconds = SOLVER_TIME;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	LOGGER_DEBUG("Iterations : ", static_cast<int>(summary.iterations.size()));

	// update state parameters after optimize
	double2vector();
}

void Estimator::vector2double() {
	for (int i = 0; i <= WINDOW_SIZE; i++) {
		// rotation and translation
		para_pose[i][0] = Ps[i].x();
		para_pose[i][1] = Ps[i].y();
		para_pose[i][2] = Ps[i].z();
		Eigen::Quaterniond q{Rs[i]};
		para_pose[i][3] = q.x();
		para_pose[i][4] = q.y();
		para_pose[i][5] = q.z();
		para_pose[i][6] = q.w();

		if (USE_IMU) {
			// speed, acceleration bias and gyroscope bias
			para_speed_bias[i][0] = Vs[i].x();
			para_speed_bias[i][1] = Vs[i].y();
			para_speed_bias[i][2] = Vs[i].z();

			para_speed_bias[i][3] = Bas[i].x();
			para_speed_bias[i][4] = Bas[i].y();
			para_speed_bias[i][5] = Bas[i].z();

			para_speed_bias[i][6] = Bgs[i].x();
			para_speed_bias[i][7] = Bgs[i].y();
			para_speed_bias[i][8] = Bgs[i].z();
		}
	}

	for (int i = 0; i < NUM_OF_CAM; i++) {
		// extrinsic parameters
		para_ex_pose[i][0] = tic[i].x();
		para_ex_pose[i][1] = tic[i].y();
		para_ex_pose[i][2] = tic[i].z();
		Eigen::Quaterniond q{ric[i]};
		para_ex_pose[i][3] = q.x();
		para_ex_pose[i][4] = q.y();
		para_ex_pose[i][5] = q.z();
		para_ex_pose[i][6] = q.w();
	}

	// reverse depth of feature points in the current frame, restrict only observe frame number is greater than 4
	auto deps = feature_manager.getDepthVector();
	param_feature_id.clear();
	LOGGER_INFO("Solve features: ", deps.size());
	for (auto& it : deps) {
		para_feature[param_feature_id.size()][0] = it.second;
		param_feature_id_to_index[it.first] = param_feature_id.size();
		param_feature_id.push_back(it.first);
	}

	// healthy check weight adjust
	visual_residual_weight = feature_manager.stable_scale;
	if (USE_IMU) {
		if (USE_MAGNETOMETER)
			imu_residual_weight = 3 - mag_residual_weight - visual_residual_weight;
		else
			imu_residual_weight = 2 - visual_residual_weight;
	}
	// } else {
	// 	visual_residual_weight = 1.0;
	// }
	LOGGER_INFO("visual residual scale: ", visual_residual_weight);
	LOGGER_INFO("imu residual scale: ", imu_residual_weight);
	LOGGER_INFO("mag residual scale: ", mag_residual_weight);

	// time different between camera and IMU
	para_td[0][0] = td;
}

void Estimator::double2vector() {
	// get pose in the first frame before optimization
	Eigen::Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
	Eigen::Vector3d origin_P0 = Ps[0];

	// if last optimization is failed, clear Rs and Ps and replace with last_R0 and last_P0
	if (failure_occur_flag) {
		origin_R0 = Utility::R2ypr(last_R0);
		origin_P0 = last_P0;
		failure_occur_flag = false;
	}

	// pose in the first frame will be changed because of unfixed
	if (USE_IMU) {
		// pose in the first frame after optimization
		Eigen::Vector3d origin_R00 = Utility::R2ypr(
		    Eigen::Quaterniond(para_pose[0][6], para_pose[0][3], para_pose[0][4], para_pose[0][5]).toRotationMatrix());
		// yaw difference
		double y_diff = origin_R0.x() - origin_R00.x();
		// rotation matrix correspond to yaw difference
		Eigen::Matrix3d rot_diff = Utility::ypr2R(Eigen::Vector3d(y_diff, 0, 0));

		// if pitch is close to 90°
		if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
			LOGGER_DEBUG("euler singular point!");
			// compute rotation
			rot_diff = Rs[0] * Eigen::Quaterniond(para_pose[0][6], para_pose[0][3], para_pose[0][4], para_pose[0][5])
			                       .toRotationMatrix()
			                       .transpose();
		}

		// traverse sliding window and update state parameters after optimization
		for (int i = 0; i <= WINDOW_SIZE; i++) {
			Rs[i] = rot_diff * Eigen::Quaterniond(para_pose[i][6], para_pose[i][3], para_pose[i][4], para_pose[i][5])
			                       .normalized()
			                       .toRotationMatrix();

			Ps[i] = rot_diff * Eigen::Vector3d(para_pose[i][0] - para_pose[0][0], para_pose[i][1] - para_pose[0][1],
			                                   para_pose[i][2] - para_pose[0][2]) +
			        origin_P0;

			Vs[i] = rot_diff * Eigen::Vector3d(para_speed_bias[i][0], para_speed_bias[i][1], para_speed_bias[i][2]);

			Bas[i] = Eigen::Vector3d(para_speed_bias[i][3], para_speed_bias[i][4], para_speed_bias[i][5]);

			Bgs[i] = Eigen::Vector3d(para_speed_bias[i][6], para_speed_bias[i][7], para_speed_bias[i][8]);
		}

		LOGGER_DEBUG("IMU double2vector finish");

	}
	// if IMU is not using, set the first frame fixed and set values for parameters directly
	else {
		for (int i = 0; i <= WINDOW_SIZE; i++) {
			Rs[i] = Eigen::Quaterniond(para_pose[i][6], para_pose[i][3], para_pose[i][4], para_pose[i][5])
			            .normalized()
			            .toRotationMatrix();

			Ps[i] = Eigen::Vector3d(para_pose[i][0], para_pose[i][1], para_pose[i][2]);
		}
	}

	// update extrinsic parameters
	if (USE_IMU) {
		for (int i = 0; i < NUM_OF_CAM; i++) {
			tic[i] = Eigen::Vector3d(para_ex_pose[i][0], para_ex_pose[i][1], para_ex_pose[i][2]);
			ric[i] = Eigen::Quaterniond(para_ex_pose[i][6], para_ex_pose[i][3], para_ex_pose[i][4], para_ex_pose[i][5])
			             .normalized()
			             .toRotationMatrix();
		}
	}

	// update reverse depth
	std::map<int, double> deps;
	for (unsigned int i = 0; i < param_feature_id.size(); i++) {
		int _id = param_feature_id[i];
		// ROS_INFO("Id %d depth %f", i, 1/para_Feature[i][0]);
		deps[_id] = para_feature[i][0];
	}

	feature_manager.setDepth(deps);

	// update time difference between camera and IMU
	if (USE_IMU && ESTIMATE_TD)
		td = para_td[0][0];
}

bool Estimator::failureDetection() {
	// feature point which tracking long time almost lost all
	if (feature_manager.last_track_num < 2) {
		LOGGER_WARN(" little feature ", feature_manager.last_track_num);
		// return true;
	}
	// acceleration bias is too big
	if (Bas[WINDOW_SIZE].norm() > 2.5) {
		LOGGER_WARN(" big IMU acc bias estimation ", Bas[WINDOW_SIZE].norm());
		return true;
	}
	// gyroscope bias is too big
	if (Bgs[WINDOW_SIZE].norm() > 1.0) {
		LOGGER_WARN(" big IMU gyr bias estimation ", Bgs[WINDOW_SIZE].norm());
		return true;
	}
	// pose difference is to big after optimization
	Eigen::Vector3d tmp_P = Ps[WINDOW_SIZE];
	if ((tmp_P - last_P).norm() > 5) {
		LOGGER_WARN(" big translation");
		return true;
	}
	if (abs(tmp_P.z() - last_P.z()) > 1) {
		LOGGER_WARN(" big z translation");
		return true;
	}
	// angle difference is to big after optimization
	Eigen::Matrix3d tmp_R = Rs[WINDOW_SIZE];
	Eigen::Matrix3d delta_R = tmp_R.transpose() * last_R;
	Eigen::Quaterniond delta_Q(delta_R);
	double delta_angle;
	delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
	if (delta_angle > 50) {
		LOGGER_WARN(" big delta_angle ");
		// return true;
	}
	return false;
}

void Estimator::margOld(ceres::LossFunction* loss_function) {
	auto* marginalization_info = new MarginalizationInfo();
	// 1. convert state parameters in sliding window to array for marginalization
	vector2double();

	// 2. add priority residual to block
	if (last_marginalization_info && last_marginalization_info->valid) {
		std::vector<int> drop_set;
		// parameter block from last marginalization
		for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
			if (last_marginalization_parameter_blocks[i] == para_pose[0] ||
			    last_marginalization_parameter_blocks[i] == para_speed_bias[0])
				drop_set.push_back(i);
		}
		// construct new marginalization_factor
		auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
		auto* residual_block_info =
		    new ResidualBlockInfo(marginalization_factor, nullptr, last_marginalization_parameter_blocks, drop_set);
		marginalization_info->addResidualBlockInfo(residual_block_info);
	}

	// 3. add IMU residual between the first frame in sliding window and the second frame to block
	if (USE_IMU) {
		if (pre_integrations[1]->sum_dt < 10.0) {
			auto* imu_factor = new IMUFactor(pre_integrations[1]);
			auto* residual_block_info = new ResidualBlockInfo(
			    imu_factor, nullptr,
			    std::vector<double*>{para_pose[0], para_speed_bias[0], para_pose[1], para_speed_bias[1]},
			    std::vector<int>{0, 1});
			marginalization_info->addResidualBlockInfo(residual_block_info);
		}
	}

	// 4. add visual reprojection residual between the first frame in sliding window and other frames to block

	// traverse feature point

	for (int _id : param_feature_id) {
		auto& it_per_id = feature_manager.feature[_id];

		int feature_index = param_feature_id_to_index[it_per_id.feature_id];
		int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
		if (imu_i != 0)
			continue;

		// get normalize camera point in the first frame
		Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

		// traverse observe frame and get residual to block
		for (auto& it_per_frame : it_per_id.feature_per_frame) {
			imu_j++;
			// if not the first frame
			if (imu_i != imu_j) {
				Eigen::Vector3d pts_j = it_per_frame.point;
				auto* f_td = new ProjectionTwoFrameOneCamFactor(
				    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
				    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
				auto* residual_block_info =
				    new ResidualBlockInfo(f_td, loss_function,
				                          std::vector<double*>{para_pose[imu_i], para_pose[imu_j], para_ex_pose[0],
				                                               para_feature[feature_index], para_td[0]},
				                          std::vector<int>{0, 3});
				marginalization_info->addResidualBlockInfo(residual_block_info);
			}
			if ((STEREO) && it_per_frame.is_stereo) {
				Eigen::Vector3d pts_j_right = it_per_frame.point_right;
				if (imu_i != imu_j) {
					auto* f = new ProjectionTwoFrameTwoCamFactor(
					    pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity_right,
					    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
					auto* residual_block_info = new ResidualBlockInfo(
					    f, loss_function,
					    std::vector<double*>{para_pose[imu_i], para_pose[imu_j], para_ex_pose[0], para_ex_pose[1],
					                         para_feature[feature_index], para_td[0]},
					    std::vector<int>{0, 4});
					marginalization_info->addResidualBlockInfo(residual_block_info);
				} else {
					auto* f = new ProjectionOneFrameTwoCamFactor(
					    pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity_right,
					    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
					auto* residual_block_info = new ResidualBlockInfo(
					    f, loss_function,
					    std::vector<double*>{para_ex_pose[0], para_ex_pose[1], para_feature[feature_index], para_td[0]},
					    std::vector<int>{2});
					marginalization_info->addResidualBlockInfo(residual_block_info);
				}
			}
		}
	}

	// 5. pre marginalization and marginlization
	marginalization_info->preMarginalize();

	marginalization_info->marginalize();

	// 6. set parameter array value to the value before it after marginalization and record to addr_shift
	std::unordered_map<long, double*> addr_shift;
	for (auto i = 1; i <= WINDOW_SIZE; i++) {
		addr_shift[reinterpret_cast<long>(para_pose[i])] = para_pose[i - 1];
		if (USE_IMU)
			addr_shift[reinterpret_cast<long>(para_speed_bias[i])] = para_speed_bias[i - 1];
	}
	for (auto i = 0; i < NUM_OF_CAM; i++)
		addr_shift[reinterpret_cast<long>(para_ex_pose[i])] = para_ex_pose[i];
	addr_shift[reinterpret_cast<long>(para_td[0])] = para_td[0];

	// 7. get parameter blocks for marginalization
	std::vector<double*> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

	delete last_marginalization_info;
	// 8. save marginalize information
	last_marginalization_info = marginalization_info;
	last_marginalization_parameter_blocks = parameter_blocks;
}

void Estimator::margNew() {
	// if last marginalization info in not empty and the count for parameter block is not equal to zero
	if (last_marginalization_info && count(begin(last_marginalization_parameter_blocks),
	                                       end(last_marginalization_parameter_blocks), para_pose[WINDOW_SIZE - 1])) {
		auto* marginalization_info = new MarginalizationInfo();
		// 1. convert state parameters in sliding window to array for marginalization
		vector2double();

		// 2. add priority residual to block
		if (last_marginalization_info && last_marginalization_info->valid) {
			std::vector<int> drop_set;
			for (auto i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
				assert(last_marginalization_parameter_blocks[i] != para_speed_bias[WINDOW_SIZE - 1]);
				if (last_marginalization_parameter_blocks[i] == para_pose[WINDOW_SIZE - 1])
					drop_set.push_back(i);
			}
			// construct new marginalization_factor
			auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
			auto* residual_block_info =
			    new ResidualBlockInfo(marginalization_factor, nullptr, last_marginalization_parameter_blocks, drop_set);

			marginalization_info->addResidualBlockInfo(residual_block_info);
		}
		// 3. pre marginalization and marginlization
		marginalization_info->preMarginalize();
		marginalization_info->marginalize();

		// 4. set parameter array value to the value before it after marginalization and record to addr_shift
		std::unordered_map<long, double*> addr_shift;
		for (auto i = 0; i <= WINDOW_SIZE; i++) {
			// this frame is marginalized
			if (i == WINDOW_SIZE - 1)
				continue;
			// this frame is the current frame
			else if (i == WINDOW_SIZE) {
				addr_shift[reinterpret_cast<long>(para_pose[i])] = para_pose[i - 1];
				if (USE_IMU)
					addr_shift[reinterpret_cast<long>(para_speed_bias[i])] = para_speed_bias[i - 1];
			} else {
				addr_shift[reinterpret_cast<long>(para_pose[i])] = para_pose[i];
				if (USE_IMU)
					addr_shift[reinterpret_cast<long>(para_speed_bias[i])] = para_speed_bias[i];
			}
		}
		for (int i = 0; i < NUM_OF_CAM; i++)
			addr_shift[reinterpret_cast<long>(para_ex_pose[i])] = para_ex_pose[i];
		addr_shift[reinterpret_cast<long>(para_td[0])] = para_td[0];
		// 5. get parameter blocks for marginalization
		std::vector<double*> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
		// 6. save marginalize information
		delete last_marginalization_info;
		last_marginalization_info = marginalization_info;
		last_marginalization_parameter_blocks = parameter_blocks;
	}
}

void Estimator::slideWindow() {
	if (marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD) {
		// sliding window by delete the first frame

		double t_0 = headers[0];
		// backup Rs and Ps
		back_R0 = Rs[0];
		back_P0 = Ps[0];
		if (frame_count == WINDOW_SIZE) {
			// delete the first frame and move all the data forward
			for (auto i = 0; i < WINDOW_SIZE; i++) {
				headers[i] = headers[i + 1];
				Rs[i].swap(Rs[i + 1]);
				Ps[i].swap(Ps[i + 1]);
				if (USE_IMU) {
					std::swap(pre_integrations[i], pre_integrations[i + 1]);

					dt_buf[i].swap(dt_buf[i + 1]);
					linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
					angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

					Vs[i].swap(Vs[i + 1]);
					Bas[i].swap(Bas[i + 1]);
					Bgs[i].swap(Bgs[i + 1]);
				}
			}
			// add the current frame in sliding window and initialize with the last data
			headers[WINDOW_SIZE] = headers[WINDOW_SIZE - 1];
			Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
			Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

			if (USE_IMU) {
				Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
				Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
				Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

				delete pre_integrations[WINDOW_SIZE];
				pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

				dt_buf[WINDOW_SIZE].clear();
				linear_acceleration_buf[WINDOW_SIZE].clear();
				angular_velocity_buf[WINDOW_SIZE].clear();
			}
			// delete the first image frame
			std::map<double, ImageFrame>::iterator it_0;
			it_0 = all_image_frame.find(t_0);
			delete it_0->second.pre_integration;
			all_image_frame.erase(all_image_frame.begin(), it_0);
			// move sliding window and delete frame from set of observe feature point frame, then calculate depth of
			// the new first frame
			slideWindowOld();
		}
	} else {
		// sliding window by delete the second new frame

		if (frame_count == WINDOW_SIZE) {
			// replace the second new frame with the current frame, save the current frame
			headers[frame_count - 1] = headers[frame_count];
			Ps[frame_count - 1] = Ps[frame_count];
			Rs[frame_count - 1] = Rs[frame_count];

			if (USE_IMU) {
				// IMU data in the current frame
				for (auto i = 0; i < dt_buf[frame_count].size(); i++) {
					double tmp_dt = dt_buf[frame_count][i];
					Eigen::Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
					Eigen::Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
					pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
					dt_buf[frame_count - 1].push_back(tmp_dt);
					linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
					angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
				}

				Vs[frame_count - 1] = Vs[frame_count];
				Bas[frame_count - 1] = Bas[frame_count];
				Bgs[frame_count - 1] = Bgs[frame_count];

				delete pre_integrations[WINDOW_SIZE];
				pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

				dt_buf[WINDOW_SIZE].clear();
				linear_acceleration_buf[WINDOW_SIZE].clear();
				angular_velocity_buf[WINDOW_SIZE].clear();
			}
			// delete the second new frame after marginalization, and delete feature point which have not any
			// observe frame
			slideWindowNew();
		}
	}
}

void Estimator::slideWindowNew() {
	// slide window second new and delete feature point which have not any observe frame
	feature_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld() {
	bool shift_depth = (solver_flag == NON_LINEAR);
	// if the vio system has been initialized
	if (shift_depth) {
		Eigen::Matrix3d R0, R1;
		Eigen::Vector3d P0, P1;
		// get Rwc and Twc from marginalize frame(not the first frame, should be the frame before the first frame)
		R0 = back_R0 * ric[0];
		P0 = back_P0 + back_R0 * tic[0];
		// get Rwc and Twc from  the first frame
		R1 = Rs[0] * ric[0];
		P1 = Ps[0] + Rs[0] * tic[0];

		// after marginalizing the first frame, delete the frame from the observation frame set of the feature
		// point, and the index of the observation frame is all - 1. if the feature point has no observation frame
		// and less than 2 frames, delete the estimated_depth value bound to this feature point and the first frame,
		// and recalculate
		feature_manager.removeBackShiftDepth(R0, P0, R1, P1);
	} else
		// after marginalizing the first frame, delete the frame from the observation frame set of the feature
		// point, and the index of the observation frame is all -1. if there is no observation frame for the feature
		// point, delete the feature point
		feature_manager.removeBack();
}

double Estimator::reprojectionError(Eigen::Matrix3d& Ri, Eigen::Vector3d& Pi, Eigen::Matrix3d& rici,
                                    Eigen::Vector3d& tici, Eigen::Matrix3d& Rj, Eigen::Vector3d& Pj,
                                    Eigen::Matrix3d& ricj, Eigen::Vector3d& ticj, double depth, Eigen::Vector3d& uvi,
                                    Eigen::Vector3d& uvj) {
	// world point correspond to point i
	Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
	// convert point world to camera point j
	Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
	// calculate error in normalize camera coordinate j
	Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
	double rx = residual.x();
	double ry = residual.y();
	return sqrt(rx * rx + ry * ry);
}

double Estimator::reprojectionError3D(Eigen::Matrix3d& Ri, Eigen::Vector3d& Pi, Eigen::Matrix3d& rici,
                                      Eigen::Vector3d& tici, Eigen::Matrix3d& Rj, Eigen::Vector3d& Pj,
                                      Eigen::Matrix3d& ricj, Eigen::Vector3d& ticj, double depth, Eigen::Vector3d& uvi,
                                      Eigen::Vector3d& uvj) {
	// world point correspond to point i
	Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
	// convert point world to camera point j
	Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
	// calculate error in normalize camera coordinate j with depth
	return (pts_cj - uvj).norm() / depth;
}

void Estimator::movingConsistencyCheck(std::set<int>& remove_index) {
	auto pts_status = feature_manager.ft.getFeatureStatus();
	// traverse feature point
	for (auto& it : feature_manager.feature) {
		auto& it_per_id = it.second;
		// check frame number and depth
		it_per_id.used_num = it_per_id.feature_per_frame.size();
		if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
			continue;

		double depth = it_per_id.estimated_depth;
		if (depth < 0)
			continue;

		double err = 0;
		double err3D = 0;
		int errCnt = 0;
		int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
		Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
		// traverse frame from the set of observe frame
		for (auto& it_per_frame : it_per_id.feature_per_frame) {
			imu_j++;
			// calculate reprojection error from two frames
			if (imu_i != imu_j) {
				Eigen::Vector3d pts_j = it_per_frame.point;
				err += reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], tic[0],
				                         depth, pts_i, pts_j);
				// only for depth camera
				if (DEPTH && it_per_frame.is_depth)
					err3D += reprojectionError3D(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0],
					                             tic[0], depth, pts_i, pts_j);
				errCnt++;
			}
			// only for binocular
			if (STEREO && it_per_frame.is_stereo) {
				Eigen::Vector3d pts_j_right = it_per_frame.point_right;

				if (imu_i != imu_j) {
					// calculate reprojection error between frame 1 in left camera and frame 2 in right camera
					double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
					                                     ric[1], tic[1], depth, pts_i, pts_j_right);
					err += tmp_error;
					errCnt++;
				} else {
					// calculate reprojection error in same frame from left camera and right camera
					double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
					                                     ric[1], tic[1], depth, pts_i, pts_j_right);
					err += tmp_error;
					errCnt++;
				}
			}
		}
		if (errCnt > 0) {
			// if feature point error is greater than threshold, add it to remove index and set it to dynamic
			if (FOCAL_LENGTH * err / errCnt > 3 || err3D / errCnt > 2.0) {
				remove_index.insert(it_per_id.feature_id);
			}
		}
	}
}

} // namespace FLOW_VINS