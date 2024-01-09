#include "interface.h"
#include "include/estimator.h"

namespace FLOW_VINS {

// vio estimator
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;

CameraPoseVisualization estimator_camera_pose_visual(1, 0, 0, 1);

void registerEstimatorPub(rclcpp::Node::SharedPtr n) {

	pub_path = n->create_publisher<nav_msgs::msg::Path>("path", 1000);
	pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("odometry", 1000);
	pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 1000);
	pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud", 1000);
	pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("key_poses", 1000);
	pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 1000);
	pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
	pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("keyframe_pose", 1000);
	pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("keyframe_point", 1000);
	pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 1000);
	pub_image_track = n->create_publisher<sensor_msgs::msg::Image>("image_track", 1000);

	estimator_camera_pose_visual.setScale(0.1);
	estimator_camera_pose_visual.setLineWidth(0.01);
}

void printStatistics(const Estimator& estimator, double t) {
	static double sum_of_path = 0;
	static Eigen::Vector3d last_path(0.0, 0.0, 0.0);

	if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
		return;
	LOGGER_INFO("position: ", estimator.Ps[WINDOW_SIZE].transpose());
	LOGGER_INFO("orientation: ", estimator.Vs[WINDOW_SIZE].transpose());
	if (ESTIMATE_EXTRINSIC) {
		cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
		for (int i = 0; i < NUM_OF_CAM; i++) {
			LOGGER_INFO("calibration result for camera %d", i);

			LOGGER_INFO("extirnsic tic: ", estimator.tic[i].transpose());
			LOGGER_INFO("extrinsic ric: ", Utility::R2ypr(estimator.ric[i]).transpose());

			Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
			eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
			eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
			cv::Mat cv_T;
			cv::eigen2cv(eigen_T, cv_T);
			if (i == 0)
				fs << "body_T_cam0" << cv_T;
			else
				fs << "body_T_cam1" << cv_T;
		}
		fs.release();
	}
	// compute average solver cost
	static double sum_of_time = 0;
	static int sum_of_calculation = 0;
	sum_of_time += t;
	sum_of_calculation++;
	LOGGER_INFO("vo solver costs: ", t, " ms");
	LOGGER_INFO("average of time ", sum_of_time / sum_of_calculation, "ms");

	// compute sum of path
	sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
	last_path = estimator.Ps[WINDOW_SIZE];
	LOGGER_INFO("sum of path ", sum_of_path);
	if (ESTIMATE_TD)
		LOGGER_INFO("td ", estimator.td);
}

void pubTrackImage(const cv::Mat& imgTrack, const double t) {
	std_msgs::msg::Header header;
	header.frame_id = "world";
	int sec_ts = (int)t;
	uint nsec_ts = (uint)((t - sec_ts) * 1e9);
	header.stamp.sec = sec_ts;
	header.stamp.nanosec = nsec_ts;

	sensor_msgs::msg::Image::SharedPtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
	pub_image_track->publish(*imgTrackMsg);
}

void pubOdometry(const Estimator& estimator, const std_msgs::msg::Header& header) {
	if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
		nav_msgs::msg::Odometry odometry;
		odometry.header = header;
		odometry.header.frame_id = "world";
		odometry.child_frame_id = "world";
		Eigen::Quaterniond tmp_Q;
		tmp_Q = Eigen::Quaterniond(estimator.Rs[WINDOW_SIZE]);
		odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
		odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
		odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
		odometry.pose.pose.orientation.x = tmp_Q.x();
		odometry.pose.pose.orientation.y = tmp_Q.y();
		odometry.pose.pose.orientation.z = tmp_Q.z();
		odometry.pose.pose.orientation.w = tmp_Q.w();
		odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
		odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
		odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
		pub_odometry->publish(odometry);

		geometry_msgs::msg::PoseStamped pose_stamped;
		pose_stamped.header = header;
		pose_stamped.header.frame_id = "world";
		pose_stamped.pose = odometry.pose.pose;
		path.header = header;
		path.header.frame_id = "world";
		path.poses.push_back(pose_stamped);
		pub_path->publish(path);

		// write result to file
		double time_stamp = header.stamp.sec + header.stamp.nanosec * (1e-9);
		std::ofstream foutC(VINS_RESULT_PATH, std::ios::app);
		foutC.setf(std::ios::fixed, std::ios::floatfield);
		foutC << time_stamp << " ";
		foutC << estimator.Ps[WINDOW_SIZE].x() << " " << estimator.Ps[WINDOW_SIZE].y() << " "
		      << estimator.Ps[WINDOW_SIZE].z() << " " << tmp_Q.x() << " " << tmp_Q.y() << " " << tmp_Q.z() << " "
		      << tmp_Q.w() << std::endl;
		foutC.close();
	}
}

void pubKeyPoses(const Estimator& estimator, const std_msgs::msg::Header& header) {
	if (estimator.key_poses.empty())
		return;
	visualization_msgs::msg::Marker key_poses;
	key_poses.header = header;
	key_poses.header.frame_id = "world";
	key_poses.ns = "key_poses";
	key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
	key_poses.action = visualization_msgs::msg::Marker::ADD;
	key_poses.pose.orientation.w = 1.0;
	key_poses.lifetime = rclcpp::Duration(std::chrono::nanoseconds(0));

	key_poses.id = 0;
	key_poses.scale.x = 0.05;
	key_poses.scale.y = 0.05;
	key_poses.scale.z = 0.05;
	key_poses.color.r = 1.0;
	key_poses.color.a = 1.0;

	for (int i = 0; i <= WINDOW_SIZE; i++) {
		geometry_msgs::msg::Point pose_marker;
		Eigen::Vector3d correct_pose;
		correct_pose = estimator.key_poses[i];
		pose_marker.x = correct_pose.x();
		pose_marker.y = correct_pose.y();
		pose_marker.z = correct_pose.z();
		key_poses.points.push_back(pose_marker);
	}
	pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator& estimator, const std_msgs::msg::Header& header) {
	int idx2 = WINDOW_SIZE - 1;

	if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
		int i = idx2;
		Eigen::Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
		Eigen::Quaterniond R = Eigen::Quaterniond(estimator.Rs[i] * estimator.ric[0]);

		nav_msgs::msg::Odometry odometry;
		odometry.header = header;
		odometry.header.frame_id = "world";
		odometry.pose.pose.position.x = P.x();
		odometry.pose.pose.position.y = P.y();
		odometry.pose.pose.position.z = P.z();
		odometry.pose.pose.orientation.x = R.x();
		odometry.pose.pose.orientation.y = R.y();
		odometry.pose.pose.orientation.z = R.z();
		odometry.pose.pose.orientation.w = R.w();

		pub_camera_pose->publish(odometry);

		estimator_camera_pose_visual.reset();
		estimator_camera_pose_visual.add_pose(P, R);
		if (STEREO) {
			Eigen::Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
			Eigen::Quaterniond R = Eigen::Quaterniond(estimator.Rs[i] * estimator.ric[1]);
			estimator_camera_pose_visual.add_pose(P, R);
		}
		estimator_camera_pose_visual.publish_by(pub_camera_pose_visual, odometry.header);
	}
}

void pubPointCloud(const Estimator& estimator, const std_msgs::msg::Header& header) {
	sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
	point_cloud.header = header;
	loop_point_cloud.header = header;

	for (auto& it : estimator.feature_manager.feature) {
		auto it_per_id = it.second;
		int used_num;

		used_num = static_cast<int>(it_per_id.feature_per_frame.size());
		if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
			continue;
		if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
			continue;
		int imu_i = it_per_id.start_frame;
		Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
		Eigen::Vector3d w_pts_i =
		    estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

		geometry_msgs::msg::Point32 p;
		p.x = static_cast<float>(w_pts_i(0));
		p.y = static_cast<float>(w_pts_i(1));
		p.z = static_cast<float>(w_pts_i(2));
		point_cloud.points.push_back(p);
	}
	pub_point_cloud->publish(point_cloud);

	// pub margined point
	sensor_msgs::msg::PointCloud margin_cloud;
	margin_cloud.header = header;

	for (auto& it : estimator.feature_manager.feature) {
		auto it_per_id = it.second;
		int used_num;

		used_num = static_cast<int>(it_per_id.feature_per_frame.size());
		if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
			continue;

		if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 && it_per_id.solve_flag == 1) {
			int imu_i = it_per_id.start_frame;
			Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
			Eigen::Vector3d w_pts_i =
			    estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

			geometry_msgs::msg::Point32 p;
			p.x = static_cast<float>(w_pts_i(0));
			p.y = static_cast<float>(w_pts_i(1));
			p.z = static_cast<float>(w_pts_i(2));
			margin_cloud.points.push_back(p);
		}
	}
	pub_margin_cloud->publish(margin_cloud);
}

void pubTF(const Estimator& estimator, const std_msgs::msg::Header& header) {

	if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
		return;

	std::shared_ptr<tf2_ros::TransformBroadcaster> br;
	geometry_msgs::msg::TransformStamped transform, transform_cam;

	tf2::Quaternion q;
	// body frame
	Eigen::Vector3d correct_t;
	Eigen::Quaterniond correct_q;

	correct_t = estimator.Ps[WINDOW_SIZE];
	correct_q = estimator.Rs[WINDOW_SIZE];

	// cout << header.stamp.sec + header.stamp.nanosec * (1e-9) << endl;
	// cout << correct_t << endl;
	// cout << correct_q.w() << " " << correct_q.x() << " " << correct_q.y() << " " << correct_q.z() << endl;

	// transform.header.stamp = header.stamp;
	transform.header.frame_id = "world";
	transform.child_frame_id = "body";

	transform.transform.translation.x = correct_t(0);
	transform.transform.translation.y = correct_t(1);
	transform.transform.translation.z = correct_t(2);

	q.setW(correct_q.w());
	q.setX(correct_q.x());
	q.setY(correct_q.y());
	q.setZ(correct_q.z());
	transform.transform.rotation.x = q.x();
	transform.transform.rotation.y = q.y();
	transform.transform.rotation.z = q.z();
	transform.transform.rotation.w = q.w();

	// br->sendTransform(transform);

	// camera frame
	transform_cam.header.stamp = header.stamp;
	transform_cam.header.frame_id = "body";
	transform_cam.child_frame_id = "camera";

	transform_cam.transform.translation.x = estimator.tic[0].x();
	transform_cam.transform.translation.y = estimator.tic[0].y();
	transform_cam.transform.translation.z = estimator.tic[0].z();

	q.setW(Eigen::Quaterniond(estimator.ric[0]).w());
	q.setX(Eigen::Quaterniond(estimator.ric[0]).x());
	q.setY(Eigen::Quaterniond(estimator.ric[0]).y());
	q.setZ(Eigen::Quaterniond(estimator.ric[0]).z());

	transform_cam.transform.rotation.x = q.x();
	transform_cam.transform.rotation.y = q.y();
	transform_cam.transform.rotation.z = q.z();
	transform_cam.transform.rotation.w = q.w();

	// br->sendTransform(transform_cam);

	nav_msgs::msg::Odometry odometry;
	odometry.header = header;
	odometry.header.frame_id = "world";
	odometry.pose.pose.position.x = estimator.tic[0].x();
	odometry.pose.pose.position.y = estimator.tic[0].y();
	odometry.pose.pose.position.z = estimator.tic[0].z();
	Eigen::Quaterniond tmp_q{estimator.ric[0]};
	odometry.pose.pose.orientation.x = tmp_q.x();
	odometry.pose.pose.orientation.y = tmp_q.y();
	odometry.pose.pose.orientation.z = tmp_q.z();
	odometry.pose.pose.orientation.w = tmp_q.w();

	pub_extrinsic->publish(odometry);
}

void pubKeyframe(const Estimator& estimator) {
	// pub camera pose, 2D-3D points of keyframe
	if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0) {
		int i = WINDOW_SIZE - 2;
		Eigen::Vector3d P = estimator.Ps[i];
		Eigen::Quaterniond R = Eigen::Quaterniond(estimator.Rs[i]);

		nav_msgs::msg::Odometry odometry;
		int sec_ts = (int)estimator.headers[WINDOW_SIZE - 2];
		uint nsec_ts = (uint)((estimator.headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
		odometry.header.stamp.sec = sec_ts;
		odometry.header.stamp.nanosec = nsec_ts;

		odometry.header.frame_id = "world";
		odometry.pose.pose.position.x = P.x();
		odometry.pose.pose.position.y = P.y();
		odometry.pose.pose.position.z = P.z();
		odometry.pose.pose.orientation.x = R.x();
		odometry.pose.pose.orientation.y = R.y();
		odometry.pose.pose.orientation.z = R.z();
		odometry.pose.pose.orientation.w = R.w();

		pub_keyframe_pose->publish(odometry);

		sensor_msgs::msg::PointCloud point_cloud;
		sec_ts = (int)estimator.headers[WINDOW_SIZE - 2];
		nsec_ts = (uint)((estimator.headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
		point_cloud.header.stamp.sec = sec_ts;
		point_cloud.header.stamp.nanosec = nsec_ts;
		point_cloud.header.frame_id = "world";
		for (auto& it : estimator.feature_manager.feature) {
			auto it_per_id = it.second;
			int frame_size = static_cast<int>(it_per_id.feature_per_frame.size());
			if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 &&
			    it_per_id.solve_flag == 1) {
				int imu_i = it_per_id.start_frame;
				Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
				Eigen::Vector3d w_pts_i =
				    estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
				geometry_msgs::msg::Point32 p;
				p.x = static_cast<float>(w_pts_i(0));
				p.y = static_cast<float>(w_pts_i(1));
				p.z = static_cast<float>(w_pts_i(2));
				point_cloud.points.push_back(p);

				int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
				sensor_msgs::msg::ChannelFloat32 p_2d;
				p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].point.x()));
				p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].point.y()));
				p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].uv.x()));
				p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].uv.y()));
				p_2d.values.push_back(static_cast<float>(it_per_id.feature_id));
				point_cloud.channels.push_back(p_2d);
			}
		}
		pub_keyframe_point->publish(point_cloud);
	}
}

} // namespace FLOW_VINS