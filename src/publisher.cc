/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: ROS topic
 */

#include "../include/publisher.h"
#include "../include/estimator.h"

namespace FLOW_VINS {

const Vector3d CameraPoseVisualization::imlt = Vector3d(-1.0, -0.5, 1.0);
const Vector3d CameraPoseVisualization::imrt = Vector3d(1.0, -0.5, 1.0);
const Vector3d CameraPoseVisualization::imlb = Vector3d(-1.0, 0.5, 1.0);
const Vector3d CameraPoseVisualization::imrb = Vector3d(1.0, 0.5, 1.0);
const Vector3d CameraPoseVisualization::lt0 = Vector3d(-0.7, -0.5, 1.0);
const Vector3d CameraPoseVisualization::lt1 = Vector3d(-0.7, -0.2, 1.0);
const Vector3d CameraPoseVisualization::lt2 = Vector3d(-1.0, -0.2, 1.0);
const Vector3d CameraPoseVisualization::oc = Vector3d(0.0, 0.0, 0.0);

// vio estimator
static ros::Publisher pub_odometry;
static ros::Publisher pub_point_cloud, pub_margin_cloud;
static ros::Publisher pub_key_poses;
static ros::Publisher pub_keyframe_pose;
static ros::Publisher pub_keyframe_point;
static ros::Publisher pub_extrinsic;
static ros::Publisher pub_image_track;
static ros::Publisher pub_camera_pose_visual;
static ros::Publisher pub_path;
static ros::Publisher pub_semantic_image;
static ros::Publisher pub_semantic_mask;

static CameraPoseVisualization camera_pose_visual(1, 0, 0, 1);

void registerPub(ros::NodeHandle &n) {
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_path = n.advertise<nav_msgs::Path>("vio_path", 1000);
    pub_semantic_image = n.advertise<sensor_msgs::Image>("semantic_image", 1000);
    pub_semantic_mask = n.advertise<sensor_msgs::Image>("semantic_mask", 1000);

    camera_pose_visual.setScale(0.1);
    camera_pose_visual.setLineWidth(0.01);
}

void pubTrackImage(const cv::Mat &imgTrack, const double t) {
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track.publish(imgTrackMsg);
}

void printStatistics(const Estimator &estimator, double t) {
    static double sum_of_path = 0;
    static Vector3d last_path(0.0, 0.0, 0.0);
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    ROS_INFO_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_INFO_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC) {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++) {
            ROS_INFO("calibration result for camera %d", i);
            ROS_INFO_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            ROS_INFO_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Matrix4d eigen_T = Matrix4d::Identity();
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
    ROS_INFO("vo solver costs: %f ms", t);
    ROS_INFO("average of time %f ms", sum_of_time / sum_of_calculation);

    // compute sum of path
    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_INFO("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header) {
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.pose_graph.Rs);
        odometry.pose.pose.position.x = estimator.pose_graph.Ps.x();
        odometry.pose.pose.position.y = estimator.pose_graph.Ps.y();
        odometry.pose.pose.position.z = estimator.pose_graph.Ps.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry.publish(odometry);
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header) {
    if (estimator.key_poses.empty())
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    key_poses.id = 0;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++) {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header) {
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
        Vector3d vio_t_cam;
        Quaterniond vio_q_cam;
        vio_t_cam = estimator.pose_graph.Ps + estimator.pose_graph.Rs * LOOP_TIC;
        vio_q_cam = estimator.pose_graph.Rs * LOOP_QIC;

        camera_pose_visual.reset();
        camera_pose_visual.add_pose(vio_t_cam, vio_q_cam);
        camera_pose_visual.publish_by(pub_camera_pose_visual, header);
    }
}

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header) {
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;

    for (auto &it : estimator.feature_manager.feature) {
        auto it_per_id = it.second;
        int used_num;

        used_num = static_cast<int>(it_per_id.feature_per_frame.size());
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i =
            it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i =
            estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = static_cast<float>(w_pts_i(0));
        p.y = static_cast<float>(w_pts_i(1));
        p.z = static_cast<float>(w_pts_i(2));
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);

    // pub margined point
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it : estimator.feature_manager.feature) {
        auto it_per_id = it.second;
        int used_num;

        used_num = static_cast<int>(it_per_id.feature_per_frame.size());
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 && it_per_id.solve_flag == 1) {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
            Vector3d tmp = estimator.pose_graph.r_drift * w_pts_i + estimator.pose_graph.t_drift;

            geometry_msgs::Point32 p;
            p.x = static_cast<float>(tmp(0));
            p.y = static_cast<float>(tmp(1));
            p.z = static_cast<float>(tmp(2));
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}

void pubTF(const Estimator &estimator, const std_msgs::Header &header) {
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0), correct_t(1), correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(), estimator.tic[0].y(), estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);
}

void pubKeyframe(const Estimator &estimator) {
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0) {
        int i = WINDOW_SIZE - 2;
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.headers[WINDOW_SIZE - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_keyframe_pose.publish(odometry);

        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(estimator.headers[WINDOW_SIZE - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &it : estimator.feature_manager.feature) {
            auto it_per_id = it.second;
            int frame_size = static_cast<int>(it_per_id.feature_per_frame.size());
            if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1) {
                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
                // loop data convert
                Vector3d tmp = estimator.pose_graph.r_drift * w_pts_i + estimator.pose_graph.t_drift;

                geometry_msgs::Point32 p;
                p.x = static_cast<float>(tmp(0));
                p.y = static_cast<float>(tmp(1));
                p.z = static_cast<float>(tmp(2));
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].point.x()));
                p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].point.y()));
                p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].uv.x()));
                p_2d.values.push_back(static_cast<float>(it_per_id.feature_per_frame[imu_j].uv.y()));
                p_2d.values.push_back(static_cast<float>(it_per_id.feature_id));
                point_cloud.channels.push_back(p_2d);
            }
        }
        pub_keyframe_point.publish(point_cloud);
    }
}

void pubPoseGraph(const PoseGraph &pose_graph) {
    for (int i = 1; i <= pose_graph.sequence_cnt; i++) {
        pub_path.publish(pose_graph.path[i]);
    }
}

void pubSemanticImage(const cv::Mat &img, const std_msgs::Header &header) {
    sensor_msgs::ImagePtr SemanticMsg =
        cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
    pub_semantic_image.publish(SemanticMsg);
}

void pubSemanticMask(const cv::Mat &img, const std_msgs::Header &header) {
    sensor_msgs::ImagePtr SemanticMsg =
        cv_bridge::CvImage(header, "mono8", img).toImageMsg();
    pub_semantic_mask.publish(SemanticMsg);
}

//--------------------------------------------camera visualization--------------------------------------------//

static void Eigen2Point(const Vector3d &v, geometry_msgs::Point &p) {
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
}

CameraPoseVisualization::CameraPoseVisualization(float r, float g, float b,
                                                 float a) :
    marker_ns("CameraPoseVisualization"),
    scale(0.2), line_width(0.01) {
    image_boundary_color.r = r;
    image_boundary_color.g = g;
    image_boundary_color.b = b;
    image_boundary_color.a = a;
    optical_center_connector_color.r = r;
    optical_center_connector_color.g = g;
    optical_center_connector_color.b = b;
    optical_center_connector_color.a = a;
}

void CameraPoseVisualization::setScale(double s) {
    scale = s;
}

void CameraPoseVisualization::setLineWidth(double width) {
    line_width = width;
}

void CameraPoseVisualization::add_edge(const Vector3d &p0,
                                       const Vector3d &p1) {
    visualization_msgs::Marker marker;

    marker.ns = marker_ns;
    marker.id = static_cast<int>(markers.size()) + 1;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.005;

    marker.color.g = 1.0f;
    marker.color.a = 1.0;

    geometry_msgs::Point point0, point1;

    Eigen2Point(p0, point0);
    Eigen2Point(p1, point1);

    marker.points.push_back(point0);
    marker.points.push_back(point1);

    markers.push_back(marker);
}

void CameraPoseVisualization::add_loopedge(const Vector3d &p0,
                                           const Vector3d &p1) {
    visualization_msgs::Marker marker;

    marker.ns = marker_ns;
    marker.id = static_cast<int>(markers.size()) + 1;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.04;
    // marker.scale.x = 0.3;

    marker.color.r = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;

    geometry_msgs::Point point0, point1;

    Eigen2Point(p0, point0);
    Eigen2Point(p1, point1);

    marker.points.push_back(point0);
    marker.points.push_back(point1);

    markers.push_back(marker);
}

void CameraPoseVisualization::add_pose(const Vector3d &p,
                                       const Quaterniond &q) {
    visualization_msgs::Marker marker;

    marker.ns = marker_ns;
    marker.id = static_cast<int>(markers.size()) + 1;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = line_width;

    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1,
        pt_lt2;

    Eigen2Point(q * (scale * imlt) + p, pt_lt);
    Eigen2Point(q * (scale * imlb) + p, pt_lb);
    Eigen2Point(q * (scale * imrt) + p, pt_rt);
    Eigen2Point(q * (scale * imrb) + p, pt_rb);
    Eigen2Point(q * (scale * lt0) + p, pt_lt0);
    Eigen2Point(q * (scale * lt1) + p, pt_lt1);
    Eigen2Point(q * (scale * lt2) + p, pt_lt2);
    Eigen2Point(q * (scale * oc) + p, pt_oc);

    // image boundaries
    marker.points.push_back(pt_lt);
    marker.points.push_back(pt_lb);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    marker.points.push_back(pt_lb);
    marker.points.push_back(pt_rb);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    marker.points.push_back(pt_rb);
    marker.points.push_back(pt_rt);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    marker.points.push_back(pt_rt);
    marker.points.push_back(pt_lt);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    // top-left indicator
    marker.points.push_back(pt_lt0);
    marker.points.push_back(pt_lt1);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    marker.points.push_back(pt_lt1);
    marker.points.push_back(pt_lt2);
    marker.colors.push_back(image_boundary_color);
    marker.colors.push_back(image_boundary_color);

    // optical center connector
    marker.points.push_back(pt_lt);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(optical_center_connector_color);
    marker.colors.push_back(optical_center_connector_color);

    marker.points.push_back(pt_lb);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(optical_center_connector_color);
    marker.colors.push_back(optical_center_connector_color);

    marker.points.push_back(pt_rt);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(optical_center_connector_color);
    marker.colors.push_back(optical_center_connector_color);

    marker.points.push_back(pt_rb);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(optical_center_connector_color);
    marker.colors.push_back(optical_center_connector_color);

    markers.push_back(marker);
}

void CameraPoseVisualization::reset() {
    markers.clear();
}

void CameraPoseVisualization::publish_by(ros::Publisher &pub,
                                         const std_msgs::Header &header) {
    visualization_msgs::MarkerArray markerArray_msg;

    for (auto &marker : markers) {
        marker.header = header;
        markerArray_msg.markers.push_back(marker);
    }

    pub.publish(markerArray_msg);
}
} // namespace FLOW_VINS