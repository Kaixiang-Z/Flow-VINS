/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: ROS消息话题订阅
 */

#pragma once

#include "parameters.h"

#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>

namespace FLOW_VINS {

class Estimator;
class PoseGraph;
class CameraPoseVisualization;

/**
 * @brief: register vio estimator node publisher
 */
void registerPub(ros::NodeHandle &n);

/**
 * @brief: publish tracking image
 */
void pubTrackImage(const cv::Mat &imgTrack, double t);

/**
 * @brief: print state data and compute average process cost
 */
void printStatistics(const Estimator &estimator, double t);

/**
 * @brief: publish odometry & path message and write files
 */
void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

/**
 * @brief: publish key poses maker (only in window size)
 */
void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

/**
 * @brief: publish camera visualization pose
 */
void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

/**
 * @brief: publish point cloud
 */
void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

/**
 * @brief: publish tf state
 */
void pubTF(const Estimator &estimator, const std_msgs::Header &header);

/**
 * @brief: publish key frame
 */
void pubKeyframe(const Estimator &estimator);

/**
 * @brief: publish pose graph
 */
void pubPoseGraph(const PoseGraph &pose_graph);

/**
 * @brief: publish semantic mask
 */
void pubSemanticMask(const cv::Mat &img, const std_msgs::Header &header);

/**
 * @brief: publish semantic image
 */
void pubSemanticImage(const cv::Mat &img, const std_msgs::Header &header);

class CameraPoseVisualization {
public:
    string marker_ns;
    /**
     * @brief: set camera color and instance
     */
    CameraPoseVisualization(float r, float g, float b, float a);

    /**
     * @brief: set camera scale, scale relates to the size of camera display 
     */
    void setScale(double s);

    /**
     * @brief: set camera line width
     */
    void setLineWidth(double width);

    /**
     * @brief: set camera pose and orientation
     */
    void add_pose(const Vector3d &p, const Quaterniond &q);

    /**
     * @brief: reset camera display
     */
    void reset();

    /**
     * @brief: publish camera display topic
     */
    void publish_by(ros::Publisher &pub, const std_msgs::Header &header);

    /**
     * @brief: add camera edge line to display 
     */
    void add_edge(const Vector3d &p0, const Vector3d &p1);

    /**
     * @brief: add camera loop edge line to display 
     */
    void add_loopedge(const Vector3d &p0, const Vector3d &p1);

private:
    vector<visualization_msgs::Marker> markers;
    std_msgs::ColorRGBA image_boundary_color;
    std_msgs::ColorRGBA optical_center_connector_color;
    double scale;
    double line_width;

    static const Vector3d imlt;
    static const Vector3d imlb;
    static const Vector3d imrt;
    static const Vector3d imrb;
    static const Vector3d oc;
    static const Vector3d lt0;
    static const Vector3d lt1;
    static const Vector3d lt2;
};
} // namespace FLOW_VINS
