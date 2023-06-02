/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-02-01 16:07:33
 * @Description: estimator node
 */

#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <mutex>
#include <ros/ros.h>
#include <ros/console.h>
#include <thread>
#include <unistd.h>
#include "include/estimator.h"
#include "include/segment.h"

using namespace FLOW_VINS;

static Estimator estimator;
static std::queue<sensor_msgs::ImageConstPtr> img0_buf;
static std::queue<sensor_msgs::ImageConstPtr> img1_buf;

/**
 * @brief: subscribe IMU topic data and sent it to estimator
 */
void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;

    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);

    estimator.inputImu(t, acc, gyr);
}

/**
 * @brief: get left image topic data from camera
 */
void imageCallback(const sensor_msgs::ImageConstPtr &image_msg) {
    mutex_image.lock();
    img0_buf.push(image_msg);
    mutex_image.unlock();
    mutex_segment.lock();
    estimator.yolo.image_buf.push(image_msg);
    mutex_segment.unlock();

    // detect unstable camera stream
    if (estimator.last_image_time == -1)
        image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - estimator.last_image_time > 1.0
             || image_msg->header.stamp.toSec() < estimator.last_image_time) {
        ROS_WARN("image discontinue! detect a new sequence!");
        estimator.newSequence();
    }
    estimator.last_image_time = image_msg->header.stamp.toSec();
}

/**
 * @brief: extract image from image buffer and sent it to estimator
 */
void process() {
    while (true) {
        if (STEREO || DEPTH) {
            cv::Mat img0, img1, mask;
            double time = 0;

            if (!img0_buf.empty() && !img1_buf.empty()) {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                mutex_image.lock();
                // binoculer image time delay less than 0.003s
                if (time0 < time1 - 0.003) {
                    img0_buf.pop();
                    mutex_image.unlock();
                } else if (time0 > time1 + 0.003) {
                    img1_buf.pop();
                    mutex_image.unlock();
                } else {
                    // extract the oldest frame from queue and dequeue it
                    time = img0_buf.front()->header.stamp.toSec();
                    img0 = getGrayImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    if (DEPTH) {
                        // get depth image
                        img1 = getDepthImageFromMsg(img1_buf.front());
                        img1_buf.pop();
                    } else {
                        // get right image
                        img1 = getGrayImageFromMsg(img1_buf.front());
                        img1_buf.pop();
                    }
                    mutex_image.unlock();
                    if (USE_SEGMENTATION) {
                        while (!estimator.yolo.semanticAvailable(time)) {
                            ROS_INFO("wait for semantic ...");
                            std::chrono::milliseconds dura(5);
                            std::this_thread::sleep_for(dura);
                        }
                        mutex_segment.lock();
                        // get semantic image
                        mask = getGrayImageFromMsg(estimator.yolo.mask_buf.front());
                        estimator.yolo.mask_buf.pop();
                        mutex_segment.unlock();
                    }
                }

                if (!img0.empty()) {
                    estimator.inputImage(time, img0, img1, mask);
                }
            }

        } else {
            cv::Mat img;
            double time = 0;
            if (!img0_buf.empty()) {
                // extract the oldest frame from queue and dequeue it
                mutex_image.lock();
                time = img0_buf.front()->header.stamp.toSec();
                img = getGrayImageFromMsg(img0_buf.front());
                img0_buf.pop();
                mutex_image.unlock();
            }
            if (!img.empty()) {
                estimator.inputImage(time, img);
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * @brief: vio system code entrance 
 */
int main(int argc, char *argv[]) {
    ros::init(argc, argv, "vio_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2) {
        std::cout << "please input: rosrun vio_system vio_system_node [config file] \n"
                     "for example: rosrun vio_system vio_system_node "
                     "src/Flow-VINS/config/euroc_stereo_imu_config.yaml "
                  << std::endl;
        return 1;
    }
    // read configuration file data
    readParameters(argv[1]);
    // set estimator parameter
    estimator.setParameter();
    // register estimator publisher
    registerPub(n);
    // subscribe IMU topic
    ros::Subscriber sub_imu =
        n.subscribe(IMU_TOPIC, 2000, imuCallback, ros::TransportHints().tcpNoDelay());
    // subscribe left image
    ros::Subscriber sub_img0 =
        n.subscribe(IMAGE0_TOPIC, 2000, imageCallback);
    // subscribe right image
    ros::Subscriber sub_img1 =
        n.subscribe<sensor_msgs::Image>(IMAGE1_TOPIC, 2000, [&](const sensor_msgs::ImageConstPtr &msg) {mutex_image.lock(); img1_buf.push(msg); mutex_image.unlock(); });
    // synchronize process
    std::thread thread_synchronize = std::thread(process);

    ros::spin();
    return 0;
}
