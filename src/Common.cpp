/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-28 11:07:04
 * @Description: common functions
 */
#include "../include/Common.h"

namespace FLOW_VINS {

    cv::Mat getGrayImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
        cv_bridge::CvImageConstPtr ptr;
        if (img_msg->encoding == "8UC1") {
            sensor_msgs::Image img;
            img.header = img_msg->header;
            img.height = img_msg->height;
            img.width = img_msg->width;
            img.is_bigendian = img_msg->is_bigendian;
            img.step = img_msg->step;
            img.data = img_msg->data;
            img.encoding = "mono8";
            ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        } else
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

        cv::Mat img = ptr->image.clone();
        return img;
    }

    cv::Mat getDepthImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
        cv::Mat depth_img;
        if (img_msg->encoding == "mono16" || img_msg->encoding == "16UC1") {
            depth_img = cv_bridge::toCvShare(img_msg)->image;
        } else if (img_msg->encoding == "32FC1") {
            depth_img = cv_bridge::toCvShare(img_msg)->image;
            depth_img.convertTo(depth_img, CV_16UC1, 1000);
        } else {
            ROS_ASSERT_MSG(1, "Unknown depth encoding!");
        }
        cv::medianBlur(depth_img, depth_img, 5);
        return depth_img;
    }

    cv::Mat getRgbImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
        cv_bridge::CvImageConstPtr ptr;
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = sensor_msgs::image_encodings::BGR8;
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        cv::Mat semantic_img = ptr->image.clone();
        return semantic_img;
    }

} // namespace FLOW_VINS