/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-08-15 16:05:04
 * @Description: 
 */
#include "../include/HealthyCheck.h"

namespace FLOW_VINS {

    void HealthyCheck::computeSensorScale() {
    }

    double HealthyCheck::getMarginScale() {
        return margin_scale;
    }

    double HealthyCheck::getVisualScale() {
        return visual_scale;
    }

    double HealthyCheck::getImuScale() {
        return imu_scale;
    }

    double HealthyCheck::getMagnetScale() {
        return magnet_scale;
    }
} // namespace FLOW_VINS