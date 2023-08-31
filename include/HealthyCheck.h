/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-08-15 16:05:17
 * @Description: 
 */
#pragma once

namespace FLOW_VINS {
class HealthyCheck {
public:
    /**
     * @brief: constructor function for healthy check
     */
    HealthyCheck() :
        margin_scale(1.0), visual_scale(1.0), imu_scale(1.0), magnet_scale(1.0) {
    }

    /**
     * @brief: compute sensor weight for ceres solver
     */
    void computeSensorScale();

    /**
     * @brief: get marginalization weight scale
     */
    double getMarginScale();

    /**
     * @brief: get visualization weight scale
     */
    double getVisualScale();

    /**
     * @brief: get imu weight scale
     */
    double getImuScale();

    /**
     * @brief: get magnetometer weight scale
     */
    double getMagnetScale();

private:
    double margin_scale, visual_scale, imu_scale, magnet_scale;
};
} // namespace FLOW_VINS
