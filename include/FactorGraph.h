/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-28 10:45:46
 * @Description: factor
 */
#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <ros/assert.h>
#include <cstdlib>
#include <map>
#include <pthread.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <unordered_map>
#include <utility>

#include "Parameters.h"
#include "Common.h"

namespace FLOW_VINS {

    class IntegrationBase {
    public:
        IntegrationBase() = delete;

        /**
         * @brief: constructor function to initial parameters ;
         */
        IntegrationBase(const Vector3d &_acc_0, const Vector3d &_gyr_0,
                        Vector3d _linearized_ba, Vector3d _linearized_bg);

        /**
         * @brief: push back one IMU frame and propagate by midian integration
         */
        void push_back(double dt, const Vector3d &acc, const Vector3d &gyr);

        /**
         * @brief: repropagate IMU data, set new bias and propagate by midian integration
         */
        void repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg);

        /**
         * @brief: main process of midian integration and calculate jacobian and residual covariance matrix
         */
        void midPointIntegration(double _dt, const Vector3d &_acc_0, const Vector3d &_gyr_0,
                                 const Vector3d &_acc_1, const Vector3d &_gyr_1,
                                 const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                                 const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                                 Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                                 Vector3d &result_linearized_ba, Vector3d &result_linearized_bg,
                                 bool update_jacobian);

        /**
         * @brief: propagate IMU data, set new bias and propagate by midian integration
         */
        void propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1);

        /**
         * @brief: calculate residual of the pre-integration
         */
        Matrix<double, 15, 1> evaluate(const Vector3d &Pi, const Quaterniond &Qi,
                                       const Vector3d &Vi, const Vector3d &Bai,
                                       const Vector3d &Bgi, const Vector3d &Pj,
                                       const Quaterniond &Qj, const Vector3d &Vj,
                                       const Vector3d &Baj, const Vector3d &Bgj);

        double dt{};
        Vector3d acc_0, gyr_0;
        Vector3d acc_1, gyr_1;

        const Vector3d linearized_acc, linearized_gyr;
        Vector3d linearized_ba, linearized_bg;

        Matrix<double, 15, 15> jacobian, covariance;
        Matrix<double, 18, 18> noise;

        double sum_dt;
        Vector3d delta_p;
        Quaterniond delta_q;
        Vector3d delta_v;

        vector<double> dt_buf;
        vector<Vector3d> acc_buf;
        vector<Vector3d> gyr_buf;
    };

/**
 * @brief: class for IMU residual factor, which dimension is 7(PQ), 9(V,Ba,Bg), 7(PQ), 9(V,Ba,Bg)
 */
    class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
    public:
        IMUFactor() = delete;

        /**
         * @brief: constructor function
         */
        explicit IMUFactor(IntegrationBase *_pre_integration) :
                pre_integration(_pre_integration) {
        }

        /**
         * @brief: iteratively optimize each step and calculate the residual of the variable x in the current state, and the Jacobian of the residual to the variable
         */
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        IntegrationBase *pre_integration;
    };

/**
 * @brief: class for projection factor on binocular in one frame 
 */
    class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1, 1> {
    public:
        ProjectionOneFrameTwoCamFactor(Vector3d _pts_i, Vector3d _pts_j,
                                       const Vector2d &_velocity_i, const Vector2d &_velocity_j,
                                       double _td_i, double _td_j);

        /**
         * @brief: calculate residual jacobian, residual dimension: 2,
         * optimization variables: ric & tic (7), ric_right & tic_right (7), feature point inverse depth (1), td(1)
         */
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        Vector3d pts_i, pts_j;
        Vector3d velocity_i, velocity_j;
        double td_i, td_j;
        Matrix<double, 2, 3> tangent_base;
        static Matrix2d sqrt_info;
        static double sum_t;
    };

/**
 * @brief: class for projection factor on monocular in two frame 
 */
    class ProjectionTwoFrameOneCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> {
    public:
        ProjectionTwoFrameOneCamFactor(Vector3d _pts_i, Vector3d _pts_j,
                                       const Vector2d &_velocity_i, const Vector2d &_velocity_j,
                                       double _td_i, double _td_j);

        /**
         * @brief: calculate residual jacobian, residual dimension: 2,
         * optimization variables: P & Q in last frame (7), P & Q in current frame (7), ric & tic (7), feature point inverse depth (1), td(1)
         */
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        Vector3d pts_i, pts_j;
        Vector3d velocity_i, velocity_j;
        double td_i, td_j;
        Matrix<double, 2, 3> tangent_base;
        static Matrix2d sqrt_info;
        static double sum_t;
    };

/**
 * @brief: class for projection factor on binocular in two frame, last frame left camera to current frame right camera
 */
    class ProjectionTwoFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1> {
    public:
        ProjectionTwoFrameTwoCamFactor(Vector3d _pts_i, Vector3d _pts_j,
                                       const Vector2d &_velocity_i, const Vector2d &_velocity_j,
                                       double _td_i, double _td_j);

        /**
         * @brief: calculate residual jacobian, residual dimension: 2,
         * optimization variables: P & Q in last frame (7), P & Q in current frame (7), ric & tic (7), ric_right & tic_right (7), feature point inverse depth (1), td(1)
         */
        bool Evaluate(double const *const *parameters, double *residuals,
                      double **jacobians) const override;

        Vector3d pts_i, pts_j;
        Vector3d velocity_i, velocity_j;
        double td_i, td_j;
        Matrix<double, 2, 3> tangent_base;
        static Matrix2d sqrt_info;
        static double sum_t;
    };

/**
 * @brief: class for projection factor in two relocation frame 
 */
    class ProjectionRelocationFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
    public:
        ProjectionRelocationFactor(const Vector3d &_pts_i, const Vector3d &_pts_j);

        /**
         * @brief: calculate residual jacobian, residual dimension: 2,
         * optimization variables: P & Q in last frame (7), P & Q in current frame (7), ric & tic (7), feature point inverse depth (1)
         */
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        Vector3d pts_i, pts_j;
        Matrix<double, 2, 3> tangent_base;
        static Matrix2d sqrt_info;
        static double sum_t;
    };

/**
 * @brief: get local parameters size. R & P has 7 dims, for residual, only used 6 dims and drop the real part of the quaternion
 */
    inline int localSize(int size) {
        return size == 7 ? 6 : size;
    }

/**
 * @brief: class for marginlization residual block 
 */
    struct ResidualBlockInfo {
        ResidualBlockInfo(ceres::CostFunction *_cost_function,
                          ceres::LossFunction *_loss_function,
                          vector<double *> _parameter_blocks,
                          vector<int> _drop_set) :
                cost_function(_cost_function),
                loss_function(_loss_function),
                parameter_blocks(move(_parameter_blocks)), drop_set(move(_drop_set)) {
        }

        /**
         * @brief: calculate jacobian and residual
         */
        void Evaluate();

        // ceres problem build parameters
        ceres::CostFunction *cost_function;
        ceres::LossFunction *loss_function;
        double **raw_jacobians{};
        vector<Matrix<double, Dynamic, Dynamic, RowMajor>> jacobians;
        VectorXd residuals;
        // parameter block, (Twi, extrinsic params, depth, td, etc)
        vector<double *> parameter_blocks;
        // parameters waiting for marginalize, correspond to the index in parameter block
        vector<int> drop_set;
    };

/**
 * @brief: multi-thread struct to construct information matrix A = J^T * J, b = J^T * r
 */
    struct ThreadsStruct {
        vector<ResidualBlockInfo *> sub_factors;
        MatrixXd A;
        VectorXd b;
        unordered_map<long, int> parameter_block_size;
        unordered_map<long, int> parameter_block_idx;
    };

/**
 * @brief: class holds the prior information that was retained when the previous step was marginalized during optimization
 */
    class MarginalizationInfo {
    public:
        /**
         * @brief: constructor function, set valid flag with true
         */
        MarginalizationInfo() :
                valid(true) {};

        /**
         * @brief: destructor function
         */
        ~MarginalizationInfo();

        /**
         * @brief: add residual block info to marginalization info
         */
        void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);

        /**
         * @brief: get the parameter block, Jacobian matrix, and residual value corresponding to each IMU and visual observation
         */
        void preMarginalize();

        /**
         * @brief: turn on multi-threading to build information matrices H and b, and recover the linearized Jacobian and residual from H, b
         */
        void marginalize();

        /**
         * @brief: get parameter blocks
         */
        vector<double *> getParameterBlocks(unordered_map<long, double *> &addr_shift);

        // all residual terms associated with the current Marg frame
        vector<ResidualBlockInfo *> factors;
        // m is the number of variables to be marg, n is the number of reserved variables
        int m{}, n{};
        // <variable block start address, variable block size>
        unordered_map<long, int> parameter_block_size;
        // <variable block start address，variable block index>
        unordered_map<long, int> parameter_block_idx;
        // <variable block start address，variable data>
        unordered_map<long, double *> parameter_block_data;

        // variable to be retained size, index and data
        vector<int> keep_block_size;
        vector<int> keep_block_idx;
        vector<double *> keep_block_data;

        // related to calculate jacobian
        MatrixXd linearized_jacobians;
        VectorXd linearized_residuals;
        const double eps = 1e-8;
        bool valid;
    };

/**
 * @brief: class for hold the prior information cost function retained after the previous step of marginalization
 */
    class MarginalizationFactor : public ceres::CostFunction {
    public:
        /**
         * @brief: constructor function, set prior residual dimension
         */
        explicit MarginalizationFactor(MarginalizationInfo *_marginalization_info);

        /**
         * @brief: construct ceres residual parameter
         */
        bool Evaluate(double const *const *parameters, double *residuals,
                      double **jacobians) const override;

        MarginalizationInfo *marginalization_info;
    };

} // namespace FLOW_VINS