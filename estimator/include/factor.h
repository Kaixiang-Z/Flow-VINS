/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 15:07:14
 * @Description:
 */
#pragma once

#include "parameter.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <pthread.h>
#include <unordered_map>
#include <utility>

namespace FLOW_VINS {

class IntegrationBase {
public:
	IntegrationBase() = delete;

	/**
	 * @brief: constructor function to initial parameters ;
	 */
	IntegrationBase(const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0, Eigen::Vector3d _linearized_ba,
	                Eigen::Vector3d _linearized_bg);

	/**
	 * @brief: push back one IMU frame and propagate by midian integration
	 */
	void push_back(double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);

	/**
	 * @brief: repropagate IMU data, set new bias and propagate by midian integration
	 */
	void repropagate(const Eigen::Vector3d& _linearized_ba, const Eigen::Vector3d& _linearized_bg);

	/**
	 * @brief: main process of midian integration and calculate jacobian and residual covariance matrix
	 */
	void midPointIntegration(double _dt, const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
	                         const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1,
	                         const Eigen::Vector3d& delta_p, const Eigen::Quaterniond& delta_q,
	                         const Eigen::Vector3d& delta_v, const Eigen::Vector3d& linearized_ba,
	                         const Eigen::Vector3d& linearized_bg, Eigen::Vector3d& result_delta_p,
	                         Eigen::Quaterniond& result_delta_q, Eigen::Vector3d& result_delta_v,
	                         Eigen::Vector3d& result_linearized_ba, Eigen::Vector3d& result_linearized_bg,
	                         bool update_jacobian);

	/**
	 * @brief: propagate IMU data, set new bias and propagate by midian integration
	 */
	void propagate(double _dt, const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1);

	/**
	 * @brief: calculate residual of the pre-integration
	 */
	Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi,
	                                      const Eigen::Vector3d& Vi, const Eigen::Vector3d& Bai,
	                                      const Eigen::Vector3d& Bgi, const Eigen::Vector3d& Pj,
	                                      const Eigen::Quaterniond& Qj, const Eigen::Vector3d& Vj,
	                                      const Eigen::Vector3d& Baj, const Eigen::Vector3d& Bgj);

	double dt{};
	Eigen::Vector3d acc_0, gyr_0;
	Eigen::Vector3d acc_1, gyr_1;

	const Eigen::Vector3d linearized_acc, linearized_gyr;
	Eigen::Vector3d linearized_ba, linearized_bg;

	Eigen::Matrix<double, 15, 15> jacobian, covariance;
	Eigen::Matrix<double, 18, 18> noise;

	double sum_dt;
	Eigen::Vector3d delta_p;
	Eigen::Quaterniond delta_q;
	Eigen::Vector3d delta_v;

	std::vector<double> dt_buf;
	std::vector<Eigen::Vector3d> acc_buf;
	std::vector<Eigen::Vector3d> gyr_buf;
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
	explicit IMUFactor(IntegrationBase* _pre_integration)
	    : pre_integration(_pre_integration) {}

	/**
	 * @brief: iteratively optimize each step and calculate the residual of the variable x in the current state, and the
	 * Jacobian of the residual to the variable
	 */
	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

	IntegrationBase* pre_integration;
};

const int NUM_THREADS = 4;

/**
 * @brief: get local parameters size. R & P has 7 dims, for residual, only used 6 dims and drop the real part of the
 * quaternion
 */
inline int localSize(int size) { return size == 7 ? 6 : size; }

/**
 * @brief: class for marginlization residual block
 */
struct ResidualBlockInfo {
	ResidualBlockInfo(ceres::CostFunction* _cost_function, ceres::LossFunction* _loss_function,
	                  std::vector<double*> _parameter_blocks, std::vector<int> _drop_set)
	    : cost_function(_cost_function)
	    , loss_function(_loss_function)
	    , parameter_blocks(std::move(_parameter_blocks))
	    , drop_set(std::move(_drop_set)) {}

	/**
	 * @brief: calculate jacobian and residual
	 */
	void Evaluate();

	// ceres problem build parameters
	ceres::CostFunction* cost_function;
	ceres::LossFunction* loss_function;
	double** raw_jacobians{};
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
	Eigen::VectorXd residuals;
	// parameter block, (Twi, extrinsic params, depth, td, etc)
	std::vector<double*> parameter_blocks;
	// parameters waiting for marginalize, correspond to the index in parameter block
	std::vector<int> drop_set;
};

/**
 * @brief: multi-thread struct to construct information matrix A = J^T * J, b = J^T * r
 */
struct ThreadsStruct {
	std::vector<ResidualBlockInfo*> sub_factors;
	Eigen::MatrixXd A;
	Eigen::VectorXd b;
	std::unordered_map<long, int> parameter_block_size;
	std::unordered_map<long, int> parameter_block_idx;
};

/**
 * @brief: class holds the prior information that was retained when the previous step was marginalized during
 * optimization
 */
class MarginalizationInfo {
public:
	/**
	 * @brief: constructor function, set valid flag with true
	 */
	MarginalizationInfo()
	    : valid(true){};

	/**
	 * @brief: destructor function
	 */
	~MarginalizationInfo();

	/**
	 * @brief: add residual block info to marginalization info
	 */
	void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);

	/**
	 * @brief: get the parameter block, Jacobian matrix, and residual value corresponding to each IMU and visual
	 * observation
	 */
	void preMarginalize();

	/**
	 * @brief: turn on multi-threading to build information matrices H and b, and recover the linearized Jacobian and
	 * residual from H, b
	 */
	void marginalize();

	/**
	 * @brief: get parameter blocks
	 */
	std::vector<double*> getParameterBlocks(std::unordered_map<long, double*>& addr_shift);

	// all residual terms associated with the current Marg frame
	std::vector<ResidualBlockInfo*> factors;
	// m is the number of variables to be marg, n is the number of reserved variables
	int m{}, n{};
	// <variable block start address, variable block size>
	std::unordered_map<long, int> parameter_block_size;
	// <variable block start address，variable block index>
	std::unordered_map<long, int> parameter_block_idx;
	// <variable block start address，variable data>
	std::unordered_map<long, double*> parameter_block_data;

	// variable to be retained size, index and data
	std::vector<int> keep_block_size;
	std::vector<int> keep_block_idx;
	std::vector<double*> keep_block_data;

	// related to calculate jacobian
	Eigen::MatrixXd linearized_jacobians;
	Eigen::VectorXd linearized_residuals;
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
	explicit MarginalizationFactor(MarginalizationInfo* _marginalization_info);

	/**
	 * @brief: construct ceres residual parameter
	 */
	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

	MarginalizationInfo* marginalization_info;
};

/**
 * @brief: class for projection factor on binocular in one frame
 */
class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1, 1> {
public:
	ProjectionOneFrameTwoCamFactor(Eigen::Vector3d _pts_i, Eigen::Vector3d _pts_j, const Eigen::Vector2d& _velocity_i,
	                               const Eigen::Vector2d& _velocity_j, double _td_i, double _td_j);

	/**
	 * @brief: calculate residual jacobian, residual dimension: 2,
	 * optimization variables: ric & tic (7), ric_right & tic_right (7), feature point inverse depth (1), td(1)
	 */
	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

	Eigen::Vector3d pts_i, pts_j;
	Eigen::Vector3d velocity_i, velocity_j;
	double td_i, td_j;
	Eigen::Matrix<double, 2, 3> tangent_base;
	static Eigen::Matrix2d sqrt_info;
	static double sum_t;
};

/**
 * @brief: class for projection factor on monocular in two frame
 */
class ProjectionTwoFrameOneCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> {
public:
	ProjectionTwoFrameOneCamFactor(Eigen::Vector3d _pts_i, Eigen::Vector3d _pts_j, const Eigen::Vector2d& _velocity_i,
	                               const Eigen::Vector2d& _velocity_j, double _td_i, double _td_j);

	/**
	 * @brief: calculate residual jacobian, residual dimension: 2,
	 * optimization variables: P & Q in last frame (7), P & Q in current frame (7), ric & tic (7), feature point inverse
	 * depth (1), td(1)
	 */
	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

	Eigen::Vector3d pts_i, pts_j;
	Eigen::Vector3d velocity_i, velocity_j;
	double td_i, td_j;
	Eigen::Matrix<double, 2, 3> tangent_base;
	static Eigen::Matrix2d sqrt_info;
	static double sum_t;
};

/**
 * @brief: class for projection factor on binocular in two frame, last frame left camera to current frame right camera
 */
class ProjectionTwoFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1> {
public:
	ProjectionTwoFrameTwoCamFactor(Eigen::Vector3d _pts_i, Eigen::Vector3d _pts_j, const Eigen::Vector2d& _velocity_i,
	                               const Eigen::Vector2d& _velocity_j, double _td_i, double _td_j);

	/**
	 * @brief: calculate residual jacobian, residual dimension: 2,
	 * optimization variables: P & Q in last frame (7), P & Q in current frame (7), ric & tic (7), ric_right & tic_right
	 * (7), feature point inverse depth (1), td(1)
	 */
	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

	Eigen::Vector3d pts_i, pts_j;
	Eigen::Vector3d velocity_i, velocity_j;
	double td_i, td_j;
	Eigen::Matrix<double, 2, 3> tangent_base;
	static Eigen::Matrix2d sqrt_info;
	static double sum_t;
};

} // namespace FLOW_VINS