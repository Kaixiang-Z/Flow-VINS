/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-25 08:25:17
 * @Description: segmantic inference
 *
 */
#pragma once

#include "logging.h"
#include <NvInferPlugin.h>
#include <fstream>
#include <opencv4/opencv2/dnn.hpp>

namespace FLOW_VINS {
class YOLOv8_seg {
public:
	/**
	 * @brief: constructor function for segmentation class
	 */
	explicit YOLOv8_seg(const std::string& engine_file_path);
	~YOLOv8_seg();

	/**
	 * @brief: make pipe and warm up to get faster inference speed
	 */
	void makePipe(bool warmup = true);

	/**
	 * @brief: preprocess image to adjust the input size of tensorrt model
	 */
	void letterBox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

	/**
	 * @brief: copy data to GPU menmory from cv mat
	 */
	void copyFromMat(const cv::Mat& image, cv::Size& size);

	/**
	 * @brief: main process of segmentation inference
	 */
	void inference();

	/**
	 * @brief: main process of post process, from tensor to mat
	 */
	void postProcess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100,
	                 int seg_channels = 32, int seg_h = 160, int seg_w = 160);

	/**
	 * @brief:  main process of darw segmentation mask in orin image
	 */
	static void drawObjects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs);

	int num_bindings;
	int num_inputs = 0;
	int num_outputs = 0;
	std::vector<Binding> input_bindings;
	std::vector<Binding> output_bindings;
	std::vector<void*> host_ptrs;
	std::vector<void*> device_ptrs;

	PreParam pparam;

private:
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
	cudaStream_t stream = nullptr;
	Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};
} // namespace FLOW_VINS
