/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-25 08:25:17
 * @Description: segmantic inference
 *      
 */
#pragma once

#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <NvInfer.h>
#include <fstream>
#include <thread>
#include "common.h"
#include "parameters.h"
#include "publisher.h"

/**
 * @brief: check if cuda is available 
 */
#define CHECK_CUDA(call)                                \
    do {                                                \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess) {                \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

namespace FLOW_VINS {

/**
 * @brief: class for tensorrt logger 
 */
class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
        reportableSeverity(severity) {
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            cerr << "INFO: ";
            break;
        default:
            cerr << "VERBOSE: ";
            break;
        }
        cerr << msg << endl;
    }
};

/**
 * @brief: get tensor dims size 
 */
inline int getSizeByDims(const nvinfer1::Dims &dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

/**
 * @brief: get data type correspond size
 */
inline int typeToSize(const nvinfer1::DataType &data_type) {
    switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;

    default:
        return 4;
    }
}

/**
 * @brief: restrict value to range of min & max
 */
inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

/**
 * @brief: struct for tensorrt params 
 */
struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    string name;
};

struct Object {
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0;
    cv::Mat boxMask;
};

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};

class YOLOv8_seg {
public:
    /**
     * @brief: constructor function for segmentation class 
     */
    explicit YOLOv8_seg();
    ~YOLOv8_seg();

    /**
     * @brief: set segmentation model parameters
     */
    void setParameter();

    /**
     * @brief: main process of segmentation
     */
    void inferenceProcess();

    /**
     * @brief: check the semantic image is available
     */
    bool semanticAvailable(double t);

    /**
     * @brief: make pipe and warm up to get faster inference speed
     */
    void makePipe(bool warmup = true);

    /**
     * @brief: preprocess image to adjust the input size of tensorrt model
     */
    void letterBox(const cv::Mat &image, cv::Mat &out, cv::Size &size);

    /**
     * @brief: copy data to GPU menmory from cv mat
     */
    void copyFromMat(const cv::Mat &image, cv::Size &size);

    /**
     * @brief: main process of segmentation inference 
     */
    void inference();

    /**
     * @brief: main process of post process, from tensor to mat
     */
    void postProcess(vector<Object> &objs,
                     float score_thres = 0.25f,
                     float iou_thres = 0.65f,
                     int topk = 100,
                     int seg_channels = 32,
                     int seg_h = 160,
                     int seg_w = 160);

    /**
     * @brief:  main process of darw segmentation mask in orin image
     */
    static void drawObjects(const cv::Mat &image, cv::Mat &res, const vector<Object> &objs);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    vector<Binding> input_bindings;
    vector<Binding> output_bindings;
    vector<void *> host_ptrs;
    vector<void *> device_ptrs;

    PreParam pparam;

    bool init_thread_flag;
    bool segment_finish_flag;
    queue<sensor_msgs::ImageConstPtr> image_buf;
    queue<sensor_msgs::ImageConstPtr> mask_buf;

private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

} // namespace FLOW_VINS
