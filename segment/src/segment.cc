/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-05-25 08:25:09
 * @Description: segmantic inference
 */
#include "../include/segment.h"

namespace FLOW_VINS {

/**
 * @brief: image erode to prevent dynamic point in image edge
 */
static void erode(cv::Mat& img) {
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::erode(img, img, kernel);
}

YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path) {
	// find engine file path
	std::ifstream file(engine_file_path, std::ios::binary);
	assert(file.good());
	file.seekg(0, std::ios::end);
	auto size = file.tellg();
	file.seekg(0, std::ios::beg);
	char* trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();

	// build tensorrt inference model
	initLibNvInferPlugins(&this->gLogger, "");
	this->runtime = nvinfer1::createInferRuntime(this->gLogger);
	assert(this->runtime != nullptr);

	this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);

	this->context = this->engine->createExecutionContext();

	assert(this->context != nullptr);
	cudaStreamCreate(&this->stream);
	this->num_bindings = this->engine->getNbBindings();

	// get model params
	for (int i = 0; i < this->num_bindings; ++i) {
		Binding binding;
		nvinfer1::Dims dims;
		nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
		std::string name = this->engine->getBindingName(i);
		binding.name = name;
		binding.dsize = typeToSize(dtype);

		bool IsInput = engine->bindingIsInput(i);
		if (IsInput) {
			this->num_inputs += 1;
			dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
			binding.size = getSizeByDims(dims);
			binding.dims = dims;
			this->input_bindings.push_back(binding);
			// set max opt shape
			this->context->setBindingDimensions(i, dims);

		} else {
			dims = this->context->getBindingDimensions(i);
			binding.size = getSizeByDims(dims);
			binding.dims = dims;
			this->output_bindings.push_back(binding);
			this->num_outputs += 1;
		}
	}
}

YOLOv8_seg::~YOLOv8_seg() {
	this->context->destroy();
	this->engine->destroy();
	this->runtime->destroy();
	cudaStreamDestroy(this->stream);
	for (auto& ptr : this->device_ptrs) {
		CHECK(cudaFree(ptr));
	}

	for (auto& ptr : this->host_ptrs) {
		CHECK(cudaFreeHost(ptr));
	}
}

void YOLOv8_seg::makePipe(bool warmup) {
	for (auto& bindings : this->input_bindings) {
		void* d_ptr;
		CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
		this->device_ptrs.push_back(d_ptr);
	}

	for (auto& bindings : this->output_bindings) {
		void *d_ptr, *h_ptr;
		size_t size = bindings.size * bindings.dsize;
		CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
		CHECK(cudaHostAlloc(&h_ptr, size, 0));
		this->device_ptrs.push_back(d_ptr);
		this->host_ptrs.push_back(h_ptr);
	}

	if (warmup) {
		for (int i = 0; i < 10; i++) {
			for (auto& bindings : this->input_bindings) {
				size_t size = bindings.size * bindings.dsize;
				void* h_ptr = malloc(size);
				memset(h_ptr, 0, size);
				CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
				free(h_ptr);
			}
			this->inference();
		}
		printf("model warmup 10 times\n");
	}
}

void YOLOv8_seg::letterBox(const cv::Mat& image, cv::Mat& out, cv::Size& size) {
	const float inp_h = size.height;
	const float inp_w = size.width;
	float height = image.rows;
	float width = image.cols;

	float r = std::min(inp_h / height, inp_w / width);
	int padw = std::round(width * r);
	int padh = std::round(height * r);

	cv::Mat tmp;
	if ((int)width != padw || (int)height != padh) {
		cv::resize(image, tmp, cv::Size(padw, padh));
	} else {
		tmp = image.clone();
	}

	float dw = inp_w - padw;
	float dh = inp_h - padh;

	dw /= 2.0f;
	dh /= 2.0f;
	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

	cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

	cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

	this->pparam.ratio = 1 / r;
	this->pparam.dw = dw;
	this->pparam.dh = dh;
	this->pparam.height = height;
	this->pparam.width = width;
}

void YOLOv8_seg::copyFromMat(const cv::Mat& image, cv::Size& size) {
	cv::Mat nchw;
	this->letterBox(image, nchw, size);

	this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});

	CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(),
	                      cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_seg::inference() {
	this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);

	for (int i = 0; i < this->num_outputs; i++) {
		size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
		CHECK(cudaMemcpyAsync(this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize,
		                      cudaMemcpyDeviceToHost, this->stream));
	}
	cudaStreamSynchronize(this->stream);
}

void YOLOv8_seg::postProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int seg_channels,
                             int seg_h, int seg_w) {
	objs.clear();
	auto input_h = this->input_bindings[0].dims.d[2];
	auto input_w = this->input_bindings[0].dims.d[3];
	auto num_anchors = this->output_bindings[0].dims.d[1];
	auto num_channels = this->output_bindings[0].dims.d[2];

	auto& dw = this->pparam.dw;
	auto& dh = this->pparam.dh;
	auto& width = this->pparam.width;
	auto& height = this->pparam.height;
	auto& ratio = this->pparam.ratio;

	auto* output = static_cast<float*>(this->host_ptrs[0]);
	cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(this->host_ptrs[1]));

	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<cv::Rect> bboxes;
	std::vector<cv::Mat> mask_confs;
	std::vector<int> indices;

	for (int i = 0; i < num_anchors; i++) {
		float* ptr = output + i * num_channels;
		float score = *(ptr + 4);
		if (score > score_thres) {
			float x0 = *ptr++ - dw;
			float y0 = *ptr++ - dh;
			float x1 = *ptr++ - dw;
			float y1 = *ptr++ - dh;

			x0 = clamp(x0 * ratio, 0.f, width);
			y0 = clamp(y0 * ratio, 0.f, height);
			x1 = clamp(x1 * ratio, 0.f, width);
			y1 = clamp(y1 * ratio, 0.f, height);

			int label = *(++ptr);
			cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, ++ptr);
			mask_confs.push_back(mask_conf);
			labels.push_back(label);
			scores.push_back(score);
			bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
		}
	}

	cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);

	cv::Mat masks;
	int cnt = 0;
	for (auto& i : indices) {
		if (cnt >= topk) {
			break;
		}
		if (labels[i] > 2)
			continue;
		cv::Rect tmp = bboxes[i];
		Object obj;
		obj.label = labels[i];
		obj.rect = tmp;
		obj.prob = scores[i];
		masks.push_back(mask_confs[i]);
		objs.push_back(obj);
		cnt += 1;
	}

	if (!masks.empty()) {
		cv::Mat matmulRes = (masks * protos).t();
		cv::Mat maskMat = matmulRes.reshape(cnt, {seg_w, seg_h});

		std::vector<cv::Mat> maskChannels;
		cv::split(maskMat, maskChannels);
		int scale_dw = dw / input_w * seg_w;
		int scale_dh = dh / input_h * seg_h;

		cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

		for (int i = 0; i < cnt; i++) {
			cv::Mat dest, mask;
			cv::exp(-maskChannels[i], dest);
			dest = 1.0 / (1.0 + dest);
			dest = dest(roi);

			cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);

			objs[i].boxMask = mask(objs[i].rect) > 0.5f;
		}
	}
}

void YOLOv8_seg::drawObjects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs) {
	res = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	cv::Mat mask = image.clone();
	for (auto& obj : objs) {
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
		cv::dilate(obj.boxMask, obj.boxMask, kernel);
		res(obj.rect).setTo(cv::Scalar(255, 255, 255), obj.boxMask);
	}
}

} // namespace FLOW_VINS
