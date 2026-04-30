#include "TensorRtDetector.h"

#include "Detection.h"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> readFile(fs::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void writeFile(fs::path const& path, void const* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write file: " + path.string());
    }
    file.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
}

} // namespace

void TrtLogger::log(Severity severity, char const* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

void TensorRtDetector::CudaDeleter::operator()(void* ptr) const {
    if (ptr) {
        cudaFree(ptr);
    }
}

TensorRtDetector::TensorRtDetector(fs::path const& onnxPath, fs::path const& cachePath) {
    initLibNvInferPlugins(&logger_, "");
    buildOrLoadEngine(onnxPath, cachePath);
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    discoverTensors();

    inputBytes_ = 3ULL * kInputW * kInputH * sizeof(float);
    outputCount_ = 300ULL * 6ULL;
    outputBytes_ = outputCount_ * sizeof(float);
    output_.resize(outputCount_);

    void* input{};
    void* output{};
    checkCuda(cudaMalloc(&input, inputBytes_), "cudaMalloc input");
    checkCuda(cudaMalloc(&output, outputBytes_), "cudaMalloc output");
    deviceInput_.reset(input);
    deviceOutput_.reset(output);
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    if (!context_->setTensorAddress(inputName_.c_str(), deviceInput_.get()) ||
        !context_->setTensorAddress(outputName_.c_str(), deviceOutput_.get())) {
        throw std::runtime_error("Failed to set TensorRT tensor addresses");
    }
}

TensorRtDetector::~TensorRtDetector() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

std::vector<float> TensorRtDetector::infer(float const* input) {
    checkCuda(cudaMemcpyAsync(deviceInput_.get(), input, inputBytes_, cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync H2D");
    return inferDeviceInput();
}

std::vector<float> TensorRtDetector::inferDeviceInput() {
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    checkCuda(cudaMemcpyAsync(output_.data(), deviceOutput_.get(), outputBytes_, cudaMemcpyDeviceToHost, stream_),
              "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    return output_;
}

void TensorRtDetector::buildOrLoadEngine(fs::path const& onnxPath, fs::path const& cachePath) {
    TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger_));
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    if (fs::exists(cachePath)) {
        auto bytes = readFile(cachePath);
        if (auto* engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size())) {
            std::cout << "Loaded TensorRT engine cache: " << cachePath << '\n';
            engine_.reset(engine);
            return;
        }
        std::cerr << "Engine cache is incompatible, rebuilding from ONNX.\n";
    }

    std::cout << "Building TensorRT engine from ONNX. First launch may take a while...\n";
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }

    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network");
    }

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser");
    }
    if (!parser->parseFromFile(onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << '\n';
        }
        throw std::runtime_error("Failed to parse ONNX: " + onnxPath.string());
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    TrtUniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("Failed to build TensorRT engine");
    }
    writeFile(cachePath, serialized->data(), serialized->size());
    std::cout << "Saved TensorRT engine cache: " << cachePath << '\n';

    auto* engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    if (!engine) {
        throw std::runtime_error("Failed to deserialize freshly built engine");
    }
    engine_.reset(engine);
}

void TensorRtDetector::discoverTensors() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        char const* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
        } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
        }
    }
    if (inputName_.empty() || outputName_.empty()) {
        throw std::runtime_error("Could not find TensorRT input/output tensors");
    }
    std::cout << "Input tensor: " << inputName_ << '\n';
    std::cout << "Output tensor: " << outputName_ << '\n';
}
