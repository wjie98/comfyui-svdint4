#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

struct SourceLocation {
    const char *file;
    int line;

#if defined(_MSC_VER) && !defined(__clang__)
    static constexpr SourceLocation current(const char *file = __FILE__, int line = __LINE__) {
        return {file, line};
    }
#else
    static constexpr SourceLocation current(const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
        return {file, line};
    }
#endif
};

inline cudaError_t checkCUDA(cudaError_t status, SourceLocation location = SourceLocation::current()) {
    if (status == cudaSuccess) {
        return status;
    }
    (void)cudaGetLastError();
    std::ostringstream msg;
    msg << "CUDA error: " << cudaGetErrorString(status) << " at " << location.file << ":" << location.line;
    throw std::runtime_error(msg.str());
}

inline thread_local std::vector<cudaStream_t> stackCUDAStreams;

inline cudaStream_t getCurrentCUDAStream() {
    return stackCUDAStreams.empty() ? cudaStream_t{} : stackCUDAStreams.back();
}

inline cudaDeviceProp *getCurrentDeviceProperties() {
    static thread_local std::map<int, cudaDeviceProp> props;

    int device = 0;
    checkCUDA(cudaGetDevice(&device));
    auto it = props.find(device);
    if (it == props.end()) {
        cudaDeviceProp prop{};
        checkCUDA(cudaGetDeviceProperties(&prop, device));
        it = props.emplace(device, prop).first;
    }
    return &it->second;
}

template <typename T, typename U>
constexpr T ceilDiv(T x, U y) {
    const T divisor = static_cast<T>(y);
    return (x + divisor - 1) / divisor;
}

struct Device {
    enum Type { CPU, CUDA };

    Type type = CPU;
    int index = 0;
};

struct TensorShape {
    std::vector<int> dataExtent;
    std::vector<int> dataStride;
    int64_t offset = 0;

    int ndims() const {
        return static_cast<int>(dataExtent.size());
    }

    const int &operator[](int dim) const {
        if (dim < 0) {
            dim += ndims();
        }
        return dataExtent.at(dim);
    }

    int &operator[](int dim) {
        return const_cast<int &>(static_cast<const TensorShape &>(*this)[dim]);
    }

    size_t stride(int dim) const {
        if (dim < 0) {
            dim += ndims();
        }
        if (!dataStride.empty()) {
            return dataStride.at(dim);
        }

        size_t value = 1;
        for (int i = ndims() - 1; i > dim; --i) {
            value *= static_cast<size_t>(dataExtent.at(i));
        }
        return value;
    }

    size_t size() const {
        if (dataExtent.empty()) {
            return 0;
        }
        size_t value = 1;
        for (int dim : dataExtent) {
            value *= static_cast<size_t>(dim);
        }
        return value;
    }
};

class Tensor {
public:
    enum ScalarType {
        INVALID_SCALAR_TYPE,
        INT8,
        INT16,
        INT32,
        INT64,
        FP16,
        FP32,
        BF16,
        FP8_E4M3,
        FP8_E5M2,
    };

    TensorShape shape;
    ScalarType scalarType = INVALID_SCALAR_TYPE;
    void *ptr = nullptr;
    Device dev{};
    std::shared_ptr<void> owner;

    bool valid() const {
        return ptr != nullptr && !shape.dataExtent.empty();
    }

    int size(int dim) const {
        return shape[dim];
    }

    size_t stride(int dim) const {
        return shape.stride(dim);
    }

    size_t numel() const {
        return shape.size();
    }

    size_t ndims() const {
        return static_cast<size_t>(shape.ndims());
    }

    ScalarType scalar_type() const {
        return scalarType;
    }

    ScalarType dtype() const {
        return scalarType;
    }

    Device device() const {
        return dev;
    }

    size_t scalar_size() const {
        switch (scalarType) {
        case INT8:
        case FP8_E4M3:
        case FP8_E5M2:
            return 1;
        case INT16:
        case FP16:
        case BF16:
            return 2;
        case INT32:
        case FP32:
            return 4;
        case INT64:
            return 8;
        default:
            throw std::runtime_error("invalid tensor scalar type");
        }
    }

    void *data_ptr() {
        return static_cast<char *>(ptr) + shape.offset * static_cast<int64_t>(scalar_size());
    }

    const void *data_ptr() const {
        return static_cast<const char *>(ptr) + shape.offset * static_cast<int64_t>(scalar_size());
    }

    template <typename T>
    T *data_ptr() {
        return reinterpret_cast<T *>(data_ptr());
    }

    template <typename T>
    const T *data_ptr() const {
        return reinterpret_cast<const T *>(data_ptr());
    }

    Tensor &zero_() {
        if (numel() == 0) {
            return *this;
        }
        assert(dev.type == Device::CUDA);
        checkCUDA(cudaMemsetAsync(data_ptr(), 0, numel() * scalar_size(), getCurrentCUDAStream()));
        return *this;
    }
};
