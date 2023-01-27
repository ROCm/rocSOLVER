/* ************************************************************************
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cassert>
#include <cmath>
#include <memory>
#include <ostream>
#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "common_host_helpers.hpp"
#include "device_memory.hpp"

template <typename T>
class host_strided_batch_vector;

template <typename T, size_t PAD = 0, typename U = T>
class device_strided_batch_vector
{
public:
    device_strided_batch_vector(rocblas_int n,
                              rocblas_int inc,
                              rocblas_stride stride,
                              rocblas_int batch_count)
        : n_(n)
        , inc_(inc)
        , stride_(stride)
        , batch_count_(batch_count)
    {
        assert(n > 0);
        assert(stride != 0);
        assert(batch_count > 0);
        assert(size_t(n) * std::abs(inc) <= std::abs(stride));

        const size_t sz = size();
        assert(sz > 0);
        T* data;
        THROW_IF_HIP_ERROR(hipMalloc(&data, sizeof(T) * sz));
        data_ = std::unique_ptr<T[], device_deleter>(data);
    }

    // The number of elements in each vector.
    rocblas_int n() const noexcept
    {
        return n_;
    }

    // The increment between elements in each vector.
    rocblas_int inc() const noexcept
    {
        return inc_;
    }

    // The number of vectors in the batch.
    rocblas_int batch_count() const noexcept
    {
        return batch_count_;
    }

    // The total number elements in all vectors in the batch.
    rocblas_stride size() const
    {
        return size_t(std::abs(stride_)) * batch_count_;
    }

    // The number of elements from the start of one vector to the start of the next.
    rocblas_stride stride() const noexcept
    {
        return stride_;
    }

    // Returns a vector from the batch.
    T* operator[](rocblas_int batch_index)
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);

        rocblas_stride index
            = stride_ >= 0 ? stride_ * batch_index : stride_ * (batch_index - batch_count_ + 1);

        assert(index >= 0);
        assert(index < size());

        return &data_[index];
    }

    // Returns a vector from the batch.
    const T* operator[](rocblas_int batch_index) const
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);

        rocblas_stride index
            = stride_ >= 0 ? stride_ * batch_index : stride_ * (batch_index - batch_count_ + 1);

        assert(index >= 0);
        assert(index < size());

        return &data_[index];
    }

    // Returns a pointer to the underlying array.
    T* data() noexcept
    {
        return data_.get();
    }

    // Returns a pointer to the underlying array.
    const T* data() const noexcept
    {
        return data_.get();
    }

    hipError_t transfer_from(const host_strided_batch_vector<T>& that)
    {
        assert(n_ == that.n());
        assert(inc_ == that.inc());
        assert(stride_ == that.stride());
        assert(batch_count_ == that.batch_count());

        return hipMemcpy(this->data(), that.data(), sizeof(T) * size(), hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return hipSuccess;
    }

private:
    std::unique_ptr<T[], device_deleter> data_;
    rocblas_int n_;
    rocblas_int inc_;
    rocblas_stride stride_;
    rocblas_int batch_count_;
};
