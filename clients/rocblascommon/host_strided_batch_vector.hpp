/* ************************************************************************
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cassert>
#include <cmath>
#include <memory>
#include <ostream>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

template <typename T, size_t PAD, typename U>
class device_strided_batch_vector;

template <typename T>
class host_strided_batch_vector
{
public:
    host_strided_batch_vector(rocblas_int n,
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
        data_ = std::make_unique<T[]>(sz);
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

    template <size_t PAD, typename U>
    hipError_t transfer_from(const device_strided_batch_vector<T, PAD, U>& that)
    {
        assert(n_ == that.n());
        assert(inc_ == that.inc());
        assert(stride_ == that.stride());
        assert(batch_count_ == that.batch_count());

        return hipMemcpy(data_.get(), that.data(), sizeof(T) * size(), hipMemcpyDeviceToHost);
    }

private:
    std::unique_ptr<T[]> data_;
    rocblas_int n_;
    rocblas_int inc_;
    rocblas_stride stride_;
    rocblas_int batch_count_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const host_strided_batch_vector<T>& hsbv)
{
    rocblas_int n = hsbv.n();
    rocblas_int inc = std::abs(hsbv.inc());
    rocblas_int batch_count = hsbv.batch_count();

    for(rocblas_int b = 0; b < batch_count; ++b)
    {
        T* hv = hsbv[b];
        os << "[" << b << "] = { ";
        for(rocblas_int i = 0; i < n; ++i)
        {
            os << hv[i * inc];
            if(i + 1 < n)
                os << ", ";
        }
        os << " }" << std::endl;
    }
    return os;
}
