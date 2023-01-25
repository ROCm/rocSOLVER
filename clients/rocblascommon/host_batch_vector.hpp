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
class device_batch_vector;

template <typename T>
class host_batch_vector
{
public:
    host_batch_vector(rocblas_int n, rocblas_int inc, rocblas_int batch_count)
        : data_(std::make_unique<PtrArrT[]>(batch_count))
        , n_(n)
        , inc_(inc)
        , batch_count_(batch_count)
    {
        assert(n > 0);
        assert(batch_count > 0);

        const size_t size = vsize();
        for(rocblas_int i = 0; i < batch_count; ++i)
        {
            data_[i] = std::make_unique<T[]>(size);
        }
    }

    host_batch_vector(rocblas_int n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count)
        : host_batch_vector(n, inc, batch_count)
    {
        assert(stride == 1);
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

    // The size of each vector. This is a derived property of the number of elements in the vector
    // and the spacing between them.
    size_t vsize() const
    {
        return size_t(n_) * std::abs(inc_);
    }

    // The number of vectors in the batch.
    rocblas_int batch_count() const noexcept
    {
        return batch_count_;
    }

    // Returns a vector from the batch.
    T* operator[](rocblas_int batch_index)
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);
        return data_[batch_index].get();
    }

    // Returns a vector from the batch.
    const T* operator[](rocblas_int batch_index) const
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);
        return data_[batch_index].get();
    }

    // Copy from a device_batch_vector into host memory.
    hipError_t transfer_from(const device_batch_vector<T>& that)
    {
        assert(n_ == that.n());
        assert(inc_ == that.inc());
        assert(batch_count_ == that.batch_count());

        hipError_t err = hipSuccess;
        host_batch_vector<T>& self = *this;
        size_t num_bytes = vsize() * sizeof(T);
        for(size_t b = 0; err == hipSuccess && b < batch_count_; ++b)
            err = hipMemcpy(self[b], that[b], num_bytes, hipMemcpyDeviceToHost);
        return err;
    }

private:
    using PtrArrT = std::unique_ptr<T[]>;

private:
    std::unique_ptr<PtrArrT[]> data_;
    rocblas_int n_;
    rocblas_int inc_;
    rocblas_int batch_count_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const host_batch_vector<T>& hbv)
{
    rocblas_int n = hbv.n();
    rocblas_int inc = std::abs(hbv.inc());
    rocblas_int batch_count = hbv.batch_count();

    for(rocblas_int b = 0; b < batch_count; ++b)
    {
        T* hv = hbv[b];
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
