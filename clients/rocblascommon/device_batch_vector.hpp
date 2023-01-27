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

#include "common_host_helpers.hpp"
#include "device_memory.hpp"

template <typename T>
class host_batch_vector;

template <typename T, size_t PAD = 0, typename U = T>
class device_batch_vector
{
public:
    device_batch_vector(rocblas_int n, rocblas_int inc, rocblas_int batch_count)
        : hPtrArr_(std::make_unique<PtrDArrT[]>(batch_count))
        , n_(n)
        , inc_(inc)
        , batch_count_(batch_count)
    {
        assert(n > 0);
        assert(batch_count > 0);

        T** dPtrArr;
        THROW_IF_HIP_ERROR(hipMalloc(&dPtrArr, sizeof(T*) * batch_count));
        dPtrArr_ = std::unique_ptr<T*[], device_deleter>(dPtrArr);

        auto tmp = std::make_unique<T*[]>(batch_count);
        const size_t size = vsize();
        for(rocblas_int i = 0; i < batch_count; ++i)
        {
            T* dArr;
            THROW_IF_HIP_ERROR(hipMalloc(&dArr, sizeof(T) * size));
            hPtrArr_[i].reset(dArr);
            tmp[i] = dArr;
        }
        THROW_IF_HIP_ERROR(hipMemcpy(dPtrArr, tmp.get(), sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    }

    device_batch_vector(rocblas_int n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count)
        : device_batch_vector(n, inc, batch_count)
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

    T* const* data()
    {
        return dPtrArr_.get();
    }

    const T* const* data() const
    {
        return dPtrArr_.get();
    }
/*
    T* const* ddata()
    {
        return dPtrArr_;
    }

    const T* const* ddata() const
    {
        return dPtrArr_;
    }

    T* const* hdata()
    {
        return hPtrArr_;
    }

    const T* const* hdata() const
    {
        return hPtrArr_;
    }
*/
    T* operator[](rocblas_int batch_index)
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);
        return hPtrArr_[batch_index].get();
    }

    const T* operator[](rocblas_int batch_index) const
    {
        assert(batch_index >= 0);
        assert(batch_index < batch_count_);
        return hPtrArr_[batch_index].get();
    }

    operator const T* const *() const
    {
        return hPtrArr_;
    }

    // clang-format off
    operator T**()
    {
        return hPtrArr_;
    }
    // clang-format on

    explicit operator bool() const
    {
        return nullptr != hPtrArr_;
    }

    hipError_t transfer_from(const host_batch_vector<T>& that)
    {
        assert(n_ == that.n());
        assert(inc_ == that.inc());
        assert(batch_count_ == that.batch_count());

        hipError_t err = hipSuccess;
        device_batch_vector<T, PAD, U>& self = *this;
        size_t num_bytes = vsize() * sizeof(T);
        for(size_t b = 0; err == hipSuccess && b < batch_count_; ++b)
            err = hipMemcpy(self[b], that[b], num_bytes, hipMemcpyHostToDevice);
        return err;
    }

    hipError_t memcheck() const
    {
        return hipSuccess;
    }

private:
    using PtrDArrT = std::unique_ptr<T[], device_deleter>;

private:
    std::unique_ptr<PtrDArrT[]> hPtrArr_;
    std::unique_ptr<T*[], device_deleter> dPtrArr_;
    rocblas_int n_;
    rocblas_int inc_;
    rocblas_int batch_count_;
};
