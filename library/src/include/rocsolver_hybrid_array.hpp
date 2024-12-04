/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common_host_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
struct rocsolver_hybrid_array
{
    I dim, batch_count;
    rocblas_stride stride;

    U src_array;
    T** batch_array;
    T* val_array;

    rocsolver_hybrid_array()
        : src_array(nullptr)
        , batch_array(nullptr)
        , val_array(nullptr)
    {
    }
    ~rocsolver_hybrid_array()
    {
        if(val_array)
            free(val_array);
        if(batch_array && (val_array || this->dim < 0))
            free(batch_array);
    }

    /* Initializes internal arrays to hold pointer data. Primarily used to read device pointers
       from a batched array for use on the host; no other data is read from the device. */
    rocblas_status init_pointers_only(U array, rocblas_stride stride, I batch_count, hipStream_t stream)
    {
        if(val_array)
            free(val_array);
        if(batch_array && (val_array || this->dim < 0))
            free(batch_array);

        this->dim = -1;
        this->src_array = array;
        this->stride = stride;
        this->batch_count = batch_count;

        bool constexpr is_strided = (std::is_same<U, T*>::value || std::is_same<U, T* const>::value);
        bool is_device = is_device_pointer((void*)array);

        if(is_device)
        {
            // pointers only; don't allocate val_array
            val_array = nullptr;

            if(is_strided)
            {
                // data is strided; batch_array not needed
                batch_array = nullptr;
            }
            else
            {
                // data is batched; read device pointers into batch_array
                size_t batch_bytes = sizeof(T*) * batch_count;
                batch_array = (T**)malloc(batch_bytes);
                HIP_CHECK(
                    hipMemcpyAsync(batch_array, array, batch_bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
            }
        }
        else
        {
            // data on host; use src_array directly
            val_array = nullptr;

            if(is_strided)
                batch_array = nullptr;
            else
                batch_array = (T**)src_array;
        }

        return rocblas_status_success;
    }
    /* Initializes internal arrays to hold data from the device. Device pointers are read into batch_array
       (if applicable), and matrix data is read into val_array. */
    rocblas_status init_async(I dim, U array, rocblas_stride stride, I batch_count, hipStream_t stream)
    {
        if(val_array)
            free(val_array);
        if(batch_array && (val_array || this->dim < 0))
            free(batch_array);

        if(dim < 0)
            return rocblas_status_internal_error;

        this->dim = dim;
        this->src_array = array;
        this->stride = stride;
        this->batch_count = batch_count;

        bool constexpr is_strided = (std::is_same<U, T*>::value || std::is_same<U, T* const>::value);
        bool is_device = is_device_pointer((void*)array);

        if(is_device)
        {
            // allocate space on host for data from device
            size_t dim_bytes = sizeof(T) * dim;
            size_t val_bytes = sizeof(T) * dim * batch_count;
            val_array = (T*)malloc(val_bytes);

            if(is_strided)
            {
                // data is strided; batch_array not needed
                batch_array = nullptr;

                // read data to val_array
                if(batch_count == 1 || stride == dim)
                {
                    HIP_CHECK(hipMemcpyAsync(val_array, src_array, val_bytes, hipMemcpyDeviceToHost,
                                             stream));
                }
                else
                {
                    for(I bid = 0; bid < batch_count; bid++)
                    {
                        HIP_CHECK(hipMemcpyAsync(val_array + bid * dim, src_array + bid * stride,
                                                 dim_bytes, hipMemcpyDeviceToHost, stream));
                    }
                }
            }
            else
            {
                // data is batched; read device pointers into batch_array
                size_t batch_bytes = sizeof(T*) * batch_count;
                batch_array = (T**)malloc(batch_bytes);
                HIP_CHECK(
                    hipMemcpyAsync(batch_array, array, batch_bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                // read data to val_array
                for(I bid = 0; bid < batch_count; bid++)
                {
                    HIP_CHECK(hipMemcpyAsync(val_array + bid * dim, batch_array[bid], dim_bytes,
                                             hipMemcpyDeviceToHost, stream));
                }
            }
        }
        else
        {
            // data on host; use src_array directly
            val_array = nullptr;

            if(is_strided)
                batch_array = nullptr;
            else
                batch_array = (T**)src_array;
        }

        return rocblas_status_success;
    }
    /* Copies data from val_array back to the device. Returns an error if initialized for pointers
       only. */
    rocblas_status push_to_device_async(hipStream_t stream)
    {
        if(!src_array)
            return rocblas_status_internal_error;
        if(dim < 0)
            return rocblas_status_internal_error;

        if(val_array)
        {
            size_t dim_bytes = sizeof(T) * dim;
            size_t val_bytes = sizeof(T) * dim * batch_count;

            if(!batch_array)
            {
                if(batch_count == 1 || stride == dim)
                {
                    HIP_CHECK(hipMemcpyAsync(src_array, val_array, val_bytes, hipMemcpyHostToDevice,
                                             stream));
                }
                else
                {
                    for(I bid = 0; bid < batch_count; bid++)
                    {
                        HIP_CHECK(hipMemcpyAsync(src_array + bid * stride, val_array + bid * dim,
                                                 dim_bytes, hipMemcpyHostToDevice, stream));
                    }
                }
            }
            else
            {
                for(I bid = 0; bid < batch_count; bid++)
                {
                    HIP_CHECK(hipMemcpyAsync(batch_array[bid], val_array + bid * dim, dim_bytes,
                                             hipMemcpyHostToDevice, stream));
                }
            }
        }

        return rocblas_status_success;
    }

    /* Gets a pointer to the data for batch index bid. If initialized for pointers only, the
       returned pointer may be on the device. Otherwise, the pointer is on the host. */
    T* operator[](I bid)
    {
        if(!src_array)
            return nullptr;

        if(val_array)
            return val_array + bid * dim;
        else
        {
            if(batch_array)
                return batch_array[bid];
            else
                return (T*)(src_array + bid * stride);
        }
    }
};

ROCSOLVER_END_NAMESPACE
