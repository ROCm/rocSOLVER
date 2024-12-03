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

template <typename T, typename U>
struct rocsolver_hybrid_array
{
    rocblas_int dim, batch_count;
    rocblas_stride stride;

    U src_array;
    T** batch_array;
    T* curr_array;

    rocsolver_hybrid_array()
        : src_array(nullptr)
        , batch_array(nullptr)
        , curr_array(nullptr)
    {
    }
    ~rocsolver_hybrid_array()
    {
        if(curr_array)
        {
            free(curr_array);
            if(batch_array)
                free(batch_array);
        }
    }

    rocblas_status init_async(rocblas_int dim,
                              U array,
                              rocblas_stride stride,
                              rocblas_int batch_count,
                              hipStream_t stream)
    {
        if(curr_array)
        {
            free(curr_array);
            if(batch_array)
                free(batch_array);
        }

        this->dim = dim;
        this->src_array = array;
        this->stride = stride;
        this->batch_count = batch_count;

        bool constexpr is_strided = (std::is_same<U, T*>::value || std::is_same<U, T* const>::value);
        bool is_device = is_device_pointer(array);

        if(is_device)
        {
            size_t bytes = sizeof(T) * dim;
            curr_array = (T*)malloc(bytes);

            if(is_strided)
                batch_array = nullptr;
            else
            {
                bytes = sizeof(T*) * batch_count;
                batch_array = (T**)malloc(bytes);
                HIP_CHECK(hipMemcpyAsync(batch_array, array, bytes, hipMemcpyDeviceToHost, stream));
            }
        }
        else
        {
            curr_array = nullptr;

            if(is_strided)
                batch_array = nullptr;
            else
                batch_array = (T**)src_array;
        }

        return rocblas_status_success;
    }

    rocblas_status get_from_device_async(T** dst, rocblas_int bid, hipStream_t stream)
    {
        if(!src_array)
            return rocblas_status_internal_error;

        if(curr_array)
        {
            *dst = curr_array;
            size_t bytes = sizeof(T) * dim;

            if(batch_array)
            {
                HIP_CHECK(hipMemcpyAsync(curr_array, batch_array[bid], bytes, hipMemcpyDeviceToHost,
                                         stream));
            }
            else
            {
                HIP_CHECK(hipMemcpyAsync(curr_array, src_array + bid * stride, bytes,
                                         hipMemcpyDeviceToHost, stream));
            }
        }
        else
        {
            if(batch_array)
            {
                *dst = batch_array[bid];
            }
            else
            {
                *dst = src_array + bid * stride;
            }
        }

        return rocblas_status_success;
    }
    rocblas_status push_to_device_async(rocblas_int bid, hipStream_t stream)
    {
        if(!src_array)
            return rocblas_status_internal_error;

        if(curr_array)
        {
            size_t bytes = sizeof(T) * dim;

            if(batch_array)
            {
                HIP_CHECK(hipMemcpyAsync(batch_array[bid], curr_array, bytes, hipMemcpyHostToDevice,
                                         stream));
            }
            else
            {
                HIP_CHECK(hipMemcpyAsync(src_array + bid * stride, curr_array, bytes,
                                         hipMemcpyHostToDevice, stream));
            }
        }

        return rocblas_status_success;
    }
};

ROCSOLVER_END_NAMESPACE
