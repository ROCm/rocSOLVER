/************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver.h"

template <bool BATCHED, typename T>
void rocsolver_getri_outofplace_getMemorySize(const rocblas_int n,
                                              const rocblas_int batch_count,
                                              size_t* size_work1,
                                              size_t* size_work2,
                                              size_t* size_work3,
                                              size_t* size_work4,
                                              bool* optim_mem)
{
    // if quick return, no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *optim_mem = true;
        return;
    }

    // requirements for calling GETRS
    rocsolver_getrs_getMemorySize<BATCHED, T>(n, n, batch_count, size_work1, size_work2, size_work3,
                                              size_work4, optim_mem);
}

template <typename T>
rocblas_status rocsolver_getri_outofplace_argCheck(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   const rocblas_int lda,
                                                   const rocblas_int ldc,
                                                   T A,
                                                   T C,
                                                   rocblas_int* ipiv,
                                                   rocblas_int* info,
                                                   const bool pivot,
                                                   const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || ldc < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !C) || (n && pivot && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocsolver_getri_outofplace_template(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   U A,
                                                   const rocblas_int shiftA,
                                                   const rocblas_int lda,
                                                   const rocblas_stride strideA,
                                                   rocblas_int* ipiv,
                                                   const rocblas_int shiftP,
                                                   const rocblas_stride strideP,
                                                   U C,
                                                   const rocblas_int shiftC,
                                                   const rocblas_int ldc,
                                                   const rocblas_stride strideC,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count,
                                                   void* work1,
                                                   void* work2,
                                                   void* work3,
                                                   void* work4,
                                                   const bool optim_mem,
                                                   const bool pivot)
{
    ROCSOLVER_ENTER("getri_outofplace", "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        rocblas_int blocks = (batch_count - 1) / 32 + 1;
        hipLaunchKernelGGL(reset_info, dim3(blocks, 1, 1), dim3(32, 1, 1), 0, stream, info,
                           batch_count, 0);
        return rocblas_status_success;
    }

    // check for singularities
    hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, BLOCKSIZE, 1), 0,
                       stream, n, A, shiftA, lda, strideA, info);

    // initialize C to the identity
    rocblas_int blocks = (n - 1) / 32 + 1;
    hipLaunchKernelGGL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0, stream, n,
                       n, C, shiftC, ldc, strideC);

    // compute inverse
    rocsolver_getrs_template<BATCHED, T>(handle, rocblas_operation_none, n, n, A, shiftA, lda,
                                         strideA, ipiv, strideP, C, shiftC, ldc, strideC,
                                         batch_count, work1, work2, work3, work4, optim_mem, pivot);

    return rocblas_status_success;
}
