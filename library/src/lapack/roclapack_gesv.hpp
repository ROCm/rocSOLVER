/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_gesv_argCheck(rocblas_handle handle,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       T B,
                                       const rocblas_int* ipiv,
                                       const rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (nrhs * n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_gesv_getMemorySize(const rocblas_int n,
                                  const rocblas_int nrhs,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work,
                                  size_t* size_work1,
                                  size_t* size_work2,
                                  size_t* size_work3,
                                  size_t* size_work4,
                                  size_t* size_pivotval,
                                  size_t* size_pivotidx,
                                  size_t* size_iinfo)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iinfo = 0;
        return;
    }

    size_t w1, w2, w3, w4;

    // workspace required for calling GETRF
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, true, T, S>(
        n, n, batch_count, size_scalars, size_work, size_work1, size_work2, size_work3, size_work4,
        size_pivotval, size_pivotidx, size_iinfo);

    // workspace required for calling GETRS
    rocsolver_getrs_getMemorySize<BATCHED, T>(n, nrhs, batch_count, &w1, &w2, &w3, &w4);

    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gesv_template(rocblas_handle handle,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       rocblas_int* ipiv,
                                       const rocblas_stride strideP,
                                       U B,
                                       const rocblas_int shiftB,
                                       const rocblas_int ldb,
                                       const rocblas_stride strideB,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       rocblas_index_value_t<S>* work,
                                       void* work1,
                                       void* work2,
                                       void* work3,
                                       void* work4,
                                       T* pivotval,
                                       rocblas_int* pivotidx,
                                       rocblas_int* iinfo,
                                       bool optim_mem)
{
    ROCSOLVER_ENTER("gesv", "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
