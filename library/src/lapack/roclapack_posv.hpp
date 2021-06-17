/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_potrf.hpp"
#include "roclapack_potrs.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_posv_argCheck(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       T B,
                                       rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (nrhs * n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_posv_getMemorySize(const rocblas_int n,
                                  const rocblas_int nrhs,
                                  const rocblas_fill uplo,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work1,
                                  size_t* size_work2,
                                  size_t* size_work3,
                                  size_t* size_work4,
                                  size_t* size_pivots,
                                  size_t* size_iinfo)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots = 0;
        *size_iinfo = 0;
        return;
    }

    size_t w1, w2, w3, w4;

    // workspace required for potrf
    rocsolver_potrf_getMemorySize<BATCHED, T>(n, uplo, batch_count, size_scalars, size_work1,
                                              size_work2, size_work3, size_work4, size_pivots,
                                              size_iinfo);

    // workspace required for potrs
    rocsolver_potrs_getMemorySize<BATCHED, T>(n, nrhs, batch_count, &w1, &w2, &w3, &w4);

    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocsolver_posv_template(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       U B,
                                       const rocblas_int shiftB,
                                       const rocblas_int ldb,
                                       const rocblas_stride strideB,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       void* work1,
                                       void* work2,
                                       void* work3,
                                       void* work4,
                                       T* pivots,
                                       rocblas_int* iinfo,
                                       bool optim_mem)
{
    ROCSOLVER_ENTER("posv", "uplo:", uplo, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
