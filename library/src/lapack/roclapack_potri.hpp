/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_trtri.hpp"
#include "rocsolver.h"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_potri_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_tmpcopy,
                                   size_t* size_workArr)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_tmpcopy = 0;
        *size_workArr = 0;
        return;
    }

    // requirements for calling TRTRI
    rocsolver_trtri_getMemorySize<BATCHED, STRIDED, T>(rocblas_diagonal_non_unit, n, batch_count,
                                                       size_work1, size_work2, size_work3,
                                                       size_work4, size_tmpcopy, size_workArr);
}

template <typename T>
rocblas_status rocsolver_potri_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocsolver_potri_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* tmpcopy,
                                        T** workArr,
                                        const bool optim_mem)
{
    ROCSOLVER_ENTER("potri", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    return rocblas_status_not_implemented;
}
