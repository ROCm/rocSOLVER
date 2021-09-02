/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T>
void rocsolver_lasyf_getMemorySize(const rocblas_int n,
                                   const rocblas_int nb,
                                   const rocblas_int batch_count,
                                   size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || nb == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // size of workspace
    *size_work = sizeof(T) * n * nb * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        U kb,
                                        T A,
                                        U ipiv,
                                        U info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nb < 0 || nb > n || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((batch_count && !kb) || (n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        rocblas_int* kb,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* work)
{
    ROCSOLVER_ENTER("lasyf", "uplo:", uplo, "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    return rocblas_status_not_implemented;
}
