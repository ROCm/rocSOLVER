/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 *
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_lasyf.hpp"
#include "rocblas.hpp"
#include "roclapack_sytf2.hpp"
#include "rocsolver.h"

template <typename T>
void rocsolver_sytrf_getMemorySize(const rocblas_int n, const rocblas_int batch_count, size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // size of workspace
    if(n <= SYTRF_BLOCKSIZE)
        rocsolver_lasyf_getMemorySize<T>(n, SYTRF_BLOCKSIZE, batch_count, size_work);
    else
        *size_work = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_sytrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
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
    ROCSOLVER_ENTER("sytrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    return rocblas_status_not_implemented;
}
