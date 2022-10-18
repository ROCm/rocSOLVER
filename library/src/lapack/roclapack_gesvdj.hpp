/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/** Argument checking **/
template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesvdj_argCheck(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_int lda,
                                         SS* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         SS* S,
                                         T* U,
                                         const rocblas_int ldu,
                                         T* V,
                                         const rocblas_int ldv,
                                         rocblas_int* info,
                                         const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(left_svect != rocblas_svect_singular && left_svect != rocblas_svect_none)
        return rocblas_status_invalid_value;
    if(right_svect != rocblas_svect_singular && right_svect != rocblas_svect_none)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || max_sweeps <= 0 || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(left_svect == rocblas_svect_singular && ldu < m)
        return rocblas_status_invalid_size;
    if(right_svect == rocblas_svect_singular && ldv < min(m, n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n * m && !A) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (min(m, n) && !S) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if(left_svect == rocblas_svect_singular && min(m, n) && !U)
        return rocblas_status_invalid_pointer;
    if(right_svect == rocblas_svect_singular && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename SS>
void rocsolver_gesvdj_getMemorySize(const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_scalars)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        return;
    }

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // to be implemented
}

template <bool BATCHED, bool STRIDED, typename T, typename SS, typename W>
rocblas_status rocsolver_gesvdj_template(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         const SS abstol,
                                         SS* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         SS* S,
                                         const rocblas_stride strideS,
                                         T* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         T* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         T* scalars)
{
    ROCSOLVER_ENTER("gesvdj", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps,
                    "ldu:", ldu, "ldv:", ldv, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
