/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_BDSQR_H
#define ROCLAPACK_BDSQR_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"


/*
template <typename T>
void rocsolver_bdsqr_getMemorySize(const rocblas_int n, const rocblas_int nv, const rocblas_int nu, const rocblas_int nc, 
                                   const rocblas_int batch_count, size_t *size)
{
    // size of workspace
    *size = 0;
    if (nv) *size += 2;
    if (nu || nc) *size += 2;
    
    *size *= sizeof(T)*(n-1)*batch_count;
}
*/

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        TT* E,
                                        rocblas_int *info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if ((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none) ||
       (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none) ||
       (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if (n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if ((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if ((right_svect == rocblas_svect_all && ldv < n) || (right_svect == rocblas_svect_singular && ldv < min(m,n)))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n*m && !A) || (min(m,n) > 1 && !E) || (min(m,n) && !S) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if ((left_svect == rocblas_svect_all && m && !U) || (left_svect == rocblas_svect_singular && min(m,n) && !U))
        return rocblas_status_invalid_pointer;
    if ((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A, const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_stride strideA,
                                        TT* S, const rocblas_stride strideS,
                                        T* U, 
                                        const rocblas_int ldu, const rocblas_stride strideU,
                                        T* V, 
                                        const rocblas_int ldv, const rocblas_stride strideV,
                                        TT* E, const rocblas_stride strideE,
                                        rocblas_int *info,
                                        const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || m == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    return rocblas_status_success;
}

#endif /* ROCLAPACK_BDSQR_H */
