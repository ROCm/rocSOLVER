/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_STEQR_HPP
#define ROCLAPACK_STEQR_HPP

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename S, typename T>
rocblas_status rocsolver_steqr_argCheck(const rocblas_evect compc,
                                        const rocblas_int n,
                                        S D,
                                        S E,
                                        T C,
                                        const rocblas_int ldc,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(compc != rocblas_evect_none && compc != rocblas_evect_tridiagonal
       && compc != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(compc != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || (compc != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename S, typename T, typename U>
rocblas_status rocsolver_steqr_template(rocblas_handle handle,
                                        const rocblas_evect compc,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocblas_status_not_implemented;
}

#endif
