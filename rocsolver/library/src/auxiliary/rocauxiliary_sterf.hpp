/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_STERF_HPP
#define ROCLAPACK_STERF_HPP

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_sterf_argCheck(const rocblas_int n, T D, T E, rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_sterf_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocblas_status_not_implemented;
}

#endif
