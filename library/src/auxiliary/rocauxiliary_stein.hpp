/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename S>
rocblas_status rocsolver_stein_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        S* D,
                                        S* E,
                                        rocblas_int* nev,
                                        S* W,
                                        rocblas_int* iblock,
                                        rocblas_int* isplit,
                                        T* Z,
                                        const rocblas_int ldz,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || ldz < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || !nev || (n && !W) || (n && !iblock) || (n && !isplit) || (n && !Z)
       || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_stein_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* nev,
                                        S* W,
                                        const rocblas_int shiftW,
                                        const rocblas_stride strideW,
                                        rocblas_int* iblock,
                                        const rocblas_stride strideIblock,
                                        rocblas_int* isplit,
                                        const rocblas_stride strideIsplit,
                                        U Z,
                                        const rocblas_int shiftZ,
                                        const rocblas_int ldz,
                                        const rocblas_stride strideZ,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("stein", "n:", n, "shiftD:", shiftD, "shiftE:", shiftE, "shiftW:", shiftW,
                    "shiftZ:", shiftZ, "ldz:", ldz, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
