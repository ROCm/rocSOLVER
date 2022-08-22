/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.10.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

// Helper to calculate workspace size requirements
template <typename T>
void rocsolver_bdsvdx_getMemorySize(const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work = 0;
        return;
    }

    // to be completed
    *size_work = 0;
}

// Helper to check argument correctnesss
template <typename T, typename U>
rocblas_status rocsolver_bdsvdx_argCheck(rocblas_handle handle,
                                         const rocblas_fill uplo,
                                         const rocblas_svect svect,
                                         const rocblas_srange srange,
                                         const rocblas_int n,
                                         T* D,
                                         T* E,
                                         const T vl,
                                         const T vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         T* S,
                                         U Z,
                                         const rocblas_int ldz,
                                         rocblas_int* ifail,
                                         rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if(svect != rocblas_svect_none && svect != rocblas_svect_singular)
        return rocblas_status_invalid_value;
    if(srange != rocblas_srange_all && srange != rocblas_srange_value
       && srange != rocblas_srange_index)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if((svect == rocblas_svect_none && ldz < 1) || (svect != rocblas_svect_none && ldz < 2 * n))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !S)) || (n > 1 && !E) || !info || !nsv)
        return rocblas_status_invalid_pointer;
    if(svect != rocblas_svect_none && n && (!Z || !ifail))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// bdsvdx template function implementation
template <typename T, typename U>
rocblas_status rocsolver_bdsvdx_template(rocblas_handle handle,
                                         const rocblas_fill uplo,
                                         const rocblas_svect svect,
                                         const rocblas_srange srange,
                                         const rocblas_int n,
                                         T* D,
                                         const rocblas_stride strideD,
                                         T* E,
                                         const rocblas_stride strideE,
                                         const T vl,
                                         const T vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const T abstol,
                                         rocblas_int* nsv,
                                         T* S,
                                         const rocblas_stride strideS,
                                         U Z,
                                         const rocblas_int shiftZ,
                                         const rocblas_int ldz,
                                         const rocblas_stride strideZ,
                                         rocblas_int* ifail,
                                         const rocblas_stride strideIfail,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         void* work)
{
    ROCSOLVER_ENTER("bdsvdx", "uplo:", uplo, "svect:", svect, "srange:", srange, "n:", n, "vl:", vl,
                    "vu:", vu, "il:", il, "iu:", iu, "abstol:", abstol, "ldz:", ldz,
                    "shiftZ:", shiftZ, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
