/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_BDSQR_H
#define ROCLAPACK_BDSQR_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename W1, typename W2>
rocblas_status rocsolver_bdsqr_argCheck(const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nv,
                                       const rocblas_int nu,
                                       const rocblas_int nc,
                                       const rocblas_int ldv,
                                       const rocblas_int ldu,
                                       const rocblas_int ldc,
                                       W1   D,
                                       W1   E,
                                       W2   V,
                                       W2   U,
                                       W2   C,
                                       rocblas_int *info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if (n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if ((nv > 0 && ldv < n) || (nc > 0 && ldc < n))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !D) || (n > 1 && !E) || (n*nv && !V) || (n*nu && !U) || (n*nc && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename W1, typename W2, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nv,
                                           const rocblas_int nu,
                                           const rocblas_int nc,
                                           W1 D, const rocblas_stride strideD,
                                           W1 E, const rocblas_stride strideE,
                                           W2 V, const rocblas_int shiftV,
                                           const rocblas_int ldv, const rocblas_stride strideV,
                                           W2 U, const rocblas_int shiftU,
                                           const rocblas_int ldu, const rocblas_stride strideU,
                                           W2 C, const rocblas_int shiftC,
                                           const rocblas_int ldc, const rocblas_stride strideC,
                                           rocblas_int *info,
                                           const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

/*
    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device); 


    rocblas_set_pointer_mode(handle,old_mode);  
*/
    return rocblas_status_success;
}

#endif /* ROCLAPACK_BDSQR_H */
