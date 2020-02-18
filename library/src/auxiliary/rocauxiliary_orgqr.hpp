/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_ORGQR_HPP
#define ROCLAPACK_ORGQR_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"

template <typename T, typename U>
rocblas_status rocsolver_orgqr_template(rocsolver_handle handle, const rocsolver_int m, 
                                   const rocsolver_int n, const rocsolver_int k, U A, const rocblas_int shiftA, 
                                   const rocsolver_int lda, const rocsolver_int strideA, const T* ipiv, 
                                   const rocsolver_int strideP, const rocsolver_int batch_count)
{
    // quick return
    if (!n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
printf("This is ORGQR...");   
 
    return rocblas_status_success;
}

#endif
