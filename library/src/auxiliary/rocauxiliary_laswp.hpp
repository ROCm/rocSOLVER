/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_LASWP_HPP
#define ROCLAPACK_LASWP_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "ideal_sizes.hpp"
#include "laswp_device.hpp"

template <typename T, typename U>
rocblas_status rocsolver_laswp_template(rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA,
                              const rocblas_int lda, const rocblas_int strideA, const rocblas_int k1, const rocblas_int k2,
                              const rocblas_int *ipiv, const rocblas_int shiftP, const rocblas_int strideP, rocblas_int incx, 
                              const rocblas_int batch_count) {
    // quick return
    if (n == 0 || !batch_count) 
        return rocblas_status_success;

    rocblas_int start, end, inc;
    if (incx < 0) {
        start = k2;
        end = k1 - 1;
        inc = -1;
        incx = -incx;
    } 
    else {
        start = k1;
        end = k2 + 1;
        inc = 1;
    }

    rocblas_int blocksPivot = (n - 1) / LASWP_BLOCKSIZE + 1;
    dim3 gridPivot(blocksPivot, batch_count, 1);
    dim3 threads(LASWP_BLOCKSIZE, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    for (rocblas_int i = start; i != end; i += inc) {
        hipLaunchKernelGGL(laswp_kernel<T>, gridPivot, threads, 0, stream, n, A, shiftA,
                           lda, strideA, i, k1, ipiv, shiftP, strideP, incx);
    }

    return rocblas_status_success;

}

#endif /* ROCLAPACK_LASWP_HPP */
