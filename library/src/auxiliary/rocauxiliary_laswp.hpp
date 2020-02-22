/* ************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_LASWP_HPP
#define ROCLAPACK_LASWP_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"

template <typename T>
__device__ void swap(const rocblas_int n, T *a, const rocblas_int lda,
                               const rocblas_int i,
                               const rocblas_int exch) {

    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid < n) {
        T orig = a[i + lda * tid];
        a[i + lda * tid] = a[exch + lda * tid];
        a[exch + lda * tid] = orig;
    }
}

template <typename T, typename U>
__global__ void laswp_kernel(const rocblas_int n, U AA, const rocblas_int shiftA,
                            const rocblas_int lda, const rocblas_int stride, const rocblas_int i, const rocblas_int k1,
                            const rocblas_int *ipivA, const rocblas_int shiftP, const rocblas_int strideP, const rocblas_int incx) {

    int id = hipBlockIdx_y;

    //shiftP must be used so that ipiv[k1] is the desired first index of ipiv
    const rocblas_int *ipiv = ipivA + id*strideP + shiftP;
    rocblas_int exch = ipiv[k1 + (i - k1) * incx - 1];

    //will exchange rows i and exch if they are not the same
    if (exch != i) {
        T* A = load_ptr_batch(AA,shiftA,id,stride);
        swap(n,A,lda,i-1,exch-1);  //row indices are base-1 from the API
    }
}


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
