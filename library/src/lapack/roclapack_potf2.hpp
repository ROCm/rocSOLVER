/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_POTF2_HPP
#define ROCLAPACK_POTF2_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U> 
__global__ void sqrtDiagOnward(U A, const rocblas_int shiftA, const rocblas_int strideA, const size_t loc, 
                               const rocblas_int j, T *res, rocblas_int *info) 
{
    int id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A,shiftA,id,strideA);
    T t = M[loc] - res[id];

    // error for non-positive definiteness
    if (t <= 0.0) {
        if (info[id] == 0)
            info[id] = j + 1;   //use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    // minor is positive definite
    } else {
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}

template <typename T, typename U>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo, const rocblas_int n, U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_stride strideA,
                                        rocblas_int *info, const rocblas_int batch_count) 
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;
    
    //constants for rocblas functions calls
    T h_one = 1;
    T h_minone = -1;
    T *d_one, *d_minone;
    hipMalloc(&d_one, sizeof(T));
    hipMemcpy(d_one, &h_one, sizeof(T), hipMemcpyHostToDevice);
    hipMalloc(&d_minone, sizeof(T));
    hipMemcpy(d_minone, &h_minone, sizeof(T), hipMemcpyHostToDevice);

    // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
    // workspace on GPU (for reduction in rocblas_dot)
    size_t sizeW = sizeof(T) * ((n-1)/ROCBLAS_DOT_NB + 2) * batch_count;
    T* work;
    hipMalloc(&work, sizeW);
    //diagonal info in device (device memory workspace to avoid synchronization with CPU)
    T *pivotGPU; 
    hipMalloc(&pivotGPU, sizeof(T)*batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);
    T* M;

    //info=0 (starting with a positive definite matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);

    if (uplo == rocblas_fill_upper) { // Compute the Cholesky factorization A = U'*U.
        for (rocblas_int j = 0; j < n; ++j) {
            // Compute U(J,J) and test for non-positive-definiteness.
            rocblas_dot<false,T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                 A, shiftA + idx2D(0, j, lda), 1, strideA, batch_count, pivotGPU, work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, 
                               A, shiftA, strideA, idx2D(j, j, lda), j, pivotGPU, info);

            // Compute elements J+1:N of row J
            if (j < n - 1) {
                rocblas_gemv<T>(handle, rocblas_operation_transpose, j, n-j-1, d_minone, 0,
                                A, shiftA + idx2D(0, j+1, lda), lda, strideA,
                                A, shiftA + idx2D(0, j, lda), 1, strideA, d_one, 0,
                                A, shiftA + idx2D(j, j+1, lda), lda, strideA, batch_count);
                                    
                rocblas_scal<T>(handle, n-j-1, pivotGPU, 1, A, shiftA + idx2D(j, j+1, lda), lda, strideA, batch_count);
            }
        }

    } else { // Compute the Cholesky factorization A = L'*L.
        for (rocblas_int j = 0; j < n; ++j) {
            // Compute L(J,J) and test for non-positive-definiteness.
            rocblas_dot<false,T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                 A, shiftA + idx2D(j, 0, lda), lda, strideA, batch_count, pivotGPU, work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, 
                               A, shiftA, strideA, idx2D(j, j, lda), j, pivotGPU, info);

            // Compute elements J+1:N of row J
            if (j < n - 1) {
                rocblas_gemv<T>(handle, rocblas_operation_none, n-j-1, j, d_minone, 0,
                                A, shiftA + idx2D(j+1, 0, lda), lda, strideA,
                                A, shiftA + idx2D(j, 0, lda), lda, strideA, d_one, 0,
                                A, shiftA + idx2D(j+1, j, lda), 1, strideA, batch_count);

                rocblas_scal<T>(handle, n-j-1, pivotGPU, 1, A, shiftA + idx2D(j+1, j, lda), 1, strideA, batch_count);
            }
        }
    }

    hipFree(pivotGPU);
    hipFree(d_minone);
    hipFree(d_one);
    hipFree(work);

    return rocblas_status_success;
}

#endif /* ROCLAPACK_POTF2_HPP */
