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

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U> 
__global__ void sqrtDiagOnward(U A, const rocblas_int shiftA, const rocblas_int strideA, const size_t loc, 
                               const rocblas_int j, T *res, rocblas_int *info) 
{
    int id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A,id,shiftA,strideA);
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

template <typename T>
void rocsolver_potf2_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // size of scalars (constants)
    *size_1 = sizeof(T)*3;

    // size of workspace
    *size_2 = sizeof(T) * ((n-1)/ROCBLAS_DOT_NB + 2) * batch_count;

    // size of array of pivots
    *size_3 = sizeof(T)*batch_count;
}


template <typename T, typename U>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo, const rocblas_int n, U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_stride strideA,
                                        rocblas_int *info, const rocblas_int batch_count,
                                        T*scalars, T* work, T* pivotGPU) 
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;
    
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    //info=0 (starting with a positive definite matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);

    if (uplo == rocblas_fill_upper) { // Compute the Cholesky factorization A = U'*U.
        for (rocblas_int j = 0; j < n; ++j) {
            // Compute U(J,J) and test for non-positive-definiteness.
            rocblasCall_dot<false,T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                 A, shiftA + idx2D(0, j, lda), 1, strideA, batch_count, pivotGPU, work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, 
                               A, shiftA, strideA, idx2D(j, j, lda), j, pivotGPU, info);

            // Compute elements J+1:N of row J
            if (j < n - 1) {
                rocblasCall_gemv<T>(handle, rocblas_operation_transpose, j, n-j-1, scalars, 0,
                                A, shiftA + idx2D(0, j+1, lda), lda, strideA,
                                A, shiftA + idx2D(0, j, lda), 1, strideA, scalars+2, 0,
                                A, shiftA + idx2D(j, j+1, lda), lda, strideA, batch_count, nullptr);
                                    
                rocblasCall_scal<T>(handle, n-j-1, pivotGPU, 1, A, shiftA + idx2D(j, j+1, lda), lda, strideA, batch_count);
            }
        }

    } else { // Compute the Cholesky factorization A = L'*L.
        for (rocblas_int j = 0; j < n; ++j) {
            // Compute L(J,J) and test for non-positive-definiteness.
            rocblasCall_dot<false,T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                 A, shiftA + idx2D(j, 0, lda), lda, strideA, batch_count, pivotGPU, work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, 
                               A, shiftA, strideA, idx2D(j, j, lda), j, pivotGPU, info);

            // Compute elements J+1:N of row J
            if (j < n - 1) {
                rocblasCall_gemv<T>(handle, rocblas_operation_none, n-j-1, j, scalars, 0,
                                A, shiftA + idx2D(j+1, 0, lda), lda, strideA,
                                A, shiftA + idx2D(j, 0, lda), lda, strideA, scalars+2, 0,
                                A, shiftA + idx2D(j+1, j, lda), 1, strideA, batch_count, nullptr);

                rocblasCall_scal<T>(handle, n-j-1, pivotGPU, 1, A, shiftA + idx2D(j+1, j, lda), 1, strideA, batch_count);
            }
        }
    }

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_POTF2_HPP */
