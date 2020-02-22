/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_POTRF_HPP
#define ROCLAPACK_POTRF_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "definitions.h"
#include "helpers.h"
#include "common_device.hpp"
#include "ideal_sizes.hpp"
#include "roclapack_potf2.hpp"

inline __global__ void chk_positive(rocblas_int *iinfo, rocblas_int *info, int j) 
{
    int id = hipBlockIdx_x;

    if (info[id] == 0 && iinfo[id] > 0)
            info[id] = iinfo[id] + j;   
}

template <typename T, typename U>
rocblas_status rocsolver_potrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo, const rocblas_int n, U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_int strideA,
                                        rocblas_int *info, const rocblas_int batch_count) 
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the algorithm
    if (n < POTRF_POTF2_SWITCHSIZE) 
        return rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info, batch_count);

    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    //constants for rocblas functions calls
    T h_one = 1;
    T h_minone = -1;
    T *d_one, *d_minone;
    hipMalloc(&d_one, sizeof(T));
    hipMemcpy(d_one, &h_one, sizeof(T), hipMemcpyHostToDevice);
    hipMalloc(&d_minone, sizeof(T));
    hipMemcpy(d_minone, &h_minone, sizeof(T), hipMemcpyHostToDevice);

    //info in device (device memory workspace to avoid synchronization with CPU)
    rocblas_int *iinfo; 
    hipMalloc(&iinfo, sizeof(rocblas_int)*batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);
    T* M;
    rocblas_int jb;

    //info=0 (starting with a positive definite matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);

    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****

    if (uplo == rocblas_fill_upper) { // Compute the Cholesky factorization A = U'*U.
        for (rocblas_int j = 0; j < n; j += POTRF_POTF2_SWITCHSIZE) {
            // Factor diagonal and subdiagonal blocks 
            jb = min(n - j, POTRF_POTF2_SWITCHSIZE);  //number of columns in the block
            hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,iinfo,batch_count,0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo, batch_count);
            
            // test for non-positive-definiteness.
            hipLaunchKernelGGL(chk_positive,gridReset,threads,0,stream,iinfo,info,j);
            
            if (j + jb < n) {
                // update trailing submatrix
                for (int b=0;b<batch_count;++b) {
                    M = load_ptr_batch<T>(AA,shiftA,b,strideA);
                    rocblas_trsm(handle, rocblas_side_left, uplo, rocblas_operation_transpose,
                             rocblas_diagonal_non_unit, jb, (n - j - jb), d_one,
                             (M + idx2D(j, j, lda)), lda, (M + idx2D(j, j + jb, lda)), lda);
                }

                // *** GEMM MUST BE REPLACED BY SYRK ONCE IT IS AVAILABLE IN ROCBLAS ****                
                for (int b=0;b<batch_count;++b) {
                    M = load_ptr_batch<T>(AA,shiftA,b,strideA);
                    rocblas_gemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                                 (n - j - jb), (n - j - jb), jb, d_minone,
                                 (M + idx2D(j, j + jb, lda)), lda, (M + idx2D(j, j + jb, lda)),
                                 lda, d_one,
                                 (M + idx2D(j + jb, j + jb, lda)), lda);
                }
            }
        }

    } else { // Compute the Cholesky factorization A = L'*L.
        for (rocblas_int j = 0; j < n; j += POTRF_POTF2_SWITCHSIZE) {
            // Factor diagonal and subdiagonal blocks 
            jb = min(n - j, POTRF_POTF2_SWITCHSIZE);  //number of columns in the block
            hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,iinfo,batch_count,0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo, batch_count);
            
            // test for non-positive-definiteness.
            hipLaunchKernelGGL(chk_positive,gridReset,threads,0,stream,iinfo,info,j);
            
            if (j + jb < n) {
                // update trailing submatrix
                for (int b=0;b<batch_count;++b) {
                    M = load_ptr_batch<T>(AA,shiftA,b,strideA);
                    rocblas_trsm(handle, rocblas_side_right, uplo, rocblas_operation_transpose,
                             rocblas_diagonal_non_unit, (n - j - jb), jb, d_one,
                             (M + idx2D(j, j, lda)), lda, (M + idx2D(j + jb, j, lda)), lda);
                }

                // *** GEMM MUST BE REPLACED BY SYRK ONCE IT IS AVAILABLE IN ROCBLAS ****                
                for (int b=0;b<batch_count;++b) {
                    M = load_ptr_batch<T>(AA,shiftA,b,strideA);
                    rocblas_gemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                                 (n - j - jb), (n - j - jb), jb, d_minone,
                                 (M + idx2D(j + jb, j, lda)), lda, (M + idx2D(j + jb, j, lda)),
                                 lda, d_one,
                                 (M + idx2D(j + jb, j + jb, lda)), lda);
                }
            }
        }
    }

    hipFree(iinfo);
    hipFree(d_minone);
    hipFree(d_one);

    return rocblas_status_success;
}

#endif /* ROCLAPACK_POTRF_HPP */
