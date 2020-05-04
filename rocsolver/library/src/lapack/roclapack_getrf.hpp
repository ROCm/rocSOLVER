/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRF_HPP
#define ROCLAPACK_GETRF_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "roclapack_getf2.hpp"
#include "../auxiliary/rocauxiliary_laswp.hpp"

template<typename U>
__global__ void getrf_check_singularity(const rocblas_int n, const rocblas_int j, rocblas_int *ipivA, const rocblas_int shiftP,
                                const rocblas_stride strideP, const rocblas_int *iinfo, rocblas_int *info) {
    int id = hipBlockIdx_y;

    rocblas_int *ipiv = ipivA + id*strideP + shiftP;

    if (info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;

    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid < n)
        ipiv[tid] += j;
}

template <typename T>
void rocsolver_getrf_getMemorySize(const rocblas_int n, const rocblas_int m, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    rocsolver_getf2_getMemorySize<T>(batch_count,size_1,size_2);
    if (m < GETRF_GETF2_SWITCHSIZE || n < GETRF_GETF2_SWITCHSIZE) {
        *size_3 = 0;
    } else {
        *size_3 = sizeof(rocblas_int)*batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getrf_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        rocblas_int *ipiv, const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int *info, const rocblas_int batch_count,
                                        T* scalars, T* pivotGPU, rocblas_int* iinfo)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the algorithm
    if (m < GETRF_GETF2_SWITCHSIZE || n < GETRF_GETF2_SWITCHSIZE) 
        return rocsolver_getf2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, scalars, pivotGPU);

    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      TRSM_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif
    
    //constants to use when calling rocablas functions
    T one = 1;                    //constant 1 in host
    T minone = -1;                //constant -1 in host

    rocblas_int blocksPivot;
    rocblas_int blocksReset = (batch_count - 1) / GETF2_BLOCKSIZE + 1;
    dim3 gridPivot;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(GETF2_BLOCKSIZE, 1, 1);
    rocblas_int dim = min(m, n);    //total number of pivots
    T* M;
    rocblas_int jb, sizePivot;

    //info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);

    // **** TRSM_BATCH IS EXECUTED IN A FOR-LOOP UNTIL 
    //      FUNCITONALITY IS ENABLED. ****

    for (rocblas_int j = 0; j < dim; j += GETRF_GETF2_SWITCHSIZE) {
        // Factor diagonal and subdiagonal blocks 
        jb = min(dim - j, GETRF_GETF2_SWITCHSIZE);  //number of columns in the block
        hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,iinfo,batch_count,0);
        rocsolver_getf2_template<T>(handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, ipiv, shiftP + j, strideP, iinfo, batch_count, scalars, pivotGPU);
        
        // adjust pivot indices and check singularity
        sizePivot = min(m - j, jb);     //number of pivots in the block
        blocksPivot = (sizePivot - 1) / GETF2_BLOCKSIZE + 1; 
        gridPivot = dim3(blocksPivot, batch_count, 1);
        hipLaunchKernelGGL(getrf_check_singularity<U>,gridPivot,threads,0,stream,
			   sizePivot,j,ipiv,shiftP + j,strideP,iinfo,info);

        // apply interchanges to columns 1 : j-1
        rocsolver_laswp_template<T>(handle, j, A, shiftA, lda, strideA, j + 1, j + jb, ipiv, shiftP, strideP, 1, batch_count);

        if (j + jb < n) {
            // apply interchanges to columns j+jb : n
            rocsolver_laswp_template<T>(handle, (n - j - jb), A,
                                  shiftA + idx2D(0, j + jb, lda), lda, strideA, j + 1, j + jb,
                                  ipiv, shiftP, strideP, 1, batch_count);

            // compute block row of U
            for (int b=0;b<batch_count;++b) {
                M = load_ptr_batch<T>(AA,b,shiftA,strideA);
                rocblas_trsm(handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                             rocblas_diagonal_unit, jb, (n - j - jb), &one,
                             (M + idx2D(j, j, lda)), lda, (M + idx2D(j, j + jb, lda)), lda);
            }

            // update trailing submatrix
            if (j + jb < m) {
                rocblasCall_gemm<BATCHED,STRIDED,T>(handle, rocblas_operation_none, rocblas_operation_none,
                                                m - j - jb, n - j - jb, jb, &minone,
                                                A, shiftA+idx2D(j + jb, j, lda), lda, strideA,
                                                A, shiftA+idx2D(j, j + jb, lda), lda, strideA, &one,
                                                A, shiftA+idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, nullptr);
            }
        } 
    }

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}


#endif /* ROCLAPACK_GETRF_HPP */
