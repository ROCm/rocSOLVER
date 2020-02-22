/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGLQ_HPP
#define ROCLAPACK_ORGLQ_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include "ideal_sizes.hpp"
#include "../auxiliary/rocauxiliary_orgl2.hpp"
#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"

template <typename T, typename U>
__global__ void set_zero_row(const rocblas_int m, const rocblas_int kk, U A,
                             const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int strideA)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x + kk;

    if (i < m && j < kk) {
        T *Ap = load_ptr_batch<T>(A,shiftA,b,strideA);
        
        Ap[i + j*lda] = 0.0;
    }
}


template <typename T, typename U>
rocblas_status rocsolver_orglq_template(rocsolver_handle handle, const rocsolver_int m, 
                                   const rocsolver_int n, const rocsolver_int k, U A, const rocblas_int shiftA, 
                                   const rocsolver_int lda, const rocsolver_int strideA, T* ipiv, 
                                   const rocsolver_int strideP, const rocsolver_int batch_count)
{
    // quick return
    if (!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    // if the matrix is small, use the unblocked variant of the algorithm
    if (k <= GEQRF_GEQR2_SWITCHSIZE) 
        return rocsolver_orgl2_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count);

    //memory in GPU (workspace)
    T* work;
    rocblas_int ldw = GEQRF_GEQR2_BLOCKSIZE;
    rocblas_int strideW = ldw *ldw;
    hipMalloc(&work, sizeof(T)*strideW*batch_count);

    // start of first blocked block
    rocblas_int jb = GEQRF_GEQR2_BLOCKSIZE;
    rocblas_int j = ((k - GEQRF_GEQR2_SWITCHSIZE - 1) / jb) * jb;
    
    // start of the unblocked block
    rocblas_int kk = min(k, j + jb); 

    rocblas_int blocksy, blocksx;
    
    // compute the unblockled part and set to zero the 
    // corresponding left submatrix
    if (kk < m) {
        blocksx = (m - kk - 1)/32 + 1;
        blocksy = (kk - 1)/32 + 1;
        hipLaunchKernelGGL(set_zero_row<T>,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,
                           m,kk,A,shiftA,lda,strideA);
        
        rocsolver_orgl2_template<T>(handle, m - kk, n - kk, k - kk, 
                                    A, shiftA + idx2D(kk, kk, lda), lda, 
                                    strideA, (ipiv + kk), strideP, batch_count);
    }

    // compute the blocked part
    while (j >= 0) {
        
        // first update the already computed part
        // applying the current block reflector using larft + larfb
        if (j + jb < m) {
            rocsolver_larft_template<T>(handle, rocsolver_forward_direction, 
                                        rocsolver_row_wise, n-j, jb, 
                                        A, shiftA + idx2D(j,j,lda), lda, strideA, 
                                        (ipiv + j), strideP,
                                        work, ldw, strideW, batch_count);

            rocsolver_larfb_template<T>(handle,rocblas_side_right,rocblas_operation_transpose,rocsolver_forward_direction,
                                        rocsolver_row_wise,m-j-jb, n-j, jb,
                                        A, shiftA + idx2D(j,j,lda), lda, strideA,
                                        work, 0, ldw, strideW,
                                        A, shiftA + idx2D(j+jb,j,lda), lda, strideA, batch_count);
        }

        // now compute the current block and set to zero
        // the corresponding top submatrix
        if (j > 0) {
            blocksx = (jb - 1)/32 + 1;
            blocksy = (j - 1)/32 + 1;
            hipLaunchKernelGGL(set_zero_row<T>,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,
                               j+jb,j,A,shiftA,lda,strideA);
        }
        rocsolver_orgl2_template<T>(handle, jb, n - j, jb, 
                                    A, shiftA + idx2D(j, j, lda), lda, 
                                    strideA, (ipiv + j), strideP, batch_count);

        j -= jb;
    }
 
    hipFree(work);

    return rocblas_status_success;
}

#endif
