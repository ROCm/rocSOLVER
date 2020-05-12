/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGLQ_UNGLQ_HPP
#define ROCLAPACK_ORGLQ_UNGLQ_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "ideal_sizes.hpp"
#include "../auxiliary/rocauxiliary_orgl2_ungl2.hpp"
#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"

template <typename T, typename U>
__global__ void set_zero_row(const rocblas_int m, const rocblas_int kk, U A,
                             const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x + kk;

    if (i < m && j < kk) {
        T *Ap = load_ptr_batch<T>(A,b,shiftA,strideA);
        
        Ap[i + j*lda] = 0.0;
    }
}

template <typename T, bool BATCHED>
void rocsolver_orglq_unglq_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int k, const rocblas_int batch_count,
                                         size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    size_t s1, s2, s3;
    rocsolver_orgl2_ungl2_getMemorySize<T,BATCHED>(m,n,batch_count,size_1,size_2,size_3);

    if (k <= GEQRF_GEQR2_SWITCHSIZE) {
        *size_4 = 0;
    } else {
        // size of workspace
        // maximum of what is needed by org2r, larft and larfb
        rocblas_int jb = GEQRF_GEQR2_BLOCKSIZE;
        rocblas_int j = ((k - GEQRF_GEQR2_SWITCHSIZE - 1) / jb) * jb;
        rocblas_int kk = min(k, j + jb);
        rocsolver_orgl2_ungl2_getMemorySize<T>(max(m-kk,jb),n,batch_count,&s1);
        rocsolver_larft_getMemorySize<T>(jb, batch_count, &s2);
        rocsolver_larfb_getMemorySize<T>(rocblas_side_left, m-jb, n, jb, batch_count, &s3);

        *size_2 = max(max(s1,s2),s3);

        // size of temporary array for triangular factor
        *size_4 = sizeof(T)*jb*jb*batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orglq_unglq_template(rocblas_handle handle, const rocblas_int m, 
                                   const rocblas_int n, const rocblas_int k, U A, const rocblas_int shiftA, 
                                   const rocblas_int lda, const rocblas_stride strideA, T* ipiv, 
                                   const rocblas_stride strideP, const rocblas_int batch_count,
                                   T* scalars, T* work, T** workArr, T* trfact)
{
    // quick return
    if (!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    // if the matrix is small, use the unblocked variant of the algorithm
    if (k <= GEQRF_GEQR2_SWITCHSIZE) 
        return rocsolver_orgl2_ungl2_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work, workArr);

    rocblas_int ldw = GEQRF_GEQR2_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) *ldw;

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
        
        rocsolver_orgl2_ungl2_template<T>(handle, m - kk, n - kk, k - kk, 
                                          A, shiftA + idx2D(kk, kk, lda), lda, 
                                          strideA, (ipiv + kk), strideP, batch_count, scalars, work, workArr);
    }

    // compute the blocked part
    while (j >= 0) {
        
        // first update the already computed part
        // applying the current block reflector using larft + larfb
        if (j + jb < m) {
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, 
                                        rocblas_row_wise, n-j, jb, 
                                        A, shiftA + idx2D(j,j,lda), lda, strideA, 
                                        (ipiv + j), strideP,
                                        trfact, ldw, strideW, batch_count, scalars, work, workArr);

            rocsolver_larfb_template<BATCHED,STRIDED,T>(handle,rocblas_side_right,rocblas_operation_conjugate_transpose,rocblas_forward_direction,
                                        rocblas_row_wise,m-j-jb, n-j, jb,
                                        A, shiftA + idx2D(j,j,lda), lda, strideA,
                                        trfact, 0, ldw, strideW,
                                        A, shiftA + idx2D(j+jb,j,lda), lda, strideA, batch_count, work, workArr);
        }

        // now compute the current block and set to zero
        // the corresponding top submatrix
        if (j > 0) {
            blocksx = (jb - 1)/32 + 1;
            blocksy = (j - 1)/32 + 1;
            hipLaunchKernelGGL(set_zero_row<T>,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,
                               j+jb,j,A,shiftA,lda,strideA);
        }
        rocsolver_orgl2_ungl2_template<T>(handle, jb, n - j, jb, 
                                          A, shiftA + idx2D(j, j, lda), lda, 
                                          strideA, (ipiv + j), strideP, batch_count, scalars, work, workArr);

        j -= jb;
    }
 
    return rocblas_status_success;
}

#endif
