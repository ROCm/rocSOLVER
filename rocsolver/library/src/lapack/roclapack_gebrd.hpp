/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.8.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2017
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEBRD_H
#define ROCLAPACK_GEBRD_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "roclapack_gebd2.hpp"
#include "../auxiliary/rocauxiliary_labrd.hpp"


template <typename T, bool BATCHED>
void rocsolver_gebrd_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4, size_t *size_5, size_t *size_6)
{
    if (m <= GEBRD_GEBD2_SWITCHSIZE || n <= GEBRD_GEBD2_SWITCHSIZE)
    {
        rocsolver_gebd2_getMemorySize<T,BATCHED>(m,n,batch_count,size_1,size_2,size_3,size_4);
        *size_5 = 0;
        *size_6 = 0;
    }
    else
    {
        size_t s1, s2, s3, s4;
        rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
        rocblas_int d = min(m / k, n / k);
        rocsolver_gebd2_getMemorySize<T,BATCHED>(m-d*k,n-d*k,batch_count,size_1,size_2,size_3,size_4);
        rocsolver_labrd_getMemorySize<T,BATCHED>(m,n,batch_count,&s1,&s2,&s3,&s4);
        *size_1 = max(*size_1, s1);
        *size_2 = max(*size_2, s2);
        *size_3 = max(*size_3, s3);
        *size_4 = max(*size_4, s4);

        // size of matrix X
        *size_5 = m * k;
        *size_5 *= sizeof(T) * batch_count;

        // size of matrix Y
        *size_6 = n * k;
        *size_6 *= sizeof(T) * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        S* D, const rocblas_stride strideD, S* E, const rocblas_stride strideE,
                                        T* tauq, const rocblas_stride strideQ, T* taup, const rocblas_stride strideP,
                                        U X, const rocblas_int shiftX, const rocblas_int ldx, const rocblas_stride strideX,
                                        U Y, const rocblas_int shiftY, const rocblas_int ldy, const rocblas_stride strideY,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr, T* diag)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host); 
    
    T minone = -1;
    T one = 1;
    rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
    rocblas_int dim = min(m, n);    //total number of pivots
    rocblas_int jb, j = 0;
    rocblas_int blocks;

    // if the matrix is small, use the unblocked variant of the algorithm
    if (m <= k || n <= k)
    {
        rocsolver_gebd2_template<S,T>(handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE,
                                      tauq, strideQ, taup, strideP, batch_count, scalars, work, workArr, diag);
        
        rocblas_set_pointer_mode(handle,old_mode);  
        return rocblas_status_success;
    }

    // zero X and Y
    blocks = (ldx*k - 1)/64 + 1;
    hipLaunchKernelGGL(reset_batch_info<T>, dim3(blocks,batch_count,1), dim3(64,1,1), 0, stream,
        X + shiftX, strideX, ldx*k, 0);
    blocks = (ldy*k - 1)/64 + 1;
    hipLaunchKernelGGL(reset_batch_info<T>, dim3(blocks,batch_count,1), dim3(64,1,1), 0, stream,
        Y + shiftY, strideY, ldy*k, 0);
    
    while (j < dim - k) {
        // Reduce block to bidiagonal form
        jb = min(dim - j, k);  //number of rows and columns in the block
        rocsolver_labrd_template<S,T>(handle, m-j, n-j, jb,
                                      A, shiftA + idx2D(j,j,lda), lda, strideA,
                                      D + j, strideD, E + j, strideE, tauq + j, strideQ, taup + j, strideP,
                                      X, shiftX, ldx, strideX, Y, shiftY, ldy, strideY,
                                      batch_count, scalars, work, workArr, diag);

        //update the rest of the matrix
        rocblasCall_gemm<BATCHED,STRIDED,T>(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                                            m-j-jb, n-j-jb, jb, &minone,
                                            A, shiftA + idx2D(j+jb,j,lda), lda, strideA,
                                            Y, shiftY + jb, ldy, strideY, &one,
                                            A, shiftA + idx2D(j+jb,j+jb,lda), lda, strideA, batch_count, workArr);
        rocblasCall_gemm<BATCHED,STRIDED,T>(handle, rocblas_operation_none, rocblas_operation_none,
                                            m-j-jb, n-j-jb, jb, &minone,
                                            X, shiftX + jb, ldx, strideX,
                                            A, shiftA + idx2D(j,j+jb,lda), lda, strideA, &one,
                                            A, shiftA + idx2D(j+jb,j+jb,lda), lda, strideA, batch_count, workArr);

        blocks = (jb - 1)/64 + 1;
        if (m >= n)
        {
            hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                D, j, strideD, A, shiftA + idx2D(j,j,lda), lda, strideA, jb);
            hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                E, j, strideE, A, shiftA + idx2D(j,j+1,lda), lda, strideA, jb);
        }
        else
        {
            hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                D, j, strideD, A, shiftA + idx2D(j,j,lda), lda, strideA, jb);
            hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                E, j, strideE, A, shiftA + idx2D(j+1,j,lda), lda, strideA, jb);
        }

        j += GEBRD_GEBD2_SWITCHSIZE;
    }

    //factor last block
    if (j < dim)
        rocsolver_gebd2_template<S,T>(handle, m-j, n-j, A, shiftA + idx2D(j,j,lda), lda, strideA, D + j, strideD, E + j, strideE,
                                      tauq + j, strideQ, taup + j, strideP, batch_count, scalars, work, workArr, diag);

    rocblas_set_pointer_mode(handle,old_mode);  
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GEBRD_H */
