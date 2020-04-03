/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGBR_HPP
#define ROCLAPACK_ORGBR_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include "ideal_sizes.hpp"
#include "../auxiliary/rocauxiliary_orgqr.hpp"
#include "../auxiliary/rocauxiliary_orglq.hpp"
#include <vector>

#define BS 32 //blocksize for kernels

template <typename T, typename U>
__global__ void copyshift_col(const bool copy, const rocblas_int dim, U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, 
                         T *W, const rocblas_int shiftW, const rocblas_int ldw, const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (i < dim && j < dim && j <= i) {
        rocblas_int offset = j*(j+1)/2; //to acommodate in smaller array W

        T *Ap = load_ptr_batch<T>(A,shiftA,b,strideA);    
        T *Wp = load_ptr_batch<T>(W,shiftW,b,strideW);
        
        if (copy) {
            //copy columns
            Wp[i + j*ldw - offset] = (j == 0 ? 0.0 : Ap[i+1 + (j-1)*lda]);    
        
        } else {
            // shift columns to the right   
            Ap[i+1 + j*lda] = Wp[i + j*ldw - offset];
            
            // make first row the identity
            if (i == j) {
                Ap[(j+1)*lda] = 0.0;
                if (i == 0)
                    Ap[0] = 1.0;
            }
        }
    }
}

template <typename T, typename U>
__global__ void copyshift_row(const bool copy, const rocblas_int dim, U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, 
                         T *W, const rocblas_int shiftW, const rocblas_int ldw, const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (i < dim && j < dim && i <= j) {
        rocblas_int offset = j*ldw - j*(j+1)/2; //to acommodate in smaller array W

        T *Ap = load_ptr_batch<T>(A,shiftA,b,strideA);    
        T *Wp = load_ptr_batch<T>(W,shiftW,b,strideW);
        
        if (copy) {
            //copy rows
            Wp[i + j*ldw - offset] = (i == 0 ? 0.0 : Ap[i-1 + (j+1)*lda]);    
        
        } else {
            // shift rows downward   
            Ap[i + (j+1)*lda] = Wp[i + j*ldw - offset];
            
            // make first column the identity
            if (i == j) {
                Ap[i+1] = 0.0;
                if (j == 0)
                    Ap[0] = 1.0;
            }
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgbr_template(rocblas_handle handle, const rocsolver_storev storev, const rocblas_int m, 
                                   const rocblas_int n, const rocblas_int k, U A, const rocblas_int shiftA, 
                                   const rocblas_int lda, const rocblas_stride strideA, T* ipiv, 
                                   const rocblas_stride strideP, const rocblas_int batch_count)
{
    // quick return
    if (!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if column-wise, compute orthonormal columns of matrix Q in the bi-diagonalization 
    // of a m-by-k matrix A (given by gebrd)
    if (storev == rocsolver_column_wise) {
        if (m >= k) {
            rocsolver_orgqr_template<BATCHED,STRIDED,T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count);    
        } else {
            // shift the householder vectors provided by gebrd as they come below the first subdiagonal
            // workspace
            // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
            T *W;
            rocblas_stride strideW = rocblas_stride(m - 1)*m/2;  //number of elements to copy
            size_t sizeW = size_t(strideW)*batch_count;
            rocblas_int ldw = m - 1;
            hipMalloc(&W, sizeof(T)*sizeW);
            rocblas_int blocks = (m - 2)/BS + 1;

            // copy
            hipLaunchKernelGGL(copyshift_col<T>,dim3(blocks,blocks,batch_count),dim3(BS,BS),0,stream, 
                                true,m-1,A,shiftA,lda,strideA,W,0,ldw,strideW);           

            // shift
            hipLaunchKernelGGL(copyshift_col<T>,dim3(blocks,blocks,batch_count),dim3(BS,BS),0,stream, 
                                false,m-1,A,shiftA,lda,strideA,W,0,ldw,strideW);           
            
            // result
            rocsolver_orgqr_template<BATCHED,STRIDED,T>(handle, m-1, m-1, m-1, A, shiftA + idx2D(1,1,lda), lda, strideA, ipiv, strideP, batch_count);    
        
            hipFree(W);
        }   
    }
    
    // if row-wise, compute orthonormal rowss of matrix P' in the bi-diagonalization 
    // of a k-by-n matrix A (given by gebrd)
    else {
        if (n > k) {
            rocsolver_orglq_template<BATCHED,STRIDED,T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count);
        } else {
            // shift the householder vectors provided by gebrd as they come above the first superdiagonal
            // workspace
            // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
            T *W;
            rocblas_stride strideW = rocblas_stride(n - 1)*n/2;  //number of elements to copy
            size_t sizeW = size_t(strideW)*batch_count;
            rocblas_int ldw = n - 1;
            hipMalloc(&W, sizeof(T)*sizeW);
            rocblas_int blocks = (n - 2)/BS + 1;

            // copy
            hipLaunchKernelGGL(copyshift_row<T>,dim3(blocks,blocks,batch_count),dim3(BS,BS),0,stream, 
                                true,n-1,A,shiftA,lda,strideA,W,0,ldw,strideW);           

            // shift
            hipLaunchKernelGGL(copyshift_row<T>,dim3(blocks,blocks,batch_count),dim3(BS,BS),0,stream, 
                                false,n-1,A,shiftA,lda,strideA,W,0,ldw,strideW);           

            // result
            rocsolver_orglq_template<BATCHED,STRIDED,T>(handle, n-1, n-1, n-1, A, shiftA + idx2D(1,1,lda), lda, strideA, ipiv, strideP, batch_count);
                
            hipFree(W);
        }
    }    

    return rocblas_status_success;
}

#endif
