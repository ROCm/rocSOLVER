/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORG2R_HPP
#define ROCLAPACK_ORG2R_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"

template <typename T, typename U>
__global__ void init_ident_col(const rocblas_int m, const rocblas_int n, const rocblas_int k, U A,
                               const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int strideA)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;

    if (i < m && j < n) {
        T *Ap = load_ptr_batch<T>(A,shiftA,b,strideA);
        
        if (i == j) 
            Ap[i + j*lda] = 1.0;
        else if (j > i) 
            Ap[i + j*lda] = 0.0;
        else if (j >= k)
            Ap[i + j*lda] = 0.0;
    }
}

template <typename T, typename U>
rocblas_status rocsolver_org2r_template(rocsolver_handle handle, const rocsolver_int m, 
                                   const rocsolver_int n, const rocsolver_int k, U A, const rocblas_int shiftA, 
                                   const rocsolver_int lda, const rocsolver_int strideA, T* ipiv, 
                                   const rocsolver_int strideP, const rocsolver_int batch_count)
{
    // quick return
    if (!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****
    
    T* M;

    // Initialize identity matrix (non used columns)
    rocblas_int blocksx = (m - 1)/32 + 1;
    rocblas_int blocksy = (n - 1)/32 + 1;
    hipLaunchKernelGGL(init_ident_col<T>,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,
                        m,n,k,A,shiftA,lda,strideA);

    for (int j = k-1; j >= 0; --j) {
        // apply H(i) to Q(i:m,i:n) from the left
        if (j < n - 1) {
            rocsolver_larf_template(handle,rocblas_side_left,           //side
                                    m - j,                              //number of rows of matrix to modify
                                    n - j - 1,                          //number of columns of matrix to modify    
                                    A, shiftA + idx2D(j,j,lda),         //householder vector x
                                    1, strideA,                         //inc of x
                                    (ipiv + j), strideP,                //householder scalar (alpha)
                                    A, shiftA + idx2D(j,j+1,lda),       //matrix to work on
                                    lda, strideA,                       //leading dimension
                                    batch_count);          
        }

        // set the diagonal element and negative tau
        hipLaunchKernelGGL(setdiag<T>,dim3(batch_count),dim3(1),0,stream,
                            j,A,shiftA,lda,strideA,ipiv,strideP);
        
        // update i-th column -corresponding to H(i)-
        if (j < m - 1) {
            for (int b=0;b<batch_count;++b) {
                M = load_ptr_batch<T>(AA,shiftA,b,strideA);
                rocblas_scal(handle, (m-j-1), (ipiv + b*strideP + j), 
                            (M + idx2D(j + 1, j, lda)), 1); 
            }          
        }
    }
    
    // restore values of tau
    blocksx = (k - 1)/128 + 1;
    hipLaunchKernelGGL(restau<T>,dim3(blocksx,batch_count),dim3(128),0,stream,
                            k,ipiv,strideP);
 
    return rocblas_status_success;
}

#endif
