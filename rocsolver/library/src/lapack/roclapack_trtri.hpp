/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_TRTRI_H
#define ROCLAPACK_TRTRI_H

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U>
__global__ void trtri_check_singularity(const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                            const rocblas_stride strideA, rocblas_int *info)
{
    // unblocked trtri kernel assuming non-unit upper triangular matrix
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);

    // compute info
    if (i == 0)
        info[b] = 0;
    __syncthreads();
    if (i < n)
    {
        if (a[i + i * lda] == 0)
        {
            rocblas_int _info = info[b];
            while (_info == 0 || _info > i + 1)
                _info = atomicCAS(info + b, _info, i + 1);
        }
    }
}

template <typename T, typename U, typename V>
__global__ void trtri_kernel(const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                            const rocblas_stride strideA, V W, const rocblas_int shiftW, const rocblas_int ldw,
                            const rocblas_stride strideW, rocblas_int *info)
{
    // unblocked trtri kernel assuming non-unit upper triangular matrix
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_y;

    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(W,b,shiftW,strideW);

    if (info[b] != 0)
        return;

    // diagonal element
    a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();
    
    // compute element i of each column j
    T aij;
    for (rocblas_int j = 1; j < n; j++)
    {
        if (i < j)
            w[i] = a[i + j * lda];
        __syncthreads();
        
        if (i < j)
        {
            aij = 0;

            for (rocblas_int ii = i; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -a[j + j * lda] * aij;
        }
        __syncthreads();
    }
}


template <bool BATCHED, typename T>
void rocsolver_trtri_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // for scalars
    *size_1 = sizeof(T)*3;

    // for workspace
    if (n <= TRTRI_SWITCHSIZE)
        *size_2 = n;
    else
        *size_2 = max(n, 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB);
    *size_2 *= sizeof(T)*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = 3 * sizeof(T*) * batch_count;
    else
        *size_3 = 0;
}

template <typename T>
rocblas_status rocsolver_trtri_argCheck(const rocblas_int n, const rocblas_int lda, T A, rocblas_int *info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trtri_template(rocblas_handle handle, const rocblas_fill uplo, const rocblas_diagonal diag,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                                        const rocblas_stride strideA, rocblas_int *info,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr)
{
    // quick return if zero instances in batch
    if (batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if (n == 0)
    {
        rocblas_int blocks = (batch_count - 1)/32 + 1;
        hipLaunchKernelGGL(reset_info, dim3(blocks,1,1), dim3(32,1,1), 0, stream,
                           info, batch_count, 0);
        return rocblas_status_success;
    }

    // only non-unit upper triangular matrices currently supported
    if (uplo != rocblas_fill_upper || diag != rocblas_diagonal_non_unit)
        return rocblas_status_not_implemented;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      TRSM_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    T minone = -1;
    T one = 1;
    rocblas_int jb, nb = TRTRI_SWITCHSIZE;
    rocblas_int ldw;
    rocblas_stride strideW;

    rocblas_int blocks = (n - 1)/32 + 1;
    hipLaunchKernelGGL(trtri_check_singularity<T>, dim3(batch_count,blocks,1), dim3(1,32,1), 0, stream,
                        n, A, shiftA, lda, strideA, info);
    
    if (n <= nb)
    {
        //use unblocked version
        ldw = n;
        strideW = n;
        hipLaunchKernelGGL(trtri_kernel<T>, dim3(batch_count,1,1), dim3(1,n,1), 0, stream,
                        n, A, shiftA, lda, strideA, work, 0, ldw, strideW, info);
    }
    else
    {
        //use blocked version
        T* M;
        ldw = nb;
        strideW = nb;
        for (rocblas_int j = 0; j < n; j += nb)
        {
            jb = min(n-j, nb);
            
            rocblasCall_trmm<BATCHED,STRIDED,T>(handle, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                                                diag, j, jb, &one, A, shiftA, lda, strideA,
                                                A, shiftA + idx2D(0,j,lda), lda, strideA, batch_count, work, workArr);
            for (int b = 0; b < batch_count; ++b)
            {
                M = load_ptr_batch<T>(AA,b,shiftA,strideA);
                rocblas_trsm(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                             diag, j, jb, &minone, M + idx2D(j,j,lda), lda, M + idx2D(0,j,lda), lda);
            }

            hipLaunchKernelGGL(trtri_kernel<T>, dim3(batch_count,1,1), dim3(1,jb,1), 0, stream,
                            jb, A, shiftA + idx2D(j,j,lda), lda, strideA, work, 0, ldw, strideW, info);
        }
    }

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRI_H */
