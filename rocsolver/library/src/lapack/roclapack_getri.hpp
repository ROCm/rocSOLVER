/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRI_H
#define ROCLAPACK_GETRI_H

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U, typename V>
__global__ void getri_trtri(const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                            const rocblas_stride strideA, V W, const rocblas_int shiftW, const rocblas_int ldw,
                            const rocblas_stride strideW, rocblas_int *info, rocblas_int *fails)
{
    // unblocked trtri kernel assuming non-unit upper triangular matrix
    int b = hipBlockIdx_x;
    //int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(W,b,shiftW,strideW);

    // compute info
    if (hipThreadIdx_y == 0)
        info[b] = 0;
    __syncthreads();
    for (int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if (a[i + i * lda] == 0)
        {
            rocblas_int _info = info[b];
            while (_info == 0 || _info > i + 1)
                _info = atomicCAS(info + b, _info, i + 1);
        }
    }
    __syncthreads();

    if (info[b] != 0)
        return;

    // diagonal element
    for (int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();
    
    // compute element i of each column j
    T aij;
    for (rocblas_int j = 1; j < n; j++)
    {
        for (int i = hipThreadIdx_y; i < j; i += hipBlockDim_y)
            w[i] = a[i + j * lda];
        __syncthreads();
        
        for (int i = hipThreadIdx_y; i < j; i += hipBlockDim_y)
        {
            aij = 0;

            for (rocblas_int ii = i; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -a[j + j * lda] * aij;
        }
        __syncthreads();
    }
}

template <typename T, typename U, typename V>
__global__ void copy_and_zero(const rocblas_int m, const rocblas_int n,
                              U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                              V W, const rocblas_int shiftw, const rocblas_int ldw, const rocblas_stride stridew,
                              rocblas_fill uplo)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    T* a = load_ptr_batch<T>(A,b,shifta,stridea);
    T* w = load_ptr_batch<T>(W,b,shiftw,stridew);

    if (i < m && j < n && (uplo == rocblas_fill_lower ? i > j : i <= j))
    {
        w[i + j*ldw] = a[i + j*lda];
        a[i + j*lda] = 0;
    }
}

template <typename T, typename U>
__global__ void getri_pivot(const rocblas_int n,
                            U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                            rocblas_int *ipiv, const rocblas_int shiftp, const rocblas_stride stridep)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,b,shifta,stridea);
    rocblas_int* p = load_ptr_batch<rocblas_int>(ipiv,b,shiftp,stridep);

    rocblas_int jp;
    for (rocblas_int j = n-2; j >= 0; --j)
    {
        jp = p[j] - 1;
        if (jp != j)
            swap(n, a + j*lda, 1, a + jp*lda, 1);
    }
}


template <bool BATCHED, typename T>
void rocsolver_getri_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // for scalars
    *size_1 = sizeof(T)*3;

    // for workspace
    if (n <= GETRI_SWITCHSIZE)
        *size_2 = n;
    else
        *size_2 = n * GETRI_SWITCHSIZE;
    *size_2 *= sizeof(T)*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = sizeof(T*) * batch_count;
    else
        *size_3 = 0;
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n, const rocblas_int lda, T A, rocblas_int *ipiv,
                                        rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n, const rocblas_int lda, const rocblas_int ldc, T A, T C,
                                        rocblas_int *ipiv, rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (n < 0 || lda < n || ldc < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (n && !C) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getri_template(rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv,
                                        const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int *info,
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

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

    T minone = -1;
    T one = 1;
    T *M, *W;
    rocblas_int fails = 0;
    rocblas_int jb, nb = GETRI_SWITCHSIZE;
    rocblas_int ldw = n;
    rocblas_stride strideW;

    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      TRTRI_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    // compute inv(U)
    strideW = n;
    rocblas_int threads = min(n, 1024);
    hipHostRegister(&fails, sizeof(rocblas_int), hipHostRegisterDefault);
    hipLaunchKernelGGL(getri_trtri<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                       n, A, shiftA, lda, strideA, work, 0, ldw, strideW, info, &fails);
    hipHostUnregister(&fails);
    if (fails == batch_count)
        return rocblas_status_success;
    
    if (n <= nb)
    {
        // use unblocked version
        // strideW = n;

        for (rocblas_int j = n-2; j >= 0; --j)
        {
            rocblas_int blocks = ((n-j) - 1)/64 + 1;
            hipLaunchKernelGGL(copy_and_zero<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                               n-j, 1, A, shiftA + idx2D(j,j,lda), lda, strideA, work, j, ldw, strideW, rocblas_fill_lower);

            rocblasCall_gemv(handle, rocblas_operation_none, n, n-j-1,
                             &minone, 0, A, shiftA + idx2D(0,j+1,lda), lda, strideA,
                             work, j+1, 1, strideW,
                             &one, 0, A, shiftA + idx2D(0,j,lda), 1, strideA,
                             batch_count, workArr);
        }
    }
    else
    {
        //use blocked version
        strideW = n*nb;

        rocblas_int nn = ((n - 1)/nb)*nb + 1;
        for (rocblas_int j = nn-1; j >= 0; j -= nb)
        {
            jb = min(n-j, nb);

            rocblas_int blocks1 = ((n-j) - 1)/32 + 1;
            rocblas_int blocks2 = (jb - 1)/32 + 1;
            hipLaunchKernelGGL(copy_and_zero<T>, dim3(batch_count,blocks1,blocks2), dim3(1,32,32), 0, stream,
                               n-j, jb, A, shiftA + idx2D(j,j,lda), lda, strideA, work, j, ldw, strideW, rocblas_fill_lower);

            if (j+jb < n)
                rocblasCall_gemm<BATCHED,STRIDED>(handle, rocblas_operation_none, rocblas_operation_none,
                                                  n, jb, n-j-jb,
                                                  &minone, A, shiftA + idx2D(0,j+jb,lda), lda, strideA,
                                                  work, j+jb, ldw, strideW,
                                                  &one, A, shiftA + idx2D(0,j,lda), lda, strideA,
                                                  batch_count, workArr);
            
            for (int b = 0; b < batch_count; ++b)
            {
                M = load_ptr_batch<T>(AA,b,shiftA,strideA);
                rocblas_trsm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                             rocblas_diagonal_unit, n, jb,
                             &one, work + j + b*strideW, ldw,
                             M + idx2D(0,j,lda), lda);
            }
        }
    }
    
    hipLaunchKernelGGL(getri_pivot<T>, dim3(batch_count,1,1), dim3(1,1,1), 0, stream,
                       n, A, shiftA, lda, strideA, ipiv, shiftP, strideP);

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRI_H */
