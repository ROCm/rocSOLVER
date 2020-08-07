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
#include "rocblas_device_functions.hpp"
#include "rocsolver.h"

template <typename T>
__device__ void trtri_check_singularity(const rocblas_diagonal diag, const rocblas_int n, T *a,
                                        const rocblas_int lda, rocblas_int *info)
{
    // check for singularities
    int b = hipBlockIdx_x;

    if (diag == rocblas_diagonal_unit)
    {
        if (hipThreadIdx_x == 0)
            info[b] = 0;
        __syncthreads();
        return;
    }

    __shared__ rocblas_int _info;
    
    // compute info
    if (hipThreadIdx_y == 0)
        _info = 0;
    __syncthreads();
    for (int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if (a[i + i * lda] == 0)
        {
            rocblas_int _info_temp = _info;
            while (_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
    }
    __syncthreads();

    if (hipThreadIdx_y == 0)
        info[b] = _info;
    __syncthreads();
}

template <typename T>
__device__ void trtri_unblk(const rocblas_diagonal diag, const rocblas_int n, T *a, const rocblas_int lda,
                            rocblas_int *info, T *w)
{
    // unblocked trtri kernel assuming upper triangular matrix
    int i = hipThreadIdx_y;
    if (i >= n)
        return;

    // diagonal element
    if (diag == rocblas_diagonal_non_unit)
    {
        a[i + i * lda] = 1.0 / a[i + i * lda];
        __syncthreads();
    }
    
    // compute element i of each column j
    T ajj, aij;
    for (rocblas_int j = 1; j < n; j++)
    {
        ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);

        if (i < j)
            w[i] = a[i + j * lda];
        __syncthreads();
        
        if (i < j)
        {
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for (rocblas_int ii = i+1; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}


template <typename T, typename U, typename V>
__global__ void trtri_kernel(const rocblas_diagonal diag, const rocblas_int n,
                             U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                             rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    trtri_check_singularity(diag, n, a, lda, info);
    if (info[b] != 0)
        return;

    if (n <= TRTRI_SWITCHSIZE_MID)
        // use unblocked version
        trtri_unblk(diag, n, a, lda, info, w);
    else
    {
        // use blocked version
        T minone = -1;
        T one = 1;
        rocblas_int jb, nb = TRTRI_BLOCKSIZE;
        
        for (rocblas_int j = 0; j < n; j += nb)
        {
            jb = min(n-j, nb);

            trmm_kernel_left_upper(diag, j, jb, &one, a, lda, a + j*lda, lda, w);
            trsm_kernel_right_upper(diag, j, jb, &minone, a + j+j*lda, lda, a + j*lda, lda);
            trtri_unblk(diag, jb, a + j+j*lda, lda, info, w);
        }
    }
}

template <typename T, typename U, typename V>
__global__ void trtri_kernel_large(const rocblas_diagonal diag, const rocblas_int n, const rocblas_int j, const rocblas_int jb,
                                   U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                   rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n * TRTRI_BLOCKSIZE;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    if (j == 0)
        trtri_check_singularity(diag, n, a, lda, info);
    
    if (info[b] != 0)
    {
        // if A is singular, we want it to remain unaltered by trmm
        int idx = hipThreadIdx_y;

        int jj = j + idx;
        if (j > 0 && jj < n)
        {
            // restore original entries of A
            for (int i = 0; i < j; i++)
                a[i + jj * lda] = w[i + idx * n];
        }

        jj = j + TRTRI_BLOCKSIZE + idx;
        if (jj < n)
        {
            // save original entries of A
            for (int i = 0; i < j + TRTRI_BLOCKSIZE; i++)
                w[i + idx * n] = a[i + jj * lda];
        }
    }
    else
    {
        T minone = -1;
        trsm_kernel_right_upper(diag, j, jb, &minone, a + j+j*lda, lda, a + j*lda, lda);
        trtri_unblk(diag, jb, a + j+j*lda, lda, info, w);
    }
}


template <bool BATCHED, typename T>
void rocsolver_trtri_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // for scalars
    *size_1 = sizeof(T)*3;

    // for workspace
    if (n <= TRTRI_SWITCHSIZE_LARGE)
        *size_2 = n;
    else
        *size_2 = n * TRTRI_BLOCKSIZE + 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB;
    *size_2 *= sizeof(T)*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = 3 * sizeof(T*) * batch_count;
    else
        *size_3 = 0;
}

template <typename T>
rocblas_status rocsolver_trtri_argCheck(const rocblas_fill uplo, const rocblas_diagonal diag, const rocblas_int n,
                                        const rocblas_int lda, T A, rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if (diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;
    
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
    if (uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;

    rocblas_int threads = min(((n - 1)/64 + 1) * 64, TRTRI_BLOCKSIZE);
    
    if (n <= TRTRI_SWITCHSIZE_LARGE)
    {
        hipLaunchKernelGGL(trtri_kernel<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                        diag, n, A, shiftA, lda, strideA, info, work);
    }
    else
    {
        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle,&old_mode);
        rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

        T one = 1;
        rocblas_int jb, nb = TRTRI_BLOCKSIZE;
        rocblas_int shiftW = batch_count * n * TRTRI_BLOCKSIZE;

        for (rocblas_int j = 0; j < n; j += nb)
        {
            jb = min(n-j, nb);
            
            rocblasCall_trmm<BATCHED,STRIDED,T>(handle, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                                                diag, j, jb, &one, A, shiftA, lda, strideA,
                                                A, shiftA + idx2D(0,j,lda), lda, strideA, batch_count, work + shiftW, workArr);

            hipLaunchKernelGGL(trtri_kernel_large<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                            diag, n, j, jb, A, shiftA, lda, strideA, info, work);
        }

        rocblas_set_pointer_mode(handle,old_mode);
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_TRTRI_H */
