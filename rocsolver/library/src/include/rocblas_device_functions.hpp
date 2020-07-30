/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef _ROCBLAS_DEVICE_FUNCTIONS_HPP_
#define _ROCBLAS_DEVICE_FUNCTIONS_HPP_

#include "common_device.hpp"


template <typename T>
__device__ void trmm_kernel_left_upper(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                       T *a, const rocblas_int lda, T *b, const rocblas_int ldb, T *w)
{
    // trmm kernel assuming no transpose, upper triangular matrix from the left
    // min dim for w is m
    T bij;
    for (int j = 0; j < n; j++)
    {
        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();
            
        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * b[i + j * ldb];

            for (int k = i + 1; k < m; k++)
                bij += a[i + k * lda] * w[k];
            
            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}


template <typename T>
__device__ void trsm_kernel_right_upper(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                        T *a, const rocblas_int lda, T *b, const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, upper triangular matrix from the right
    T ajj, bij;
    for (int j = 0; j < n; j++)
    {
        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = *alpha * b[i + j * ldb];

            for (int k = 0; k < j; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];
            
            b[i + j * ldb] = bij;
        }
        __syncthreads();

        if (diag == rocblas_diagonal_non_unit)
        {
            ajj = 1.0 / a[j + j * lda];
            __syncthreads();

            for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
                b[i + j * ldb] *= ajj;
            __syncthreads();
        }
    }
}

template <typename T>
__device__ void trsm_kernel_right_lower(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                        T *a, const rocblas_int lda, T *b, const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, lower triangular matrix from the right
    T ajj, bij;
    for (int j = n - 1; j >= 0; j--)
    {
        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = *alpha * b[i + j * ldb];

            for (int k = j + 1; k < n; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];
            
            b[i + j * ldb] = bij;
        }
        __syncthreads();

        if (diag == rocblas_diagonal_non_unit)
        {
            ajj = 1.0 / a[j + j * lda];
            __syncthreads();

            for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
                b[i + j * ldb] *= ajj;
            __syncthreads();
        }
    }
}


template <typename T>
__device__ void gemv_kernel(const rocblas_int m, const rocblas_int n, T* alpha,
                            T *a, const rocblas_int lda, T *x, const rocblas_int incX,
                            T* beta, T *y, const rocblas_int incY)
{
    // gemv kernel assuming no transpose
    if (*beta != 1)
    {
        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            y[i * incY] *= *beta;
        __syncthreads();
    }

    if (*alpha == 0)
        return;

    T temp;
    for (int j = 0; j < n; j++)
    {
        temp = *alpha * x[j * incX];

        for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            y[i * incY] += temp * a[i + j * lda];
        
        __syncthreads();
    }
}


template <typename T>
__device__ void gemm_kernel(const rocblas_int m, const rocblas_int n, const rocblas_int k, T* alpha,
                            T *a, const rocblas_int lda, T *b, const rocblas_int ldb,
                            T* beta, T *c, const rocblas_int ldc)
{
    // gemm kernel assuming no transpose
    T temp;
    for (int j = 0; j < n; j++)
    {
        if (*beta != 1)
        {
            for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
                c[i + j * lda] *= *beta;
            __syncthreads();
        }

        for (int l = 0; l < k; l++)
        {
            temp = *alpha * b[l + j * ldb];

            for (int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
                c[i + j * ldc] += temp * a[i + l * lda];
                
            __syncthreads();
        }
    }
}


#endif // _ROCBLAS_DEVICE_FUNCTIONS_HPP_
