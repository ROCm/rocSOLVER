/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef _ROCBLAS_DEVICE_FUNCTIONS_HPP_
#define _ROCBLAS_DEVICE_FUNCTIONS_HPP_

#include "common_device.hpp"

template <typename T>
__device__ void trtri_kernel_upper(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming upper triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = 1; j < n; j++)
    {
        if(i < j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i < j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i + 1; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trtri_kernel_lower(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming lower triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = n - 2; j >= 0; j--)
    {
        if(i > j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i > j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i - 1; ii > j; ii--)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_upper(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, upper triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = i + 1; k < m; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_lower(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, lower triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = 0; k < i; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_upper(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, upper triangular matrix from the right
    T ajj, bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = 0; k < j; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_lower(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, lower triangular matrix from the right
    T ajj, bij;
    for(int j = n - 1; j >= 0; j--)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = j + 1; k < n; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void gemv_kernel(const rocblas_int m,
                            const rocblas_int n,
                            T* alpha,
                            T* a,
                            const rocblas_int lda,
                            T* x,
                            const rocblas_int incX,
                            T* beta,
                            T* y,
                            const rocblas_int incY)
{
    // gemv kernel assuming no transpose
    T yi;
    for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
    {
        yi = 0;

        if(*alpha != 0)
        {
            for(int k = 0; k < n; k++)
                yi += a[i + k * lda] * x[k * incX];
        }

        y[i * incY] = *alpha * yi + *beta * y[i * incY];
    }
    __syncthreads();
}

template <typename T>
__device__ void gemm_kernel(const rocblas_int m,
                            const rocblas_int n,
                            const rocblas_int k,
                            T* alpha,
                            T* a,
                            const rocblas_int lda,
                            T* b,
                            const rocblas_int ldb,
                            T* beta,
                            T* c,
                            const rocblas_int ldc)
{
    // gemm kernel assuming no transpose
    T cij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            cij = 0;

            if(*alpha != 0)
            {
                for(int l = 0; l < k; l++)
                    cij += a[i + l * lda] * b[l + j * ldb];
            }

            c[i + j * ldc] = *alpha * cij + *beta * c[i + j * ldc];
        }
        __syncthreads();
    }
}

#endif // _ROCBLAS_DEVICE_FUNCTIONS_HPP_
