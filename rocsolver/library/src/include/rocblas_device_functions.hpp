/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef _ROCBLAS_DEVICE_FUNCTIONS_HPP_
#define _ROCBLAS_DEVICE_FUNCTIONS_HPP_

#include "common_device.hpp"


template <typename T, typename U, typename V>
__global__ void trmm_kernel_left_upper(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                       U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                       V B, const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB)
{
    // trmm kernel assuming no transpose, upper triangular matrix from the left
    int batch = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,batch,shiftA,strideA);
    T* b = load_ptr_batch<T>(B,batch,shiftB,strideB);

    T temp;
    for (int j = hipThreadIdx_y; j < n; j += hipBlockDim_y)
    {
        for (int k = 0; k < m; k++)
        {
            temp = *alpha * b[k + j * ldb];

            for (int i = 0; i < k; i++)
                b[i + j * ldb] += temp * a[i + k * lda];

            if (diag == rocblas_diagonal_non_unit)
                temp *= a[k + k * lda];
            
            b[k + j * ldb] = temp;
        }
        __syncthreads();
    }
}


template <typename T, typename U, typename V>
__global__ void trsm_kernel_right_upper(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        V B, const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB)
{
    // trsm kernel assuming no transpose, upper triangular matrix from the right
    int batch = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,batch,shiftA,strideA);
    T* b = load_ptr_batch<T>(B,batch,shiftB,strideB);

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

template <typename T, typename U, typename V>
__global__ void trsm_kernel_right_lower(const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, T* alpha,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        V B, const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB)
{
    // trsm kernel assuming no transpose, lower triangular matrix from the right
    int batch = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,batch,shiftA,strideA);
    T* b = load_ptr_batch<T>(B,batch,shiftB,strideB);

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


template <typename T, typename U, typename V, typename W>
__global__ void gemv_kernel(const rocblas_int m, const rocblas_int n, T* alpha,
                            U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                            V X, const rocblas_int shiftX, const rocblas_int incX, const rocblas_stride strideX, T* beta,
                            W Y, const rocblas_int shiftY, const rocblas_int incY, const rocblas_stride strideY)
{
    // gemv kernel assuming no transpose
    int batch = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,batch,shiftA,strideA);
    T* x = load_ptr_batch<T>(X,batch,shiftX,strideX);
    T* y = load_ptr_batch<T>(Y,batch,shiftY,strideY);
    
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


template <typename T, typename U, typename V, typename W>
__global__ void gemm_kernel(const rocblas_int m, const rocblas_int n, const rocblas_int k, T* alpha,
                            U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                            V B, const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB, T* beta,
                            W C, const rocblas_int shiftC, const rocblas_int ldc, const rocblas_stride strideC)
{
    // gemm kernel assuming no transpose
    int batch = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,batch,shiftA,strideA);
    T* b = load_ptr_batch<T>(B,batch,shiftB,strideB);
    T* c = load_ptr_batch<T>(C,batch,shiftC,strideC);
    
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
