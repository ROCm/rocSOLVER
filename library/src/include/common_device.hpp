/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef COMMON_DEVICE_H
#define COMMON_DEVICE_H

#include <hip/hip_runtime.h>

template<typename T, typename U>
__forceinline__ __global__ void reset_info(T *info, const rocblas_int n, U val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (idx < n)
        info[idx] = T(val);
}

template<typename T, typename U>
__forceinline__ __global__ void reset_batch_info(T *info, const rocblas_int stride, const rocblas_int n, U val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    T* inf = info + b * stride;
    if (idx < n)
        inf[idx] = T(val);
}

template<typename T>
__forceinline__ __device__ __host__ T* load_ptr_batch(T* p, rocblas_int shift, rocblas_int batch, rocblas_int stride) {
    return p + batch * stride + shift;
}

template<typename T>
__forceinline__ __device__ __host__ T* load_ptr_batch(T *const p[], rocblas_int shift, rocblas_int batch, rocblas_int stride) {
    return p[batch] + shift;
}

template<typename T>
__forceinline__ __global__ void get_array(T** out, T* in, rocblas_int stride, rocblas_int batch) 
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (b < batch)
        out[b] = in + b*stride;
}

template <typename T, typename U>
__forceinline__ __global__ void setdiag(const rocblas_int j, U A, 
                        const rocblas_int shiftA, const rocblas_int lda, const rocblas_int strideA,
                        T *ipiv, const rocblas_int strideP)
{
    const auto b = hipBlockIdx_x;
    T *Ap = load_ptr_batch<T>(A,shiftA,b,strideA);
    T *tau = ipiv + b*strideP;

    T t = -tau[j];
    tau[j] = t; 
    Ap[j + j*lda] = 1.0 + t;
}

template <typename T>
__forceinline__ __global__ void restau(const rocblas_int k, T *ipiv, const rocblas_int strideP)
{
    const auto blocksizex = hipBlockDim_x;
    const auto b = hipBlockIdx_y;
    T *tau = ipiv + b*strideP;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;

    if (i < k)
        tau[i] = -tau[i];
}

template <typename T, typename U>
__forceinline__ __global__ void set_one_diag(T* diag, U A, const rocblas_int shifta, const rocblas_int stridea)
{
    int b = hipBlockIdx_x;

    T* d = load_ptr_batch<T>(A,shifta,b,stridea);
    diag[b] = d[0];
    d[0] = T(1);
}

template <typename T, typename U>
__forceinline__ __global__ void restore_diag(T* diag, U A, const rocblas_int shifta, const rocblas_int stridea)
{
    int b = hipBlockIdx_x;

    T* d = load_ptr_batch<T>(A,shifta,b,stridea);

    d[0] = diag[b];
}




#endif
