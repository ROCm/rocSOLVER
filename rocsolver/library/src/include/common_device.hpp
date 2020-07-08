/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef COMMON_DEVICE_H
#define COMMON_DEVICE_H

#include <hip/hip_runtime.h>
#include "ideal_sizes.hpp"
#include "libcommon.hpp"


// **********************************************************
// device functions that are used by many kernels
// **********************************************************


template <typename T>
__device__ void swap(const rocblas_int n, T *a, const rocblas_int inca,
                     T *b, const rocblas_int incb)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid < n)
    {
        T orig = a[inca * tid];
        a[inca * tid] = b[incb * tid];
        b[incb * tid] = orig;
    }
}

/** SWAPVECT device function swap vectors a and b of dimension n **/
template <typename T>
__device__ void swapvect(const rocblas_int n, T *a, const rocblas_int inca,
                        T *b, const rocblas_int incb)
{
    T orig;
    for (rocblas_int i = 0; i < n; ++i)
    {
        orig = a[inca * i];
        a[inca * i] = b[incb * i];
        b[incb * i] = orig;
    }
}


// **********************************************************
// GPU kernels that are used by many rocsolver functions
// **********************************************************


template<typename T, typename U>
__global__ void reset_info(T *info, const rocblas_int n, U val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (idx < n)
        info[idx] = T(val);
}

template<typename T, typename S, typename U>
__global__ void reset_batch_info(U info, const rocblas_stride stride, const rocblas_int n, S val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    T* inf = load_ptr_batch<T>(info,b,0,stride);
    if (idx < n)
        inf[idx] = T(val);
}

template<typename T>
__global__ void get_array(T** out, T* in, rocblas_stride stride, rocblas_int batch) 
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (b < batch)
        out[b] = in + b*stride;
}

template <typename T, typename U>
__global__ void setdiag(const rocblas_int j, U A, 
                        const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                        T *ipiv, const rocblas_stride strideP)
{
    const auto b = hipBlockIdx_x;
    T *Ap = load_ptr_batch<T>(A,b,shiftA,strideA);
    T *tau = ipiv + b*strideP;

    T t = -tau[j];
    tau[j] = t; 
    Ap[j + j*lda] = 1.0 + t;
}

template <typename T>
__global__ void restau(const rocblas_int k, T *ipiv, const rocblas_stride strideP)
{
    const auto blocksizex = hipBlockDim_x;
    const auto b = hipBlockIdx_y;
    T *tau = ipiv + b*strideP;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;

    if (i < k)
        tau[i] = -tau[i];
}


template <typename T, typename S, typename U, std::enable_if_t<!is_complex<T> || is_complex<S>, int> = 0>
__global__ void set_diag(S* D, const rocblas_int shiftd, const rocblas_stride strided,
                         U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                         const rocblas_int n, bool set_one)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i*lda;

    S* d = load_ptr_batch<S>(D,b,shiftd,strided);
    T* a = load_ptr_batch<T>(A,b,shifta,stridea);

    if (i < n)
    {
        d[i] = a[j];
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T, typename S, typename U, std::enable_if_t<is_complex<T> && !is_complex<S>, int> = 0>
__global__ void set_diag(S* D, const rocblas_int shiftd, const rocblas_stride strided,
                         U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                         const rocblas_int n, bool set_one)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i*lda;

    S* d = load_ptr_batch<S>(D,b,shiftd,strided);
    T* a = load_ptr_batch<T>(A,b,shifta,stridea);

    if (i < n)
    {
        d[i] = a[j].real();
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T, typename S, typename U>
__global__ void restore_diag(S* D, const rocblas_int shiftd, const rocblas_stride strided,
                             U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                             const rocblas_int n)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i*lda;

    S* d = load_ptr_batch<S>(D,b,shiftd,strided);
    T* a = load_ptr_batch<T>(A,b,shifta,stridea);

    if (i < n)
        a[j] = d[i];
}


#endif
