/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef COMMON_DEVICE_H
#define COMMON_DEVICE_H

#include "ideal_sizes.hpp"
#include "libcommon.hpp"
#include <hip/hip_runtime.h>

// **********************************************************
// device functions that are used by many kernels
// **********************************************************

template <typename S, typename T, std::enable_if_t<!is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val);
}

template <typename S, typename T, std::enable_if_t<is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return asum(val);
}

template <typename T>
__device__ void swap(const rocblas_int n, T* a, const rocblas_int inca, T* b, const rocblas_int incb)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        T orig = a[inca * tid];
        a[inca * tid] = b[incb * tid];
        b[incb * tid] = orig;
    }
}

/** SWAPVECT device function swap vectors a and b of dimension n **/
template <typename T>
__device__ void
    swapvect(const rocblas_int n, T* a, const rocblas_int inca, T* b, const rocblas_int incb)
{
    T orig;
    for(rocblas_int i = 0; i < n; ++i)
    {
        orig = a[inca * i];
        a[inca * i] = b[incb * i];
        b[incb * i] = orig;
    }
}

// **********************************************************
// GPU kernels that are used by many rocsolver functions
// **********************************************************

template <typename T, typename U>
__global__ void reset_info(T* info, const rocblas_int n, U val)
{
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx < n)
        info[idx] = T(val);
}

template <typename T, typename S, typename U>
__global__ void reset_batch_info(U info, const rocblas_stride stride, const rocblas_int n, S val)
{
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    T* inf = load_ptr_batch<T>(info, b, 0, stride);
    if(idx < n)
        inf[idx] = T(val);
}

template <typename T>
__global__ void get_array(T** out, T* in, rocblas_stride stride, rocblas_int batch)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch)
        out[b] = in + b * stride;
}

template <typename T, typename U>
__global__ void shift_array(T** out, U in, rocblas_int shift, rocblas_int batch)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch)
        out[b] = in[b] + shift;
}

template <typename T, typename U>
__global__ void subtract_tau(const rocblas_int i,
                             const rocblas_int j,
                             U A,
                             const rocblas_int shiftA,
                             const rocblas_int lda,
                             const rocblas_stride strideA,
                             T* ipiv,
                             const rocblas_stride strideP)
{
    const auto b = hipBlockIdx_x;
    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* tau = ipiv + b * strideP;

    T t = -(*tau);
    *tau = t;
    Ap[i + j * lda] = 1.0 + t;
}

template <typename T>
__global__ void restau(const rocblas_int k, T* ipiv, const rocblas_stride strideP)
{
    const auto blocksizex = hipBlockDim_x;
    const auto b = hipBlockIdx_y;
    T* tau = ipiv + b * strideP;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;

    if(i < k)
        tau[i] = -tau[i];
}

template <typename T, typename S, typename U, std::enable_if_t<!is_complex<T> || is_complex<S>, int> = 0>
__global__ void set_diag(S* D,
                         const rocblas_int shiftd,
                         const rocblas_stride strided,
                         U A,
                         const rocblas_int shifta,
                         const rocblas_int lda,
                         const rocblas_stride stridea,
                         const rocblas_int n,
                         bool set_one)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
    {
        d[i] = a[j];
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T, typename S, typename U, std::enable_if_t<is_complex<T> && !is_complex<S>, int> = 0>
__global__ void set_diag(S* D,
                         const rocblas_int shiftd,
                         const rocblas_stride strided,
                         U A,
                         const rocblas_int shifta,
                         const rocblas_int lda,
                         const rocblas_stride stridea,
                         const rocblas_int n,
                         bool set_one)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
    {
        d[i] = a[j].real();
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T, typename S, typename U>
__global__ void restore_diag(S* D,
                             const rocblas_int shiftd,
                             const rocblas_stride strided,
                             U A,
                             const rocblas_int shifta,
                             const rocblas_int lda,
                             const rocblas_stride stridea,
                             const rocblas_int n)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
        a[j] = d[i];
}

template <typename T, typename U>
__global__ void set_zero(const rocblas_int m,
                         const rocblas_int n,
                         U A,
                         const rocblas_int shiftA,
                         const rocblas_int lda,
                         const rocblas_stride strideA)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < m && j < n)
    {
        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Ap[i + j * lda] = 0.0;
    }
}

template <typename T, typename U>
__global__ void copyshift_right(const bool copy,
                                const rocblas_int dim,
                                U A,
                                const rocblas_int shiftA,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                T* W,
                                const rocblas_int shiftW,
                                const rocblas_int ldw,
                                const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < dim && j < dim && j <= i)
    {
        rocblas_int offset = j * (j + 1) / 2; // to acommodate in smaller array W

        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

        if(copy)
        {
            // copy columns
            Wp[i + j * ldw - offset] = (j == 0 ? 0.0 : Ap[i + 1 + (j - 1) * lda]);
        }
        else
        {
            // shift columns to the right
            Ap[i + 1 + j * lda] = Wp[i + j * ldw - offset];

            // make first row the identity
            if(i == j)
            {
                Ap[(j + 1) * lda] = 0.0;
                if(i == 0)
                    Ap[0] = 1.0;
            }
        }
    }
}

template <typename T, typename U>
__global__ void copyshift_left(const bool copy,
                               const rocblas_int dim,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* W,
                               const rocblas_int shiftW,
                               const rocblas_int ldw,
                               const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W

        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

        if(copy)
        {
            // copy columns
            Wp[i + j * ldw - offset] = (j == dim - 1 ? 0.0 : Ap[i + (j + 2) * lda]);
        }
        else
        {
            // shift columns to the left
            Ap[i + (j + 1) * lda] = Wp[i + j * ldw - offset];

            // make last row the identity
            if(i == j)
            {
                Ap[dim + j * lda] = 0.0;
                if(i == 0)
                    Ap[dim + dim * lda] = 1.0;
            }
        }
    }
}

template <typename T, typename U>
__global__ void copyshift_down(const bool copy,
                               const rocblas_int dim,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* W,
                               const rocblas_int shiftW,
                               const rocblas_int ldw,
                               const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W

        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

        if(copy)
        {
            // copy rows
            Wp[i + j * ldw - offset] = (i == 0 ? 0.0 : Ap[i - 1 + (j + 1) * lda]);
        }
        else
        {
            // shift rows downward
            Ap[i + (j + 1) * lda] = Wp[i + j * ldw - offset];

            // make first column the identity
            if(i == j)
            {
                Ap[i + 1] = 0.0;
                if(j == 0)
                    Ap[0] = 1.0;
            }
        }
    }
}

#endif
