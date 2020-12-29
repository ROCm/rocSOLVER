/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "ideal_sizes.hpp"
#include "libcommon.hpp"
#include <hip/hip_runtime.h>

/*
 * ===========================================================================
 *    common location for device functions and kernels that are used across
 *    several rocSOLVER routines, excepting those device functions and kernels
 *    that reproduce LAPACK functionality (see lapack_device_functions.hpp).
 * ===========================================================================
 */

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

/** FIND_MAX_TRIDIAG finds the element with the largest magnitude in the
    tridiagonal matrix **/
template <typename T>
__device__ T find_max_tridiag(const rocblas_int start, const rocblas_int end, T* D, T* E)
{
    T anorm = abs(D[end]);
    for(int i = start; i < end; i++)
        anorm = max(anorm, max(abs(D[i]), abs(E[i])));
    return anorm;
}

/** SCALE_TRIDIAG scales the elements of the tridiagonal matrix by a given
    scale factor **/
template <typename T>
__device__ void scale_tridiag(const rocblas_int start, const rocblas_int end, T* D, T* E, T scale)
{
    D[end] *= scale;
    for(int i = start; i < end; i++)
    {
        D[i] *= scale;
        E[i] *= scale;
    }
}

// **********************************************************
// GPU kernels that are used by many rocsolver functions
// **********************************************************

template <typename T, typename U>
__global__ void init_ident(const rocblas_int m,
                           const rocblas_int n,
                           U A,
                           const rocblas_int shiftA,
                           const rocblas_int lda,
                           const rocblas_stride strideA)
{
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto b = hipBlockIdx_z;

    if(i < m && j < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

        if(i == j)
            a[i + j * lda] = 1.0;
        else
            a[i + j * lda] = 0.0;
    }
}

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

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

    // make first row the identity
    if(i == 0 && j == 0 && !copy)
        Ap[0] = 1.0;

    if(i < dim && j < dim && j <= i)
    {
        rocblas_int offset = j * (j + 1) / 2; // to acommodate in smaller array W

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
                Ap[(j + 1) * lda] = 0.0;
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

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

    // make last row the identity
    if(i == 0 && j == 0 && !copy)
        Ap[dim + dim * lda] = 1.0;

    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W


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
                Ap[dim + j * lda] = 0.0;
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

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);
                
    // make first column the identity
    if(i == 0 && j == 0 && !copy)
        Ap[0] = 1.0;
    
    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W

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
                Ap[i + 1] = 0.0;
        }
    }
}

/** set_offdiag kernel copies the off-diagonal element of A, which is the non-zero element
    resulting by applying the Householder reflector to the working column, to E. Then set it
    to 1 to prepare for the application of the Householder reflector to the rest of the matrix **/
template <typename T, typename U, typename S, std::enable_if_t<!is_complex<T>, int> = 0>
__global__ void set_offdiag(const rocblas_int batch_count,
                            U A,
                            const rocblas_int shiftA,
                            const rocblas_stride strideA,
                            S* E,
                            const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* e = E + b * strideE;

        e[0] = a[0];
        a[0] = T(1);
    }
}

template <typename T, typename U, typename S, std::enable_if_t<is_complex<T>, int> = 0>
__global__ void set_offdiag(const rocblas_int batch_count,
                            U A,
                            const rocblas_int shiftA,
                            const rocblas_stride strideA,
                            S* E,
                            const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* e = E + b * strideE;

        e[0] = a[0].real();
        a[0] = T(1);
    }
}

/** scale_axpy kernel executes axpy to update tau computing the scalar alpha with other
    results in different memopry locations **/
template <typename T, typename U>
__global__ void scale_axpy(const rocblas_int n,
                           T* scl,
                           T* S,
                           const rocblas_stride strideS,
                           U A,
                           const rocblas_int shiftA,
                           const rocblas_stride strideA,
                           T* W,
                           const rocblas_int shiftW,
                           const rocblas_stride strideW)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < n)
    {
        T* v = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* w = load_ptr_batch<T>(W, b, shiftW, strideW);
        T* s = S + b * strideS;

        // scale
        T alpha = -0.5 * s[0] * scl[b];

        // axpy
        w[i] = alpha * v[i] + w[i];
    }
}

template <typename T, typename U>
__global__ void check_singularity(const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  rocblas_int* info)
{
    // Checks for singularities in the matrix and updates info to indicate where
    // the first singularity (if any) occurs
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ rocblas_int _info;

    if(hipThreadIdx_y == 0)
        _info = 0;
    __syncthreads();

    for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if(a[i + i * lda] == 0)
        {
            rocblas_int _info_temp = _info;
            while(_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
    }
    __syncthreads();

    if(hipThreadIdx_y == 0)
        info[b] = _info;
}
