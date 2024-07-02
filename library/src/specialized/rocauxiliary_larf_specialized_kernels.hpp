/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LARF_SSKER_THREADS)
    larf_left_kernel_small(const I m,
                           const I n,
                           U xx,
                           const rocblas_stride shiftX,
                           const I incX,
                           const rocblas_stride strideX,
                           const T* tauA,
                           const rocblas_stride strideP,
                           U AA,
                           const rocblas_stride shiftA,
                           const I lda,
                           const rocblas_stride strideA)
{
    I bid = hipBlockIdx_x;
    I rid = hipThreadIdx_x;
    I cid = hipBlockIdx_y;

    // select batch instance
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const T* tau = tauA + bid * strideP;

    // // shared variables
    // __shared__ T sval[LARF_SSKER_THREADS];

    I start = (incX > 0 ? 0 : (m - 1) * -incX);
    for(I j = 0; j < n; j += 1)
    {
        T temp = 0;
        for(I i = 0; i < m; i++)
        {
            temp += conj(A[i + j * lda]) * x[start + i * incX];
        }

        temp = -tau[0] * conj(temp);
        for(I i = 0; i < m; i++)
        {
            A[i + j * lda] += temp * x[start + i * incX];
        }
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LARF_SSKER_THREADS)
    larf_right_kernel_small(const I m,
                            const I n,
                            U xx,
                            const rocblas_stride shiftX,
                            const I incX,
                            const rocblas_stride strideX,
                            const T* tauA,
                            const rocblas_stride strideP,
                            U AA,
                            const rocblas_stride shiftA,
                            const I lda,
                            const rocblas_stride strideA)
{
    I bid = hipBlockIdx_x;
    I cid = hipThreadIdx_x;
    I rid = hipBlockIdx_y;

    // select batch instance
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const T* tau = tauA + bid * strideP;

    // // shared variables
    // __shared__ T sval[LARF_SSKER_THREADS];

    I start = (incX > 0 ? 0 : (n - 1) * -incX);
    for(I i = 0; i < m; i += 1)
    {
        T temp = 0;
        for(I j = 0; j < n; j++)
        {
            temp += A[i + j * lda] * x[start + j * incX];
        }

        temp = -tau[0] * temp;
        for(I j = 0; j < n; j++)
        {
            A[i + j * lda] += temp * conj(x[start + j * incX]);
        }
    }
}

/*************************************************************
    Launchers of specialized  kernels
*************************************************************/

template <typename T, typename I, typename U>
rocblas_status larf_run_small(rocblas_handle handle,
                              const rocblas_side side,
                              const I m,
                              const I n,
                              U x,
                              const rocblas_stride shiftX,
                              const I incX,
                              const rocblas_stride strideX,
                              const T* tau,
                              const rocblas_stride strideP,
                              U A,
                              const rocblas_stride shiftA,
                              const I lda,
                              const rocblas_stride strideA,
                              const I batch_count)
{
    // dim3 grid(batch_count, LARF_SSKER_BLOCKS, 1);
    // dim3 block(LARF_SSKER_THREADS, 1, 1);
    dim3 grid(batch_count, 1, 1);
    dim3 block(1, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(side == rocblas_side_left)
        ROCSOLVER_LAUNCH_KERNEL(larf_left_kernel_small<T>, grid, block, 0, stream, m, n, x, shiftX,
                                incX, strideX, tau, strideP, A, shiftA, lda, strideA);
    else
        ROCSOLVER_LAUNCH_KERNEL(larf_right_kernel_small<T>, grid, block, 0, stream, m, n, x, shiftX,
                                incX, strideX, tau, strideP, A, shiftA, lda, strideA);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_LARF_SMALL(T, I, U)                                                        \
    template rocblas_status larf_run_small<T, I, U>(                                           \
        rocblas_handle handle, const rocblas_side side, const I m, const I n, U x,             \
        const rocblas_stride shiftX, const I incX, const rocblas_stride strideX, const T* tau, \
        const rocblas_stride strideP, U A, const rocblas_stride shiftA, const I lda,           \
        const rocblas_stride strideA, const I batch_count)

ROCSOLVER_END_NAMESPACE
