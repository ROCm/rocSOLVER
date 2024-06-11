/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(TRTRI_MAX_COLS)
    trti2_kernel_small(const rocblas_fill uplo,
                       const rocblas_diagonal diagtype,
                       const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= n)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);

    // read corresponding row from global memory in local array
    T rA[TRTRI_MAX_COLS];
    for(int j = 0; j < n; ++j)
        rA[j] = A[i + j * lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[TRTRI_MAX_COLS];
    __shared__ T diag[TRTRI_MAX_COLS];
    T temp;

    // diagonal element
    const bool unit = (diagtype == rocblas_diagonal_unit);
    if(unit)
    {
        diag[i] = -1.0;
    }
    else
    {
        rA[i] = 1.0 / rA[i];
        diag[i] = -rA[i];
    }

    // compute element i of each column j
    if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int j = 1; j < n; j++)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i < j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = i + 1; ii < j; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
        }
    }
    else
    {
        for(rocblas_int j = n - 2; j >= 0; j--)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i > j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = j + 1; ii < i; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
        }
    }

    // write results to global memory from local array
    for(int j = 0; j < n; j++)
        A[i + j * lda] = rA[j];
}

/*************************************************************
    Launchers of specilized  kernels
*************************************************************/

template <typename T, typename U>
void trti2_run_small(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_diagonal diag,
                     const rocblas_int n,
                     U A,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const rocblas_int batch_count)
{
    dim3 grid(batch_count, 1, 1);
    dim3 block(TRTRI_MAX_COLS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((trti2_kernel_small<T>), grid, block, 0, stream, uplo, diag, n, A,
                            shiftA, lda, strideA);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_TRTI2_SMALL(T, U)                                                          \
    template void trti2_run_small<T, U>(rocblas_handle handle, const rocblas_fill uplo,        \
                                        const rocblas_diagonal diag, const rocblas_int n, U A, \
                                        const rocblas_int shiftA, const rocblas_int lda,       \
                                        const rocblas_stride strideA, const rocblas_int batch_count)

ROCSOLVER_END_NAMESPACE
