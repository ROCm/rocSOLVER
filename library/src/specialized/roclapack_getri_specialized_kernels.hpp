/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
    getri_kernel_small(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_int shiftP,
                       const rocblas_stride strideP,
                       rocblas_int* info,
                       const bool complete,
                       const bool pivot)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= n)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
    rocblas_int* ipiv;
    if(pivot)
        ipiv = load_ptr_batch<rocblas_int>(ipivA, b, shiftP, strideP);

    // shared memory (for communication between threads in group)
    __shared__ T common[TRTRI_MAX_COLS];
    T temp;
    rocblas_int jp;

    // read corresponding row from global memory in local array
    T rA[TRTRI_MAX_COLS];
    for(int j = 0; j < n; ++j)
        rA[j] = A[i + j * lda];

    if(complete)
    {
        __shared__ T diag[TRTRI_MAX_COLS];
        __shared__ rocblas_int _info;

        // compute info
        if(i == 0)
            _info = 0;
        __syncthreads();
        if(rA[i] == 0)
        {
            rocblas_int _info_temp = _info;
            while(_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
        __syncthreads();

        if(i == 0)
            info[b] = _info;
        if(_info != 0)
            return;

        //--- TRTRI ---
        // diagonal element
        rA[i] = 1.0 / rA[i];
        diag[i] = -rA[i];

        // compute element i of each column j
        for(rocblas_int j = 1; j < n; j++)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i < j)
            {
                temp = 0;

                for(rocblas_int ii = i; ii < j; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
            __syncthreads();
        }
    }

    if(info[b] != 0)
        return;

    //--- GETRI ---
    for(rocblas_int j = n - 2; j >= 0; j--)
    {
        // extract lower triangular column (copy_and_zero)
        if(i > j)
        {
            common[i] = rA[j];
            rA[j] = 0;
        }
        __syncthreads();

        // update column j (gemv)
        temp = 0;

        for(rocblas_int ii = j + 1; ii < n; ii++)
            temp += rA[ii] * common[ii];

        rA[j] -= temp;
    }

    // apply pivots (getri_pivot)
    if(pivot)
    {
        for(rocblas_int j = n - 2; j >= 0; j--)
        {
            jp = ipiv[j] - 1;
            if(jp != j)
                swap(rA[j], rA[jp]);
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
rocblas_status getri_run_small(rocblas_handle handle,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool complete,
                               const bool pivot)
{
    dim3 grid(batch_count, 1, 1);
    dim3 block(TRTRI_MAX_COLS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((getri_kernel_small<T>), grid, block, 0, stream, n, A, shiftA, lda,
                            strideA, ipiv, shiftP, strideP, info, complete, pivot);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GETRI_SMALL(T, U)                                              \
    template rocblas_status getri_run_small<T, U>(                                 \
        rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA, \
        const rocblas_int lda, const rocblas_stride strideA, rocblas_int* ipiv,    \
        const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int* info, \
        const rocblas_int batch_count, const bool complete, const bool pivot)

ROCSOLVER_END_NAMESPACE
