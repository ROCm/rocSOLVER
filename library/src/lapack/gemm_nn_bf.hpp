/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef GEMM_NN_BF_HPP
#define GEMM_NN_BF_HPP

#include "geblt_common.h"

template <typename T>
DEVICE_FUNCTION void gemm_nn_bf_device(rocblas_int batchCount,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int k,
                                       const T alpha,
                                       T * A_,
                                       const rocblas_int lda,
                                       T * B_,
                                       const rocblas_int ldb,
                                       const T beta,
                                       T* C_,
                                       const rocblas_int ldc)
{
#define A(iv, ia, ja) A_[indx3f(iv, ia, ja, batchCount, lda)]
#define B(iv, ib, jb) B_[indx3f(iv, ib, jb, batchCount, ldb)]
#define C(iv, ic, jc) C_[indx3f(iv, ic, jc, batchCount, ldc)]

#ifdef USE_GPU
    rocblas_int const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = (gridDim.x * blockDim.x);
#else
    rocblas_int const iv_start = 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = 1;
#endif

    T const zero = 0;

    bool const is_beta_zero = (beta == zero);

    for(rocblas_int jc = 1; jc <= n; jc++)
    {
        for(rocblas_int ic = 1; ic <= m; ic++)
        {
            for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
            {
                T cij = zero;
                for(rocblas_int ja = 1; ja <= k; ja++)
                {
                    cij += A(iv, ic, ja) * B(iv, ja, jc);
                };

                if(is_beta_zero)
                {
                    C(iv, ic, jc) = alpha * cij;
                }
                else
                {
                    C(iv, ic, jc) = beta * C(iv, ic, jc) + alpha * cij;
                };
            }; // end for iv

            SYNCTHREADS;

        }; // end for ic
    }; // end for jc
}

#undef A
#undef B
#undef C

#endif
