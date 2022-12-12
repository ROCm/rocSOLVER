

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
#ifndef GEMM_NN_FIXED_HPP
#define GEMM_NN_FIXED_HPP

#include "geblt_common.h"

template <typename T, int const M, int const N>
DEVICE_FUNCTION void gemm_nn_fixed_device(
                                          const rocblas_int  k,
                                          const T  alpha,
                                          T *  A_,
                                          const rocblas_int  lda,
                                          T *  B_,
                                          const rocblas_int  ldb,
                                          const T  beta,
                                          T* C_,
                                          const rocblas_int  ldc)
{
#define A(ia, ja) A_[indx2f(ia, ja, lda)]
#define B(ib, jb) B_[indx2f(ib, jb, ldb)]
#define C(ic, jc) C_[indx2f(ic, jc, ldc)]
    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

#pragma unroll
    for(rocblas_int jc = 1; jc <= N; jc++)
    {
#pragma unroll
        for(rocblas_int ic = 1; ic <= M; ic++)
        {
            T cij = zero;
            for(rocblas_int ja = 1; ja <= k; ja++)
            {
                cij += A(ic, ja) * B(ja, jc);
            };
            if(is_beta_zero)
            {
                C(ic, jc) = alpha * cij;
            }
            else
            {
                C(ic, jc) = beta * C(ic, jc) + alpha * cij;
            };
        };
    };
}

#undef A
#undef B
#undef C

#endif
