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
#ifndef GEMM_NN_HPP
#define GEMM_NN_HPP

#include "geblt_common.h"
#include "gemm_nn_fixed.hpp"

template <typename T, typename I>
DEVICE_FUNCTION void gemm_nn_device(
                                    const I  m,
                                    const I  n,
                                    const I  k,
                                    const T  alpha,
                                    T * A_,
                                    const I  lda,
                                    T *  B_,
                                    const I  ldb,
                                    const T  beta,
                                    T* C_,
                                    const I  ldc)
{
#define A(ia, ja) A_[indx2f(ia, ja, lda)]
#define B(ib, jb) B_[indx2f(ib, jb, ldb)]
#define C(ic, jc) C_[indx2f(ic, jc, ldc)]

#define FIXED_CASE(M, N)                                                              \
    {                                                                                 \
        if((m == (M)) && (n == (N)))                                                  \
        {                                                                             \
            gemm_nn_fixed_device<T, M, N>(k, alpha, A_, lda, B_, ldb, beta, C_, ldc); \
        };                                                                            \
        return;                                                                       \
    }

    // check for special cases

    FIXED_CASE(1, 1);
    FIXED_CASE(2, 2);
    FIXED_CASE(3, 3);
    FIXED_CASE(4, 4);

    // code for general case

    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

    for(I jc = 1; jc <= n; jc++)
    {
        for(I ic = 1; ic <= m; ic++)
        {
            T cij = zero;
            for(I ja = 1; ja <= k; ja++)
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

#undef FIXED_CASE
#endif
