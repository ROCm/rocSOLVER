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
#ifndef GETRF_NPVT_BF_HPP
#define GETRF_NPVT_BF_HPP

#include "geblt_common.h"

template <typename T, typename I>
DEVICE_FUNCTION void
    getrf_npvt_bf_device(I const batchCount, I const m, I const n, T* A_, I const lda, I info[])
{
    I const min_mn = (m < n) ? m : n;
    T const one = 1;

#ifdef USE_GPU
    I const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    I const iv_end = batchCount;
    I const iv_inc = (gridDim.x * blockDim.x);
#else
    I const iv_start = 1;
    I const iv_end = batchCount;
    I const iv_inc = 1;
#endif

#define A(iv, i, j) A_[indx3f(iv, i, j, batchCount, lda)]

    T const zero = 0;

    for(I j = 1; j <= min_mn; j++)
    {
        I const jp1 = j + 1;

        for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
        {
            bool const is_diag_zero = (std::abs(A(iv, j, j)) == zero);
            T const Ujj_iv = is_diag_zero ? one : A(iv, j, j);
            info[iv - 1] = is_diag_zero && (info[iv - 1] == 0) ? j : info[iv - 1];

            for(I ia = jp1; ia <= m; ia++)
            {
                A(iv, ia, j) = A(iv, ia, j) / Ujj_iv;
            };
        };

        SYNCTHREADS;

        for(I ja = jp1; ja <= n; ja++)
        {
            for(I ia = jp1; ia <= m; ia++)
            {
                for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    A(iv, ia, ja) = A(iv, ia, ja) - A(iv, ia, j) * A(iv, j, ja);
                };
            };
        };

        SYNCTHREADS;
    };
}
#undef A

#endif
