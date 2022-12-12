/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSOLVER_GEBLTTRS_NPVT_STRIDED_BATCHED_HPP
#define ROCSOLVER_GEBLTTRS_NPVT_STRIDED_BATCHED_HPP

#include "roclapack_geblttrs_npvt_strided_batched_small.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrs_npvt_strided_batched_template(rocblas_handle handle,
                                                            const I nb,
                                                            const I nblocks,
                                                            const I nrhs,
                                                            T* A_,
                                                            const I lda,
                                                            const Istride strideA,
                                                            T* B_,
                                                            const I ldb,
                                                            const Istride strideB,
                                                            T* C_,
                                                            const I ldc,
                                                            const Istride strideC,

                                                            T* X_,
                                                            const I ldx,
                                                            const Istride strideX,
                                                            const I batch_count)
{
        return (rocsolver_geblttrs_npvt_strided_batched_small_template<T, I, Istride>(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, X_,
            ldx, strideX, batch_count));
};

#endif
