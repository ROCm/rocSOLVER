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
#ifndef ROCLAPACK_GEBLTTRF_NPVT_STRIDED_BATCHED_HPP
#define ROCLAPACK_GEBLTTRF_NPVT_STRIDED_BATCHED_HPP

#include "roclapack_geblttrf_npvt_strided_batched_small.hpp"


template<typename T>
rocblas_status rocsolver_geblttrf_npvt_strided_batched_template(
                    rocblas_handle handle,
                    const rocblas_int nb,
                    const rocblas_int nblocks,
                    T *A,
                    const rocblas_int lda,
                    const rocblas_stride strideA,
                    T *B,
                    const rocblas_int ldb,
                    const rocblas_stride strideB,
                    T *C,
                    const rocblas_int ldc,
                    const rocblas_stride strideC,
                    rocblas_int info[],
                    const rocblas_int batch_count 
                    )
{


    return( rocsolver_geblttrf_npvt_strided_batched_small_template(
                 handle,
                 nb, nblocks, 
                 A, lda, strideA,
                 B, ldb, strideB,
                 C, ldc, strideC,
                 info,
                 batch_count ) );

}
#endif
