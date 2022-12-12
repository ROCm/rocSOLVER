
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
#ifndef ROCSOLVER_GEBLTTRS_STRIDED_BATCHED_SMALL_H
#define ROCSOLVER_GEBLTTRS_STRIDED_BATCHED_SMALL_H

#include "geblt_common.h"
#include "geblttrs_npvt.hpp"

template <typename T, typename I, typename Istride>
GLOBAL_FUNCTION void rocsolver_geblttrs_npvt_strided_batched_kernel(
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
                                                                    const I batchCount

)
{
    auto const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto const i_start = thread_id;
    auto const i_inc = gridDim.x * blockDim.x;


    {
        for(I i = i_start; i < batchCount; i += i_inc)
        {
            Istride indxA = (strideA) * (i - 1);
            Istride indxB = (strideB) * (i - 1);
            Istride indxC = (strideC) * (i - 1);
            Istride indxX = (strideX) * (i - 1);

            I linfo = 0;
            geblttrs_npvt_device<T>(nb, nblocks, nrhs, &(A_[indxA]), lda, &(B_[indxB]), ldb,
                                    &(C_[indxC]), ldc, &(X_[indxX]), ldx, &linfo);
        };
    };
}

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrs_npvt_strided_batched_small_template(rocblas_handle handle,
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
                                                                      const I batchCount

)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const grid_dim = (batchCount + (GEBLT_BLOCK_DIM - 1)) / GEBLT_BLOCK_DIM;
    hipLaunchKernelGGL((rocsolver_geblttrs_npvt_strided_batched_kernel<T>), dim3(grid_dim),
                       dim3(GEBLT_BLOCK_DIM), 0, stream,

                       nb, nblocks, nrhs,

                       A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

                       X_, ldx, strideX, batchCount);

    return (rocblas_status_success);
}

#endif
