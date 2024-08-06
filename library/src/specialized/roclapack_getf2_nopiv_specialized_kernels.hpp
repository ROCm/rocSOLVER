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
#include <algorithm>
#include <cmath>

ROCSOLVER_BEGIN_NAMESPACE
/**
 * ------------------------------------------------------
 * Perform LU factorization without pivoting for small n by n matrix.
 * The function executes in a single thread block.
 * ------------------------------------------------------
**/
template <typename T, typename I>
__device__ static void getf2_nopiv_simple(I const n, T* const A, I* const info)
{
    using S = decltype(std::real(T{}));
    auto const sfmin = get_safemin<S>();

    auto const lda = n;

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j0_start = tid;
    auto const j0_inc = nthreads;

    auto idx2D = [](I i, I j, I lda) { return (i + j * lda); };
    // ---------------------------------------------------
    // [  1       ]  * [ u11    vu12 ]  =  [ a11   va12]
    // [ vl21  L22]    [        U22' ]     [ va21, A22 ]
    //
    //
    // assume unit diagonal in Lower triangular matrix L
    //
    //   (1) 1 * u11 = a11 => u11 = a11
    //   (2) 1 * vu12 = va12 => vu12 = va12  (no change in 1st row)
    //
    //   (3) vl21 * u11 = va21 => vl21  va21 / u11, scale column
    //
    //   (4) A22 = A22 - vl21 * vu12
    //   (5) L22 * U22' = A22,   factorization, tail recursion
    // ---------------------------------------------------

    __syncthreads();

    T const zero = 0;
    for(I kcol = 0; kcol < n; kcol++)
    {
        auto const kk = idx2D(kcol, kcol, lda);
        auto const akk = A[kk];

        __syncthreads();

        bool const isok_akk = (akk != zero);
        T pivot_val = 1;
        if(isok_akk)
        {
            pivot_val = 1 / akk;
        }
        else
        {
            pivot_val = 1;
            if(tid == 0)
            {
                // Fortran 1-based index
                *info = (*info == 0) ? kcol + 1 : (*info);
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        //   (3) vl21 *u11  = va21 =>  vl21 = va21/u11  , scale vector
        // ------------------------------------------------------------

        {
            auto const ukk = akk;
            if(std::abs(akk) >= sfmin)
            {
                // ------------------------------------
                // perform multiplication of reciprocal
                // ------------------------------------
                auto const pivot_val = S{1.0} / ukk;
                for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
                {
                    auto const j0k = idx2D(j0, kcol, lda);

                    A[j0k] = A[j0k] * pivot_val;
                }
            }
            else
            {
                // ----------------
                // perform division
                // ----------------
                for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
                {
                    auto const j0k = idx2D(j0, kcol, lda);

                    A[j0k] = A[j0k] / ukk;
                }
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        //   (4) A22 = A22 - vl21 * vu12
        // ------------------------------------------------------------

        for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
        {
            auto const kj = idx2D(kcol, j, lda);
            auto const v_kj = A[kj];

            for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
            {
                auto const ik = idx2D(i, kcol, lda);
                auto const v_ik = A[ik];

                auto const ij = idx2D(i, j, lda);
                A[ij] = A[ij] - v_ik * v_kj;
            }
        }

        __syncthreads();

    } // end for kcol

    __syncthreads();
}

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename U>
ROCSOLVER_KERNEL void getf2_nopiv_kernel_small(const rocblas_int n,
                                               U AA,
                                               const rocblas_int shiftA,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               rocblas_int* const info)
{
    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    // --------------------------------
    // note hipGridDim_z == batch_count
    // --------------------------------
    auto const bid = hipBlockIdx_z;
    assert(AA != nullptr);

    T* const A = (AA != nullptr) ? load_ptr_batch(AA, bid, shiftA, strideA) : nullptr;

    assert(info != nullptr);
    rocblas_int* const info_bid = (info == nullptr) ? nullptr : &(info[bid]);

    assert(A != nullptr);

    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * static_cast<int64_t>(lda)); };
    auto idx_lds = [](auto i, auto j, auto lda) { return (i + j * (lda)); };

    // -----------------------------------------
    // assume n by n matrix will fit in LDS cache
    // -----------------------------------------

    auto const ld_Ash = n;

    extern __shared__ T Ash[];

    // ------------------------------------
    // copy n by n sub-matrix into shared memory
    // ------------------------------------
    __syncthreads();

    {
        for(rocblas_int j = j_start; j < n; j += j_inc)
        {
            for(rocblas_int i = i_start; i < n; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_lds = idx_lds(i, j, n);

                Ash[ij_lds] = A[ij];
            };
        };
    }

    __syncthreads();

    {
        getf2_nopiv_simple<T, rocblas_int>(n, Ash, info_bid);
    }

    __syncthreads();

    // -------------------------------------
    // copy n by n packed matrix into global memory
    // -------------------------------------

    {
        for(rocblas_int j = j_start; j < n; j += j_inc)
        {
            for(rocblas_int i = i_start; i < n; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_lds = idx_lds(i, j, n);

                A[ij] = Ash[ij_lds];
            };
        };
    }

    __syncthreads();
}

/*************************************************************
    Launchers of specilized kernels
*************************************************************/

template <typename T, typename U>
rocblas_status getf2_nopiv_run_small(rocblas_handle handle,
                                     const rocblas_int n,
                                     U A,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     rocblas_int* info,
                                     const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("getf2_nopiv_kernel_small", "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    {
        auto const lds_size = sizeof(T) * n * n;
        ROCSOLVER_LAUNCH_KERNEL((getf2_nopiv_kernel_small<T, U>), dim3(1, 1, batch_count),
                                dim3(BS2, BS2, 1), lds_size, stream, n, A, shiftA, lda, strideA,
                                info);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GETF2_NOPIV_SMALL(T, U)                                        \
    template rocblas_status getf2_nopiv_run_small<T, U>(                           \
        rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA, \
        const rocblas_int lda, const rocblas_stride strideA, rocblas_int* info,    \
        const rocblas_int batch_count)
