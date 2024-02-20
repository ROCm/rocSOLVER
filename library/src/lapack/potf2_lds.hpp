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
#include <algorithm>
#include <cmath>

__device__ static double rocsolver_conj(double x)
{
    return (x);
}
__device__ static float rocsolver_conj(float x)
{
    return (x);
}
__device__ static rocblas_double_complex rocsolver_conj(rocblas_double_complex x)
{
    return (std::conj(x));
}
__device__ static rocblas_float_complex rocsolver_conj(rocblas_float_complex x)
{
    return (std::conj(x));
}

/**
 * indexing for packed storage
 * for upper triangular
 *
 * ---------------------------
 * 0 1 3
 *   2 4
 *     5
 * ---------------------------
 *
 **/

template <typename I>
__device__ static I idx_upper(I i, I k, I n)
{
    assert((0 <= i) && (i <= (n - 1)));
    assert((0 <= k) && (k <= (n - 1)));
    bool const is_upper = (i <= k);
    assert(is_upper);
    return (i + ((k) * (k + 1)) / 2);
}

/**
 * indexing for packed storage
 * for lower triangular
 *
 * ---------------------------
 * 0
 * 1      n
 * *      (n+1)
 * *
 * (n-1)  ...        n*(n+1)/2
 * ---------------------------
 **/
template <typename I>
__device__ static I idx_lower(I i, I k, I n)
{
    assert((0 <= i) && (i <= (n - 1)));
    assert((0 <= k) && (k <= (n - 1)));
    bool const is_lower = (i >= k);
    assert(is_lower);

    return ((i - k) + (k * (2 * n + 1 - k)) / 2);
}

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block.
 * ------------------------------------------------------
**/
template <typename T, typename I>
__device__ static void potf2_simple(bool const is_upper, I const n, T* const A, I* const info)
{
    auto const lda = n;
    bool const is_lower = (!is_upper);

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

    if(is_lower)
    {
        auto idx2D = [=](I i, I k, I n) { return (idx_lower<I>(i, k, n)); };
        // ---------------------------------------------------
        // [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
        // [ vl21  L22]    [        L22' ]     [ va21, A22 ]
        //
        //
        //   assume l11 is scalar 1x1 matrix
        //
        //   (1) l11 * l11' = a11 =>  l11 = sqrt( abs(a11) ), scalar computation
        //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
        //   (3) L22 * L22' + vl21 * vl21' = A22
        //
        //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
        //   (3b) L22 * L22' = A22,   cholesky factorization, tail recursion
        // ---------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto kk = idx2D(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    *info = (*info == 0) ? kcol + 1 : (*info);
                }
                break;
            }

            auto const lkk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const inv_lkk = 1 / rocsolver_conj(lkk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const j0k = idx2D(j0, kcol, lda);
                A[j0k] *= inv_lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
            //
            //   note: update lower triangular part
            // ------------------------------------------------------------

            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx2D(j, kcol, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const lower_part = (i >= j);
                    if(lower_part)
                    {
                        auto const vi = A[idx2D(i, kcol, lda)];
                        auto const ij = idx2D(i, j, lda);
                        A[ij] = A[ij] - vi * rocsolver_conj(vj);
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }
    else
    {
        auto idx2D = [](I i, I k, I n) { return (idx_upper<I>(i, k, n)); };

        // --------------------------------------------------
        // [u11'        ] * [u11    vU12 ] = [ a11     vA12 ]
        // [vU12'   U22']   [       U22  ]   [ vA12'   A22  ]
        //
        // (1) u11' * u11 = a11 =?  u11 = sqrt( abs( a11 ) )
        // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
        //     or vU12 = vA12/u11'
        // (3) vU12' * vU12 + U22'*U22 = A22
        //
        // (3a) A22 = A22 - vU12' * vU12
        // (3b) U22' * U22 = A22,  cholesky factorization, tail recursion
        // --------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto const kk = idx2D(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                A[kk] = akk;
                if(tid == 0)
                {
                    // Fortran 1-based index
                    *info = (*info == 0) ? kcol + 1 : (*info);
                }

                break;
            }

            auto const ukk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = ukk;
            }
            __syncthreads();

            // ----------------------------------------------
            // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
            // ----------------------------------------------
            auto const inv_ukk = 1 / rocsolver_conj(ukk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const kj0 = idx2D(kcol, j0, lda);
                A[kj0] *= inv_ukk;
            }

            __syncthreads();

            // -----------------------------
            // (3a) A22 = A22 - vU12' * vU12
            //
            // note: update upper triangular part
            // -----------------------------
            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx2D(kcol, j, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const upper_part = (i <= j);
                    if(upper_part)
                    {
                        auto const vi = A[idx2D(kcol, i, lda)];
                        auto const ij = idx2D(i, j, lda);

                        A[ij] = A[ij] - rocsolver_conj(vi) * vj;
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }

    __syncthreads();
}

template <typename T, typename U>
ROCSOLVER_KERNEL void potf2_lds(const bool is_upper,
                                const rocblas_int n,
                                U AA,
                                const rocblas_int shiftA,
                                const rocblas_stride strideA,
                                const rocblas_int lda,
                                rocblas_int* const info)
{
    bool const is_lower = (!is_upper);

    using Treal = decltype(std::real(T{}));

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

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

    // -----------------------------------------
    // assume n by n matrix will fit in LDS cache
    // -----------------------------------------

    auto const ld_Ash = n;
    size_t constexpr LDS_MAXIMUM_SIZE = 64 * 1024;

    bool const use_lds = (sizeof(T) * n * (n + 1) <= LDS_MAXIMUM_SIZE * 2);
    __shared__ T Ash[LDS_MAXIMUM_SIZE / sizeof(T)];

    assert(use_lds);

    // ------------------------------------
    // copy n by n packed matrix into shared memory
    // ------------------------------------
    __syncthreads();

    if(is_lower)
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_packed = idx_lower<rocblas_int>(i, j, n);

                Ash[ij_packed] = A[ij];
            };
        };
    }
    else
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i <= j; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_packed = idx_upper<rocblas_int>(i, j, n);

                Ash[ij_packed] = A[ij];
            };
        };
    }

    __syncthreads();

    {
        potf2_simple<T, rocblas_int>(is_upper, n, Ash, info_bid);
    }

    __syncthreads();

    // -------------------------------------
    // copy n by n packed matrix into global memory
    // -------------------------------------
    if(is_lower)
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_packed = idx_lower<rocblas_int>(i, j, n);
                A[ij] = Ash[ij_packed];
            };
        };
    }
    else
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i <= j; i += i_inc)
            {
                auto const ij = idx2D(i, j, lda);
                auto const ij_packed = idx_upper<rocblas_int>(i, j, n);

                A[ij] = Ash[ij_packed];
            };
        };
    }

    __syncthreads();
}
