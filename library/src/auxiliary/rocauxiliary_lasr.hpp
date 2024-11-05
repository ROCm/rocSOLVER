/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2013
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef LASR_MAX_NTHREADS
#define LASR_MAX_NTHREADS 64
#endif

/***************** GPU Device functions *****************************************/
/********************************************************************************/

template <typename T, typename S, typename I>
__host__ __device__ static void lasr_body(const rocblas_side side,
                                          const rocblas_pivot pivot,
                                          const rocblas_direct direct,
                                          I const m,
                                          I const n,
                                          S* C_,
                                          S* S_,
                                          T* A_,
                                          I const lda,
                                          I const tid,
                                          I const t_inc)
{
    auto c = [&](auto i) -> const S { return (C_[i - 1]); };
    auto s = [&](auto i) -> const S { return (S_[i - 1]); };
    auto A = [&](auto i, auto j) -> T& { return (A_[i - 1 + (j - 1) * lda]); };

    // ---------------------
    // determine path case
    // ---------------------
    const bool is_side_Left = (side == rocblas_side_left);
    const bool is_side_Right = (side == rocblas_side_right);
    const bool is_pivot_Variable = (pivot == rocblas_pivot_variable);
    const bool is_pivot_Bottom = (pivot == rocblas_pivot_bottom);
    const bool is_pivot_Top = (pivot == rocblas_pivot_top);
    const bool is_direct_Forward = (direct == rocblas_forward_direction);
    const bool is_direct_Backward = (direct == rocblas_backward_direction);

    // ------------
    // path cases:
    // ------------

    //  -----------------------------
    //  A := P*A
    //  Variable pivot, the plane (k,k+1)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Variable && is_direct_Forward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp = A(1, j);
            for(I i = 1; i <= (m - 1); i++)
            {
                const auto ctemp = c(i);
                const auto stemp = s(i);
                const auto temp_hold = A(i + 1, j);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp = ctemp * temp_hold - stemp * temp;
            }
            A(m, j) = temp;
        }
        return;
    }

    //  -----------------------------
    //  A := P*A
    //  Variable pivot, the plane (k,k+1)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Variable && is_direct_Backward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp = A(m, j);
            for(I i = (m - 1); i >= 1; i--)
            {
                const auto ctemp = c(i);
                const auto stemp = s(i);
                const auto temp_hold = A(i, j);
                A(i + 1, j) = ctemp * temp - stemp * temp_hold;
                temp = stemp * temp + ctemp * temp_hold;
            }
            A(1, j) = temp;
        }
        return;
    }

    //  -----------------------------
    //  A := P*A
    //  Top pivot, the plane (1,k+1)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Top && is_direct_Forward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp_hold = A(1, j);
            for(I i = 2; i <= m; i++)
            {
                const auto ctemp = c(i - 1);
                const auto stemp = s(i - 1);
                const auto temp = A(i, j);
                A(i, j) = ctemp * temp - stemp * temp_hold;
                temp_hold = stemp * temp + ctemp * temp_hold;
            }
            A(1, j) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := P*A
    //  Top pivot, the plane (1,k+1)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Top && is_direct_Backward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp_hold = A(1, j);
            for(I i = m; i >= 2; i--)
            {
                const auto ctemp = c(i - 1);
                const auto stemp = s(i - 1);
                const auto temp = A(i, j);
                A(i, j) = ctemp * temp - stemp * temp_hold;
                temp_hold = stemp * temp + ctemp * temp_hold;
            }
            A(1, j) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := P*A
    //  Bottom pivot, the plane (k,z)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Bottom && is_direct_Forward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp_hold = A(m, j);
            for(I i = 1; i <= (m - 1); i++)
            {
                const auto ctemp = c(i);
                const auto stemp = s(i);
                const auto temp = A(i, j);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp_hold = ctemp * temp_hold - stemp * temp;
            }
            A(m, j) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := P*A
    //  Bottom pivot, the plane (k,z)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Left && is_pivot_Bottom && is_direct_Backward)
    {
        for(I j = 1 + tid; j <= n; j += t_inc)
        {
            auto temp_hold = A(m, j);
            for(I i = (m - 1); i >= 1; i--)
            {
                const auto ctemp = c(i);
                const auto stemp = s(i);
                const auto temp = A(i, j);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp_hold = ctemp * temp_hold - stemp * temp;
            }
            A(m, j) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Variable pivot, the plane (k,k+1)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Variable && is_direct_Forward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp = A(i, 1);
            for(I j = 1; j <= (n - 1); j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                const auto temp_hold = A(i, j + 1);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp = ctemp * temp_hold - stemp * temp;
            }
            A(i, n) = temp;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Variable pivot, the plane (k,k+1)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Variable && is_direct_Backward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp = A(i, n);
            for(I j = (n - 1); j >= 1; j--)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                const auto temp_hold = A(i, j);
                A(i, j + 1) = ctemp * temp - stemp * temp_hold;
                temp = stemp * temp + ctemp * temp_hold;
            }
            A(i, 1) = temp;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Top pivot, the plane (1,k+1)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Top && is_direct_Forward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp_hold = A(i, 1);
            for(I j = 2; j <= n; j++)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                const auto temp = A(i, j);
                A(i, j) = ctemp * temp - stemp * temp_hold;
                temp_hold = stemp * temp + ctemp * temp_hold;
            }
            A(i, 1) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Top pivot, the plane (1,k+1)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Top && is_direct_Backward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp_hold = A(i, 1);
            for(I j = n; j >= 2; j--)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                const auto temp = A(i, j);
                A(i, j) = ctemp * temp - stemp * temp_hold;
                temp_hold = stemp * temp + ctemp * temp_hold;
            }
            A(i, 1) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Bottom pivot, the plane (k,z)
    //  P = P(z-1) * ... * P(2) * P(1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Bottom && is_direct_Forward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp_hold = A(i, n);
            for(I j = 1; j <= (n - 1); j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                const auto temp = A(i, j);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp_hold = ctemp * temp_hold - stemp * temp;
            }
            A(i, n) = temp_hold;
        }
        return;
    }

    //  -----------------------------
    //  A := A*P**T
    //  Bottom pivot, the plane (k,z)
    //  P = P(1)*P(2)*...*P(z-1)
    //  -----------------------------
    if(is_side_Right && is_pivot_Bottom && is_direct_Backward)
    {
        for(I i = 1 + tid; i <= m; i += t_inc)
        {
            auto temp_hold = A(i, n);
            for(I j = (n - 1); j >= 1; j--)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                const auto temp = A(i, j);
                A(i, j) = stemp * temp_hold + ctemp * temp;
                temp_hold = ctemp * temp_hold - stemp * temp;
            }
            A(i, n) = temp_hold;
        }
        return;
    }

    return;
}

template <typename T, typename S, typename U, typename I>
__global__ static void __launch_bounds__(LASR_MAX_NTHREADS)
    lasr_kernel(const rocblas_side side,
                const rocblas_pivot pivot,
                const rocblas_direct direct,
                I const m,
                I const n,
                S* CA,
                const rocblas_stride strideC,
                S* SA,
                const rocblas_stride strideS,
                U AA,
                const rocblas_stride shiftA,
                I const lda,
                const rocblas_stride strideA)
{
    const auto nblocks = hipGridDim_x;
    const auto nthreads_per_block = hipBlockDim_x;
    const auto nthreads = nblocks * nthreads_per_block;
    I const tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const t_inc = nthreads;

    // select batch instance
    const auto bid = hipBlockIdx_z;
    T* A_ = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* C_ = CA + bid * strideC;
    S* S_ = SA + bid * strideS;

    lasr_body(side, pivot, direct, m, n, C_, S_, A_, lda, tid, t_inc);
}

/***************** GPU Device functions *****************************************/
/********************************************************************************/

template <typename SS, typename U>
rocblas_status rocsolver_lasr_argCheck(rocblas_handle handle,
                                       const rocblas_side side,
                                       const rocblas_pivot pivot,
                                       const rocblas_direct direct,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       SS* C,
                                       SS* S,
                                       U A,
                                       const rocblas_int lda)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(pivot != rocblas_pivot_variable && pivot != rocblas_pivot_top && pivot != rocblas_pivot_bottom)
        return rocblas_status_invalid_value;
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    bool is_side_left = (side == rocblas_side_left);
    bool is_side_right = (side == rocblas_side_right);
    if(m && n && !A)
        return rocblas_status_invalid_pointer;
    if(is_side_left && m > 1 && (!C || !S))
        return rocblas_status_invalid_pointer;
    if(is_side_right && n > 1 && (!C || !S))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_lasr_template(rocblas_handle handle,
                                       const rocblas_side side,
                                       const rocblas_pivot pivot,
                                       const rocblas_direct direct,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       S* CA,
                                       const rocblas_stride strideC,
                                       S* SA,
                                       const rocblas_stride strideS,
                                       U AA,
                                       const rocblas_stride shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("lasr", "side:", side, "pivot:", pivot, "direct:", direct, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    bool is_side_left = (side == rocblas_side_left);
    bool is_side_right = (side == rocblas_side_right);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;
    if((is_side_left && m < 2) || (is_side_right && n < 2))
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto const nthreads = LASR_MAX_NTHREADS;
    auto const mn = (is_side_left) ? n : m;
    auto const nblocks = (mn - 1) / nthreads + 1;

    hipLaunchKernelGGL((lasr_kernel<T>), dim3(nblocks, 1, batch_count), dim3(nthreads, 1, 1), 0,
                       stream, side, pivot, direct, m, n, CA, strideC, SA, strideS, AA, shiftA, lda,
                       strideA);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
