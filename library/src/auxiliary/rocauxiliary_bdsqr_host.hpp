
/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <assert.h>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdio.h>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef CHECK_HIP
#define CHECK_HIP(fcn)                  \
    {                                   \
        hipError_t const istat = (fcn); \
        assert(istat == hipSuccess);    \
    }
#endif

#ifndef LASR_MAX_NTHREADS
#define LASR_MAX_NTHREADS 64
#endif

template <typename S, typename T, typename I>
__global__ static void __launch_bounds__(LASR_MAX_NTHREADS) lasr_kernel(char const side,
                                                                        char const pivot,
                                                                        char const direct,
                                                                        I const m,
                                                                        I const n,
                                                                        S const* const c_,
                                                                        S const* const s_,
                                                                        T* const A_,
                                                                        I const lda)
{
    const auto nblocks = hipGridDim_x;
    const auto nthreads_per_block = hipBlockDim_x;
    const auto nthreads = nblocks * nthreads_per_block;
    const auto tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    const auto i_inc = nthreads;
    const auto ij_nb = nthreads;
    const auto ij_start = tid;

    auto max = [](auto x, auto y) { return ((x > y) ? x : y); };
    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };

    auto indx2f = [](auto i, auto j, auto lda) {
        assert((1 <= i));
        assert((1 <= lda));
        assert((1 <= j));

        // return ((i - 1) + (j - 1) * int64_t(lda));
        return (i + j * lda - (1 + lda));
    };

    auto indx1f = [](auto i) -> int64_t {
        assert((1 <= i));
        return (i - int64_t(1));
    };

    auto c = [&](auto i) -> const S& { return (c_[indx1f(i)]); };
    auto s = [&](auto i) -> const S& { return (s_[indx1f(i)]); };
    auto A = [&](auto i, auto j) -> T& { return (A_[indx2f(i, j, lda)]); };

    const S one = 1;
    const S zero = 0;

    // ----------------
    // check arguments
    // ----------------

    const bool is_side_Left = (side == 'L') || (side == 'l');
    const bool is_side_Right = (side == 'R') || (side == 'r');

    const bool is_pivot_Variable = (pivot == 'V') || (pivot == 'v');
    const bool is_pivot_Bottom = (pivot == 'B') || (pivot == 'b');
    const bool is_pivot_Top = (pivot == 'T') || (pivot == 't');

    const bool is_direct_Forward = (direct == 'F') || (direct == 'f');
    const bool is_direct_Backward = (direct == 'B') || (direct == 'b');

    {
        const bool isok_side = is_side_Left || is_side_Right;
        const bool isok_pivot = is_pivot_Variable || is_pivot_Bottom || is_pivot_Top;
        const bool isok_direct = is_direct_Forward || is_direct_Backward;

        const I info = (!isok_side) ? 1
            : (!isok_pivot)         ? 2
            : (!isok_direct)        ? 3
            : (m < 0)               ? 4
            : (n < 0)               ? 5
            : (c_ == nullptr)       ? 6
            : (s_ == nullptr)       ? 7
            : (A_ == nullptr)       ? 8
            : (lda < max(1, m))     ? 9
                                    : 0;
        if(info != 0)
            return;
    };

    {
        const bool has_work = (m >= 1) && (n >= 1);
        if(!has_work)
        {
            return;
        };
    };

    if(is_side_Left && is_pivot_Variable && is_direct_Forward)
    {
        //  -----------------------------
        //  A := P*A
        //  Variable pivot, the plane (k,k+1)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------
        {
            for(I j = 1; j <= (m - 1); j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = 1 + tid; i <= n; i += i_inc)
                    {
                        const auto temp = A(j + 1, i);
                        A(j + 1, i) = ctemp * temp - stemp * A(j, i);
                        A(j, i) = stemp * temp + ctemp * A(j, i);
                    }
                };
            };
        };

        return;
    };

    if(is_side_Left && is_pivot_Variable && is_direct_Backward)
    {
        //  -----------------------------
        //  A := P*A
        //  Variable pivot, the plane (k,k+1)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------

        auto const jend = (m - 1);
        auto const jstart = 1;
        auto const istart = 1;
        auto const iend = n;

        for(I j = jend; j >= jstart; j--)
        {
            const auto ctemp = c(j);
            const auto stemp = s(j);
            if((ctemp != one) || (stemp != zero))
            {
                for(I i = istart + tid; i <= iend; i += i_inc)
                {
                    const auto temp = A(j + 1, i);
                    A(j + 1, i) = ctemp * temp - stemp * A(j, i);
                    A(j, i) = stemp * temp + ctemp * A(j, i);
                };
            };
        };

        return;
    };

    if(is_side_Left && is_pivot_Top && is_direct_Forward)
    {
        //  -----------------------------
        //  A := P*A
        //  Top pivot, the plane (1,k+1)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------
        {
            for(I j = 2; j <= m; j++)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                for(I i = 1 + tid; i <= n; i += i_inc)
                {
                    const auto temp = A(j, i);
                    A(j, i) = ctemp * temp - stemp * A(1, i);
                    A(1, i) = stemp * temp + ctemp * A(1, i);
                };
            };
        };

        return;
    };

    if(is_side_Left && is_pivot_Top && is_direct_Backward)
    {
        //  -----------------------------
        //  A := P*A
        //  Top pivot, the plane (1,k+1)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------
        {
            auto const jend = m;
            auto const jstart = 2;
            auto const istart = 1;
            auto const iend = n;

            for(I j = jend; j >= jstart; j--)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(j, i);

                        A(j, i) = ctemp * temp - stemp * A(1, i);
                        A(1, i) = stemp * temp + ctemp * A(1, i);
                    };
                };
            };
        }

        return;
    };

    if(is_side_Left && is_pivot_Bottom && is_direct_Forward)
    {
        //  -----------------------------
        //  A := P*A
        //  Bottom pivot, the plane (k,z)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------
        {
            auto const jstart = 1;
            auto const jend = (m - 1);
            auto const istart = 1;
            auto const iend = n;

            for(I j = jstart; j <= jend; j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(j, i);
                        A(j, i) = stemp * A(m, i) + ctemp * temp;
                        A(m, i) = ctemp * A(m, i) - stemp * temp;
                    };
                };
            };
        }

        return;
    };

    if(is_side_Left && is_pivot_Bottom && is_direct_Backward)
    {
        //  -----------------------------
        //  A := P*A
        //  Bottom pivot, the plane (k,z)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------
        {
            auto const jend = (m - 1);
            auto const jstart = 1;
            auto const istart = 1;
            auto const iend = n;

            for(I j = jend; j >= jstart; j--)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(j, i);
                        A(j, i) = stemp * A(m, i) + ctemp * temp;
                        A(m, i) = ctemp * A(m, i) - stemp * temp;
                    };
                };
            };
        }

        return;
    };

    if(is_side_Right && is_pivot_Variable && is_direct_Forward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Variable pivot, the plane (k,k+1)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------

        {
            auto const jstart = 1;
            auto const jend = (n - 1);
            auto const istart = 1;
            auto const iend = m;

            for(I j = jstart; j <= jend; j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j + 1);
                        A(i, j + 1) = ctemp * temp - stemp * A(i, j);
                        A(i, j) = stemp * temp + ctemp * A(i, j);
                    };
                };
            };
        }

        return;
    };

    if(is_side_Right && is_pivot_Variable && is_direct_Backward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Variable pivot, the plane (k,k+1)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------

        {
            auto const jend = (n - 1);
            auto const jstart = 1;
            auto const istart = 1;
            auto const iend = m;

            for(I j = jend; j >= jstart; j--)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j + 1);
                        A(i, j + 1) = ctemp * temp - stemp * A(i, j);
                        A(i, j) = stemp * temp + ctemp * A(i, j);
                    };
                };
            };
        }
        return;
    };

    if(is_side_Right && is_pivot_Top && is_direct_Forward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Top pivot, the plane (1,k+1)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------

        {
            auto const jstart = 2;
            auto const jend = n;
            auto const istart = 1;
            auto const iend = m;

            for(I j = jstart; j <= jend; j++)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j);

                        A(i, j) = ctemp * temp - stemp * A(i, 1);
                        A(i, 1) = stemp * temp + ctemp * A(i, 1);
                    };
                };
            };
        }

        return;
    };

    if(is_side_Right && is_pivot_Top && is_direct_Backward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Top pivot, the plane (1,k+1)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------

        {
            auto const jend = n;
            auto const jstart = 2;
            auto const istart = 1;
            auto const iend = m;

            for(I j = jend; j >= jstart; j--)
            {
                const auto ctemp = c(j - 1);
                const auto stemp = s(j - 1);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j);

                        A(i, j) = ctemp * temp - stemp * A(i, 1);
                        A(i, 1) = stemp * temp + ctemp * A(i, 1);
                    };
                };
            };
        }

        return;
    };

    if(is_side_Right && is_pivot_Bottom && is_direct_Forward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Bottom pivot, the plane (k,z)
        //  P = P(z-1) * ... * P(2) * P(1)
        //  -----------------------------

        {
            auto const jstart = 1;
            auto const jend = (n - 1);
            auto const istart = 1;
            auto const iend = m;

            for(I j = jstart; j <= jend; j++)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j);

                        A(i, j) = stemp * A(i, n) + ctemp * temp;
                        A(i, n) = ctemp * A(i, n) - stemp * temp;
                    };
                };
            };
        }

        return;
    };

    if(is_side_Right && is_pivot_Bottom && is_direct_Backward)
    {
        //  -----------------------------
        //  A := A*P**T
        //  Bottom pivot, the plane (k,z)
        //  P = P(1)*P(2)*...*P(z-1)
        //  -----------------------------

        {
            auto const jend = (n - 1);
            auto const jstart = 1;
            auto const istart = 1;
            auto const iend = m;

            for(I j = jend; j >= jstart; j--)
            {
                const auto ctemp = c(j);
                const auto stemp = s(j);
                if((ctemp != one) || (stemp != zero))
                {
                    for(I i = istart + tid; i <= iend; i += i_inc)
                    {
                        const auto temp = A(i, j);
                        A(i, j) = stemp * A(i, n) + ctemp * temp;
                        A(i, n) = ctemp * A(i, n) - stemp * temp;
                    };
                };
            };
        }

        return;
    };

    return;
}

template <typename S, typename T, typename I>
static void lasr_template_gpu(char const side,
                              char const pivot,
                              char const direct,
                              I const m,
                              I const n,
                              S const* const c_,
                              S const* const s_,
                              T* const A_,
                              I const lda,
                              hipStream_t stream = 0)
{
    auto const nthreads = LASR_MAX_NTHREADS;

    bool const is_left_side = (side == 'L') || (side == 'l');
    auto const mn = (is_left_side) ? n : m;

    auto const nblocks = (mn - 1) / nthreads + 1;
    hipLaunchKernelGGL((lasr_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0, stream,
                       side, pivot, direct, m, n, c_, s_, A_, lda);
}

template <typename S, typename T, typename I>
__global__ static void
    rot_kernel(I const n, T* const x, I const incx, T* const y, I const incy, S const c, S const s)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    if((incx == 1) && (incy == 1))
    {
        // ------------
        // special case
        // ------------
        for(I i = i_start; i < n; i += i_inc)
        {
            auto const temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    }
    else
    {
        // ---------------------------
        // code for unequal increments
        // ---------------------------

        for(auto i = i_start; i < n; i += i_inc)
        {
            auto const ix = 0 + i * static_cast<int64_t>(incx);
            auto const iy = 0 + i * static_cast<int64_t>(incy);
            auto const temp = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = temp;
        }
    }
}

template <typename S, typename T, typename I>
static void
    rot_template(I const n, T* x, I const incx, T* y, I const incy, S const c, S const s, hipStream_t stream)
{
    auto nthreads = warpSize * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    hipLaunchKernelGGL((rot_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0, stream,
                       n, x, incx, y, incy, c, s);
}

template <typename S, typename T, typename I>
__global__ static void scal_kernel(I const n, S const da, T* const x, I const incx)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    S const zero = 0;
    bool const is_da_zero = (da == zero);
    if(incx == 1)
    {
        for(I i = i_start; i < n; i += i_inc)
        {
            x[i] = (is_da_zero) ? zero : da * x[i];
        }
    }
    else
    {
        // ---------------------------
        // code for non-unit increments
        // ---------------------------

        for(I i = i_start; i < n; i += i_inc)
        {
            auto const ix = 0 + i * static_cast<int64_t>(incx);
            x[ix] = (is_da_zero) ? zero : da * x[ix];
        }
    }
}

template <typename S, typename T, typename I>
static void scal_template(I const n, S const da, T* const x, I const incx, hipStream_t stream)
{
    auto nthreads = warpSize * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    hipLaunchKernelGGL((scal_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0, stream,
                       n, da, x, incx);
}

template <typename S, typename T, typename I>
__global__ static void swap_kernel(I const n, T* const x, I const incx, T* const y, I const incy)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    if((incx == 1) && (incy == 1))
    {
        // ------------
        // special case
        // ------------
        for(I i = i_start; i < n; i += i_inc)
        {
            auto const temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
    }
    else
    {
        // ---------------------------
        // code for unequal increments
        // ---------------------------

        for(I i = i_start; i < n; i += i_inc)
        {
            auto const ix = 0 + i * static_cast<int64_t>(incx);
            auto const iy = 0 + i * static_cast<int64_t>(incy);

            auto const temp = y[iy];
            y[iy] = x[ix];
            x[ix] = temp;
        }
    }
}

template <typename S, typename T, typename I>
static void swap_template(I const n, T* x, I const incx, T* y, I const incy, hipStream_t stream)
{
    auto nthreads = warpSize * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    hipLaunchKernelGGL((swap_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0, stream,
                       n, x, incx, y, incy);
}

extern "C" {

double dlamch_(char* cmach);
float slamch_(char* cmach);

void zswap_(int* n, std::complex<double>* zx, int* incx, std::complex<double>* zy, int* incy);

void cswap_(int* n, std::complex<float>* zx, int* incx, std::complex<float>* zy, int* incy);

void dswap_(int* n, double* zx, int* incx, double* zy, int* incy);

void sswap_(int* n, float* zx, int* incx, float* zy, int* incy);

void dlasq1_(int* n, double* D_, double* E_, double* rwork_, int* info_arg);
void slasq1_(int* n, float* D_, float* E_, float* rwork_, int* info_arg);

void zlasr_(char* side,
            char* pivot,
            char* direct,
            int* m,
            int* n,
            double* c,
            double* s,
            std::complex<double>* A,
            int* lda);
void clasr_(char* side,
            char* pivot,
            char* direct,
            int* m,
            int* n,
            float* c,
            float* s,
            std::complex<float>* A,
            int* lda);
void slasr_(char* side, char* pivot, char* direct, int* m, int* n, float* c, float* s, float* A, int* lda);
void dlasr_(char* side, char* pivot, char* direct, int* m, int* n, double* c, double* s, double* A, int* lda);

void dlasv2_(double* f,
             double* g,
             double* h,
             double* ssmin,
             double* ssmax,
             double* snr,
             double* csr,
             double* snl,
             double* csl);
void slasv2_(float* f,
             float* g,
             float* h,
             float* ssmin,
             float* ssmax,
             float* snr,
             float* csr,
             float* snl,
             float* csl);

void zdrot_(int* n,
            std::complex<double>* zx,
            int* incx,
            std::complex<double>* zy,
            int* incy,
            double* c,
            double* s);

void csrot_(int* n,
            std::complex<float>* zx,
            int* incx,
            std::complex<float>* zy,
            int* incy,
            float* c,
            float* s);

void drot_(int* n, double* dx, int* incx, double* dy, int* incy, double* c, double* s);

void srot_(int* n, float* dx, int* incx, float* dy, int* incy, float* c, float* s);

void zdscal_(int* n, double* da, std::complex<double>* zx, int* incx);
void csscal_(int* n, float* da, std::complex<float>* zx, int* incx);
void zscal_(int* n, std::complex<double>* za, std::complex<double>* zx, int* incx);
void cscal_(int* n, std::complex<float>* za, std::complex<float>* zx, int* incx);
void dscal_(int* n, double* da, double* zx, int* incx);
void sscal_(int* n, float* da, float* zx, int* incx);

void dlartg_(double* f, double* g, double* c, double* s, double* r);
void slartg_(float* f, float* g, float* c, float* s, float* r);

void zlartg_(std::complex<double>* f,
             std::complex<double>* g,
             double* c,
             std::complex<double>* s,
             std::complex<double>* r);
void clartg_(std::complex<float>* f,
             std::complex<float>* g,
             float* c,
             std::complex<float>* s,
             std::complex<float>* r);

void dlas2_(double* f, double* g, double* h, double* ssmin, double* ssmax);
void slas2_(float* f, float* g, float* h, float* ssmin, float* ssmax);
};

extern "C" {

void cbdsqr_(char* uplo,
             int* n,
             int* ncvt,
             int* nru,
             int* ncc,
             float* d,
             float* e,
             std::complex<float>* vt,
             int* ldvt,
             std::complex<float>* u,
             int* ldu,
             std::complex<float>* c,
             int* ldc,
             float* rwork,
             int* info);

void zbdsqr_(char* uplo,
             int* n,
             int* ncvt,
             int* nru,
             int* ncc,
             double* d,
             double* e,
             std::complex<double>* vt,
             int* ldvt,
             std::complex<double>* u,
             int* ldu,
             std::complex<double>* c,
             int* ldc,
             double* rwork,
             int* info);

void sbdsqr_(char* uplo,
             int* n,
             int* ncvt,
             int* nru,
             int* ncc,
             float* d,
             float* e,
             float* vt,
             int* ldvt,
             float* u,
             int* ldu,
             float* c,
             int* ldc,
             float* rwork,
             int* info);

void dbdsqr_(char* uplo,
             int* n,
             int* ncvt,
             int* nru,
             int* ncc,
             double* d,
             double* e,
             double* vt,
             int* ldvt,
             double* u,
             int* ldu,
             double* c,
             int* ldc,
             double* rwork,
             int* info);
};

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       double& d,
                       double& e,
                       std::complex<double>& vt,
                       int& ldvt,
                       std::complex<double>& u,
                       int& ldu,
                       std::complex<double>& c,
                       int& ldc,
                       double& rwork,
                       int& info)
{
    zbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, (std::complex<double>*)&vt, &ldvt,
            (std::complex<double>*)&u, &ldu, (std::complex<double>*)&c, &ldc, &rwork, &info);
}

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       double& d,
                       double& e,
                       rocblas_complex_num<double>& vt,
                       int& ldvt,
                       rocblas_complex_num<double>& u,
                       int& ldu,
                       rocblas_complex_num<double>& c,
                       int& ldc,
                       double& rwork,
                       int& info)
{
    zbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, (std::complex<double>*)&vt, &ldvt,
            (std::complex<double>*)&u, &ldu, (std::complex<double>*)&c, &ldc, &rwork, &info);
}

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       float& d,
                       float& e,
                       std::complex<float>& vt,
                       int& ldvt,
                       std::complex<float>& u,
                       int& ldu,
                       std::complex<float>& c,
                       int& ldc,
                       float& rwork,
                       int& info)
{
    cbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, (std::complex<float>*)&vt, &ldvt,
            (std::complex<float>*)&u, &ldu, (std::complex<float>*)&c, &ldc, &rwork, &info);
}

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       float& d,
                       float& e,
                       float& vt,
                       int& ldvt,
                       float& u,
                       int& ldu,
                       float& c,
                       int& ldc,
                       float& rwork,
                       int& info)
{
    sbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc, &rwork, &info);
}

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       float& d,
                       float& e,
                       rocblas_complex_num<float>& vt,
                       int& ldvt,
                       rocblas_complex_num<float>& u,
                       int& ldu,
                       rocblas_complex_num<float>& c,
                       int& ldc,
                       float& rwork,
                       int& info)
{
    cbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, (std::complex<float>*)&vt, &ldvt,
            (std::complex<float>*)&u, &ldu, (std::complex<float>*)&c, &ldc, &rwork, &info);
}

static void call_bdsqr(char& uplo,
                       int& n,
                       int& ncvt,
                       int& nru,
                       int& ncc,
                       double& d,
                       double& e,
                       double& vt,
                       int& ldvt,
                       double& u,
                       int& ldu,
                       double& c,
                       int& ldc,
                       double& rwork,
                       int& info)
{
    dbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc, &rwork, &info);
}

#ifdef USE_LAPACK
static void call_lamch(char& cmach_arg, double& eps)
{
    char cmach = cmach_arg;
    eps = dlamch_(&cmach);
}

static void call_lamch(char& cmach_arg, float& eps)
{
    char cmach = cmach_arg;
    eps = slamch_(&cmach);
}
#else

static void call_lamch(char& cmach, double& eps)
{
    eps = ((cmach == 'E') || (cmach == 'e')) ? std::numeric_limits<double>::epsilon()
        : ((cmach == 'S') || (cmach == 's')) ? std::numeric_limits<double>::min()
                                             : std::numeric_limits<double>::min();
}

static void call_lamch(char& cmach, float& eps)
{
    eps = ((cmach == 'E') || (cmach == 'e')) ? std::numeric_limits<float>::epsilon()
        : ((cmach == 'S') || (cmach == 's')) ? std::numeric_limits<float>::min()
                                             : std::numeric_limits<float>::min();
}

#endif

#ifdef USE_LAPACK
static void call_swap(int& n,
                      rocblas_complex_num<float>& zx,
                      int& incx,
                      rocblas_complex_num<float>& zy,
                      int& incy)
{
    cswap_(&n, (std::complex<float>*)&zx, &incx, (std::complex<float>*)&zy, &incy);
}

static void call_swap(int& n, std::complex<float>& zx, int& incx, std::complex<float>& zy, int& incy)
{
    cswap_(&n, &zx, &incx, &zy, &incy);
}

static void call_swap(int& n,
                      rocblas_complex_num<double>& zx,
                      int& incx,
                      rocblas_complex_num<double>& zy,
                      int& incy)
{
    zswap_(&n, (std::complex<double>*)&zx, &incx, (std::complex<double>*)&zy, &incy);
}

static void call_swap(int& n, std::complex<double>& zx, int& incx, std::complex<double>& zy, int& incy)
{
    zswap_(&n, &zx, &incx, &zy, &incy);
}

static void call_swap(int& n, float& zx, int& incx, float& zy, int& incy)
{
    sswap_(&n, &zx, &incx, &zy, &incy);
}

static void call_swap(int& n, double& zx, int& incx, double& zy, int& incy)
{
    dswap_(&n, &zx, &incx, &zy, &incy);
}
#else

template <typename T, typename I>
static void call_swap(I& n, T& x_in, I& incx, T& y_in, I& incy)
{
    T* const x = &(x_in);
    T* const y = &(y_in);
    for(I i = 0; i < n; i++)
    {
        I const ix = i * incx;
        I const iy = i * incy;

        T const temp = x[ix];
        x[ix] = y[iy];
        y[iy] = temp;
    }
}

#endif

#ifdef USE_LAPACK
static void call_las2(double& f, double& g, double& h, double& ssmin, double& ssmax)
{
    dlas2_(&f, &g, &h, &ssmin, &ssmax);
}

static void call_las2(float& f, float& g, float& h, float& ssmin, float& ssmax)
{
    slas2_(&f, &g, &h, &ssmin, &ssmax);
}
#else

template <typename T>
static void call_las2(T& f, T& g, T& h, T& ssmin, T& ssmax)
{
    T const zero = 0;
    T const one = 1;
    T const two = 2;

    T as, at, au, c, fa, fhmn, fhmx, ga, ha;

    auto abs = [](auto x) { return (std::abs(x)); };
    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };
    auto max = [](auto x, auto y) { return ((x > y) ? x : y); };
    auto sqrt = [](auto x) { return (std::sqrt(x)); };
    auto square = [](auto x) { return (x * x); };

    fa = abs(f);
    ga = abs(g);
    ha = abs(h);
    fhmn = min(fa, ha);
    fhmx = max(fa, ha);
    if(fhmn == zero)
    {
        ssmin = zero;
        if(fhmx == zero)
        {
            ssmax = ga;
        }
        else
        {
            // ssmax = max( fhmx, ga )*sqrt( one+ ( min( fhmx, ga ) / max( fhmx, ga ) )**2 );
            ssmax = max(fhmx, ga) * sqrt(one + square(min(fhmx, ga) / max(fhmx, ga)));
        }
    }
    else
    {
        if(ga < fhmx)
        {
            as = one + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = square(ga / fhmx);
            c = two / (sqrt(as * as + au) + sqrt(at * at + au));
            ssmin = fhmn * c;
            ssmax = fhmx / c;
        }
        else
        {
            au = fhmx / ga;
            if(au == zero)
            {
                //
                //               avoid possible harmful underflow if exponent range
                //               asymmetric (true ssmin may not underflow even if
                //               au underflows)
                //
                ssmin = (fhmn * fhmx) / ga;
                ssmax = ga;
            }
            else
            {
                as = one + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                // c = one / ( sqrt( one+( as*au )**2 )+ sqrt( one+( at*au )**2 ) );
                c = one / (sqrt(one + square(as * au)) + sqrt(one + square(at * au)));
                ssmin = (fhmn * c) * au;
                ssmin = ssmin + ssmin;
                ssmax = ga / (c + c);
            }
        }
    }
}

#endif

static void call_lartg(double& f, double& g, double& c, double& s, double& r)
{
    dlartg_(&f, &g, &c, &s, &r);
}

static void call_lartg(float& f, float& g, float& c, float& s, float& r)
{
    slartg_(&f, &g, &c, &s, &r);
}

static void call_lartg(std::complex<float>& f,
                       std::complex<float>& g,
                       float& c,
                       std::complex<float>& s,
                       std::complex<float>& r)
{
    clartg_(&f, &g, &c, &s, &r);
}

static void call_lartg(std::complex<double>& f,
                       std::complex<double>& g,
                       double& c,
                       std::complex<double>& s,
                       std::complex<double>& r)
{
    zlartg_(&f, &g, &c, &s, &r);
}

#ifdef USE_LAPACK
static void call_scal(int& n, rocblas_complex_num<float>& da, rocblas_complex_num<float>& zx, int& incx)
{
    cscal_(&n, (std::complex<float>*)&da, (std::complex<float>*)&zx, &incx);
}

static void call_scal(int& n, std::complex<float>& da, std::complex<float>& zx, int& incx)
{
    cscal_(&n, &da, &zx, &incx);
}

static void
    call_scal(int& n, rocblas_complex_num<double>& da, rocblas_complex_num<double>& zx, int& incx)
{
    zscal_(&n, (std::complex<double>*)&da, (std::complex<double>*)&zx, &incx);
}

static void call_scal(int& n, std::complex<double>& da, std::complex<double>& zx, int& incx)
{
    zscal_(&n, &da, &zx, &incx);
}

static void call_scal(int& n, double& da, rocblas_complex_num<double>& zx, int& incx)
{
    zdscal_(&n, &da, (std::complex<double>*)&zx, &incx);
}

static void call_scal(int& n, double& da, std::complex<double>& zx, int& incx)
{
    zdscal_(&n, &da, &zx, &incx);
}

static void call_scal(int& n, float& da, rocblas_complex_num<float>& zx, int& incx)
{
    csscal_(&n, &da, (std::complex<float>*)&zx, &incx);
}

static void call_scal(int& n, float& da, std::complex<float>& zx, int& incx)
{
    csscal_(&n, &da, &zx, &incx);
}

static void call_scal(int& n, double& da, double& zx, int& incx)
{
    dscal_(&n, &da, &zx, &incx);
}

static void call_scal(int& n, float& da, float& zx, int& incx)
{
    sscal_(&n, &da, &zx, &incx);
}
#else
template <typename T, typename S, typename I>
static void call_scal(I& n, S& a, T& x_in, I& incx)
{
    bool const is_zero = (a == 0);
    T* const x = &x_in;
    for(I i = 0; i < n; i++)
    {
        auto const ip = i * incx;
        if(is_zero)
        {
            x[ip] = 0;
        }
        else
        {
            x[ip] *= a;
        }
    };
}

#endif

#ifdef USE_LAPACK
static void call_rot(int& n,
                     std::complex<float>& zx,
                     int& incx,
                     std::complex<float>& zy,
                     int& incy,
                     float& c,
                     float& s)
{
    csrot_(&n, &zx, &incx, &zy, &incy, &c, &s);
}

static void call_rot(int& n,
                     rocblas_complex_num<float>& zx,
                     int& incx,
                     rocblas_complex_num<float>& zy,
                     int& incy,
                     float& c,
                     float& s)
{
    csrot_(&n, (std::complex<float>*)&zx, &incx, (std::complex<float>*)&zy, &incy, &c, &s);
}

static void call_rot(int& n,
                     std::complex<double>& zx,
                     int& incx,
                     std::complex<double>& zy,
                     int& incy,
                     double& c,
                     double& s)
{
    zdrot_(&n, &zx, &incx, &zy, &incy, &c, &s);
}

static void call_rot(int& n,
                     rocblas_complex_num<double>& zx,
                     int& incx,
                     rocblas_complex_num<double>& zy,
                     int& incy,
                     double& c,
                     double& s)
{
    zdrot_(&n, (std::complex<double>*)&zx, &incx, (std::complex<double>*)&zy, &incy, &c, &s);
}

static void call_rot(int& n, double& dx, int& incx, double& dy, int& incy, double& c, double& s)
{
    drot_(&n, &dx, &incx, &dy, &incy, &c, &s);
}

static void call_rot(int& n, float& dx, int& incx, float& dy, int& incy, float& c, float& s)
{
    srot_(&n, &dx, &incx, &dy, &incy, &c, &s);
}
#else

template <typename T, typename S, typename I>
static void call_rot(I& n, T& x_in, I& incx, T& y_in, I& incy, S& c, S& s)
{
    T* const x = &(x_in);
    T* const y = &(y_in);

    for(I i = 0; i < n; i++)
    {
        auto const ix = i * incx;
        auto const iy = i * incy;

        auto const temp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = temp;
    }
}

#endif

#ifdef USE_LAPACK
static void call_lasv2(double& f,
                       double& g,
                       double& h,
                       double& ssmin,
                       double& ssmax,
                       double& snr,
                       double& csr,
                       double& snl,
                       double& csl)
{
    dlasv2_(&f, &g, &h, &ssmin, &ssmax, &snr, &csr, &snl, &csl);
}

static void call_lasv2(float& f,
                       float& g,
                       float& h,
                       float& ssmin,
                       float& ssmax,
                       float& snr,
                       float& csr,
                       float& snl,
                       float& csl)
{
    slasv2_(&f, &g, &h, &ssmin, &ssmax, &snr, &csr, &snl, &csl);
}
#else
// --------------------------------------------------------
// lasv2 computes the singular value decomposition of a 2 x 2
// triangular matrix
// [ F G ]
// [ 0 H ]
//
// on return,
// abs(ssmax) is the larger singular value,
// abs(ssmin) is the smaller singular value,
// (csl,snl) and (csr,snr) are the left and right
// singular vectors for abs(ssmax)
//
// [ csl  snl]  [  F  G ]  [ csr   -snr] = [ ssmax   0    ]
// [-snl  csl]  [  0  H ]  [ snr    csr]   [  0     ssmin ]
// --------------------------------------------------------
template <typename T>
static void call_lasv2(T& f, T& g, T& h, T& ssmin, T& ssmax, T& snr, T& csr, T& snl, T& csl)
{
    T const zero = 0;
    T const one = 1;
    T const two = 2;
    T const four = 4;
    T const half = one / two;

    bool gasmal;
    bool swap;
    int pmax;
    char cmach;

    T a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m;
    T mm, r, s, slt, srt, t, temp, tsign, tt;
    T macheps;

    auto abs = [](auto x) { return (std::abs(x)); };
    auto sqrt = [](auto x) { return (std::sqrt(x)); };
    auto sign = [](auto a, auto b) {
        auto const abs_a = std::abs(a);
        return ((b >= 0) ? abs_a : -abs_a);
    };

    ft = f;
    fa = abs(ft);
    ht = h;
    ha = abs(h);
    //
    //     pmax points to the maximum absolute element of matrix
    //       pmax = 1 if f largest in absolute values
    //       pmax = 2 if g largest in absolute values
    //       pmax = 3 if h largest in absolute values
    //
    pmax = 1;
    swap = (ha > fa);
    if(swap)
    {
        pmax = 3;
        temp = ft;
        ft = ht;
        ht = temp;
        temp = fa;
        fa = ha;
        ha = temp;
        //
        //        now fa >= ha
        //
    }
    gt = g;
    ga = abs(gt);
    if(ga == zero)
    {
        //
        //        diagonal matrix
        //
        ssmin = ha;
        ssmax = fa;
        clt = one;
        crt = one;
        slt = zero;
        srt = zero;
    }
    else
    {
        gasmal = true;
        if(ga > fa)
        {
            pmax = 2;

            cmach = 'E';
            call_lamch(cmach, macheps);

            if((fa / ga) < macheps)
            {
                //
                //              case of very large ga
                //
                gasmal = false;
                ssmax = ga;
                if(ha > one)
                {
                    ssmin = fa / (ga / ha);
                }
                else
                {
                    ssmin = (fa / ga) * ha;
                }
                clt = one;
                slt = ht / gt;
                srt = one;
                crt = ft / gt;
            }
        }
        if(gasmal)
        {
            //
            //           normal case
            //
            d = fa - ha;
            if(d == fa)
            {
                //
                //              copes with infinite f or h
                //
                l = one;
            }
            else
            {
                l = d / fa;
            }
            //
            //           note that 0  <=  l <= 1
            //
            m = gt / ft;
            //
            //           note that abs(m)  <=  1/macheps
            //
            t = two - l;
            //
            //           note that t >= 1
            //
            mm = m * m;
            tt = t * t;
            s = sqrt(tt + mm);
            //
            //           note that 1  <=  s <= 1 + 1/macheps
            //
            if(l == zero)
            {
                r = abs(m);
            }
            else
            {
                r = sqrt(l * l + mm);
            }
            //
            //           note that 0  <=  r .le. 1 + 1/macheps
            //
            a = half * (s + r);
            //
            //           note that 1  <=  a .le. 1 + abs(m)
            //
            ssmin = ha / a;
            ssmax = fa * a;
            if(mm == zero)
            {
                //
                //              note that m is very tiny
                //
                if(l == zero)
                {
                    t = sign(two, ft) * sign(one, gt);
                }
                else
                {
                    t = gt / sign(d, ft) + m / t;
                }
            }
            else
            {
                t = (m / (s + t) + m / (r + l)) * (one + a);
            }
            l = sqrt(t * t + four);
            crt = two / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if(swap)
    {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    }
    else
    {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }
    //
    //     correct signs of ssmax and ssmin
    //
    if(pmax == 1)
    {
        tsign = sign(one, csr) * sign(one, csl) * sign(one, f);
    }
    if(pmax == 2)
    {
        tsign = sign(one, snr) * sign(one, csl) * sign(one, g);
    }
    if(pmax == 3)
    {
        tsign = sign(one, snr) * sign(one, snl) * sign(one, h);
    }
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(one, f) * sign(one, h));
}

#endif

static void call_lasq1(int& n, double& D_, double& E_, double& rwork_, int& info_arg)
{
    dlasq1_(&n, &D_, &E_, &rwork_, &info_arg);
};

static void call_lasq1(int& n, float& D_, float& E_, float& rwork_, int& info_arg)
{
    slasq1_(&n, &D_, &E_, &rwork_, &info_arg);
};

static void call_lasr(char& side,
                      char& pivot,
                      char& direct,
                      int& m,
                      int& n,
                      float& c,
                      float& s,
                      rocblas_complex_num<float>& A,
                      int& lda)
{
    clasr_(&side, &pivot, &direct, &m, &n, &c, &s, (std::complex<float>*)&A, &lda);
};

static void call_lasr(char& side,
                      char& pivot,
                      char& direct,
                      int& m,
                      int& n,
                      double& c,
                      double& s,
                      rocblas_complex_num<double>& A,
                      int& lda)
{
    zlasr_(&side, &pivot, &direct, &m, &n, &c, &s, (std::complex<double>*)&A, &lda);
};

static void call_lasr(char& side,
                      char& pivot,
                      char& direct,
                      int& m,
                      int& n,
                      double& c,
                      double& s,
                      std::complex<double>& A,
                      int& lda)
{
    zlasr_(&side, &pivot, &direct, &m, &n, &c, &s, &A, &lda);
};

static void call_lasr(char& side,
                      char& pivot,
                      char& direct,
                      int& m,
                      int& n,
                      float& c,
                      float& s,
                      std::complex<float>& A,
                      int& lda)
{
    clasr_(&side, &pivot, &direct, &m, &n, &c, &s, &A, &lda);
};

static void
    call_lasr(char& side, char& pivot, char& direct, int& m, int& n, float& c, float& s, float& A, int& lda)
{
    slasr_(&side, &pivot, &direct, &m, &n, &c, &s, &A, &lda);
};

static void call_lasr(char& side,
                      char& pivot,
                      char& direct,
                      int& m,
                      int& n,
                      double& c,
                      double& s,
                      double& A,
                      int& lda)
{
    dlasr_(&side, &pivot, &direct, &m, &n, &c, &s, &A, &lda);
};

template <typename S, typename T, typename I>
static void bdsqr_single_template(char uplo,
                                  I n,
                                  I ncvt,
                                  I nru,
                                  I ncc,

                                  S* d_,
                                  S* e_,

                                  T* vt_,
                                  I ldvt,
                                  T* u_,
                                  I ldu,
                                  T* c_,
                                  I ldc,

                                  S* work_,
                                  I& info,
                                  S* dwork_ = nullptr,
                                  hipStream_t stream = 0)
{
    bool const use_gpu = (dwork_ != nullptr);

    // -----------------------------------
    // Lapack code used O(n^2) algorithm for sorting
    // Consider turning off this and rely on
    // bdsqr_sort() to perform sorting
    // -----------------------------------
    bool constexpr need_sort = false;

    // ---------------------------------------------------
    // NOTE: lasq1 may return non-zero info value that
    // has a different meaning
    // Consider turning off lasq1 to have consistent info value
    // ---------------------------------------------------
    bool constexpr use_lasq1 = false;

    S const zero = 0;
    S const one = 1;
    S negone = -1;
    S const hndrd = 100;
    S const hndrth = one / hndrd;
    S const ten = 10;
    S const eight = 8;
    S const meight = -one / eight;
    I const maxitr = 6;
    I ione = 1;

    bool const lower = (uplo == 'L') || (uplo == 'l');
    bool const upper = (uplo == 'U') || (uplo == 'u');
    /*
   *     rotate is true if any singular vectors desired, false otherwise
   */
    bool const rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);

    I i = 0, idir = 0, isub = 0, iter = 0, iterdivn = 0, j = 0, ll = 0, lll = 0, m = 0,
      maxitdivn = 0, nm1 = 0, nm12 = 0, nm13 = 0, oldll = 0, oldm = 0;

    I const nrc = n; // number of rows in C matrix
    I const nrvt = n; // number of rows in VT matrix
    I const ncu = n; // number of columns in U matrix

    S abse = 0, abss = 0, cosl = 0, cosr = 0, cs = 0, eps = 0, f = 0, g = 0, h = 0, mu = 0,
      oldcs = 0, oldsn = 0, r = 0, shift = 0, sigmn = 0, sigmx = 0, sinl = 0, sinr = 0, sll = 0,
      smax = 0, smin = 0, sminl = 0, sminoa = 0, sn = 0, thresh = 0, tol = 0, tolmul = 0, unfl = 0;

    /*     ..
  *     .. external functions ..
        logical            lsame
        double precision   dlamch
        external           lsame, dlamch
  *     ..
  *     .. external subroutines ..
        external           dlartg, dlas2, dlasq1, dlasr, dlasv2, drot,
       $                   dscal, dswap, xerbla
  *     ..
  *     .. intrinsic functions ..
        intrinsic          abs, dble, max, min, sign, sqrt
   */

    auto call_swap_gpu = [=](I n, T& x, I incx, T& y, I incy) {
        swap_template<S, T, I>(n, &x, incx, &y, incy, stream);
    };

    auto call_rot_gpu = [=](I n, T& x, I incx, T& y, I incy, S cosl, S sinl) {
        rot_template<S, T, I>(n, &x, incx, &y, incy, cosl, sinl, stream);
    };

    auto call_scal_gpu
        = [=](I n, auto da, T& x, I incx) { scal_template<S, T, I>(n, da, &x, incx, stream); };

    auto call_lasr_gpu_nocopy = [=](char const side, char const pivot, char const direct, I const m,
                                    I const n, S& dc, S& ds, T& A, I const lda, hipStream_t stream) {
        bool const is_left_side = (side == 'L') || (side == 'l');
        auto const mn = (is_left_side) ? m : n;
        auto const mn_m1 = (mn - 1);

        lasr_template_gpu(side, pivot, direct, m, n, &dc, &ds, &A, lda, stream);
    };

    auto call_lasr_gpu
        = [=](char const side, char const pivot, char const direct, I const m, I const n, S& c,
              S& s, T& A, I const lda, S* const dwork_, hipStream_t stream) {
              bool const is_left_side = (side == 'L') || (side == 'l');
              auto const mn = (is_left_side) ? m : n;
              auto const mn_m1 = (mn - 1);
              S* const dc = dwork_;
              S* const ds = dwork_ + mn_m1;
              CHECK_HIP(hipStreamSynchronize(stream));

              CHECK_HIP(hipMemcpyAsync(dc, &c, sizeof(S) * mn_m1, hipMemcpyHostToDevice, stream));
              CHECK_HIP(hipMemcpyAsync(ds, &s, sizeof(S) * mn_m1, hipMemcpyHostToDevice, stream));

              lasr_template_gpu(side, pivot, direct, m, n, dc, ds, &A, lda, stream);
              CHECK_HIP(hipStreamSynchronize(stream));
          };

    auto abs = [](auto x) { return (std::abs(x)); };

    auto indx2f = [](auto i, auto j, auto ld) -> int64_t {
        assert((1 <= i) && (i <= ld));
        assert((1 <= j));
        return ((i - 1) + (j - 1) * int64_t(ld));
    };

    auto d = [=](auto i) -> S& {
        assert((1 <= i) && (i <= n));
        return (d_[i - 1]);
    };

    auto e = [=](auto i) -> S& {
        assert((1 <= i) && (i <= (n - 1)));
        return (e_[i - 1]);
    };
    auto work = [=](auto i) -> S& { return (work_[i - 1]); };
    auto dwork = [=](auto i) -> S& { return (dwork_[i - 1]); };

    auto c = [=](auto i, auto j) -> T& {
        assert((1 <= i) && (i <= nrc) && (nrc <= ldc));
        assert((1 <= j) && (j <= ncc));
        return (c_[indx2f(i, j, ldc)]);
    };

    auto u = [=](auto i, auto j) -> T& {
        assert((1 <= i) && (i <= nru) && (nru <= ldu));
        assert((1 <= j) && (j <= ncu));
        return (u_[indx2f(i, j, ldu)]);
    };

    auto vt = [=](auto i, auto j) -> T& {
        assert((1 <= i) && (i <= nrvt) && (nrvt <= ldvt));
        assert((1 <= j) && (j <= ncvt));
        return (vt_[indx2f(i, j, ldvt)]);
    };

    // ---------------------------
    // emulate Fortran  intrinsics
    // ---------------------------
    auto sign = [](auto a, auto b) {
        auto const abs_a = std::abs(a);
        return ((b >= 0) ? abs_a : -abs_a);
    };

    auto dble = [](auto x) { return (static_cast<double>(x)); };

    auto max = [](auto a, auto b) { return ((a > b) ? a : b); };

    auto min = [](auto a, auto b) { return ((a < b) ? a : b); };

    auto sqrt = [](auto x) { return (std::sqrt(x)); };

    /*     ..
   *     .. executable statements ..
   *
   *     test the input parameters.
   *
   */

    info = (!upper) && (!lower) ? -1
        : (n < 0)               ? -2
        : (ncvt < 0)            ? -3
        : (nru < 0)             ? -4
        : (ncc < 0)             ? -5
        : ((ncvt == 0) && (ldvt < 1)) || ((ncvt > 0) && (ldvt < max(1, n)) ? -9 : (ldu < max(1, nru)))
        ? -11
        : ((ncc == 0) && (ldc < 1)) || ((ncc > 0) && (ldc < max(1, n))) ? -13
                                                                        : 0;

    if(info != 0)
        return;

    if(n == 0)
        return;

    bool const need_update_singular_vectors = (nru > 0) || (ncc > 0);
    bool constexpr use_lasr_gpu_nocopy = false;

    if(n == 1)
        goto L160;
    /*
   *     if no singular vectors desired, use qd algorithm
   */
    if((!rotate) && (use_lasq1))
    {
        call_lasq1(n, d(1), e(1), work(1), info);
        /*
     *     if info equals 2, dqds didn't finish, try to finish
     */
        if(info != 2)
            return;
        info = 0;
    }

    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;
    /*
   *     get machine constants
   *
   */
    {
        char cmach_eps = 'E';
        char cmach_unfl = 'S';
        call_lamch(cmach_eps, eps);
        call_lamch(cmach_unfl, unfl);
    }
    /*
   *     if matrix lower bidiagonal, rotate to be upper bidiagonal
   *     by applying givens rotations on the left
   */

    if(lower)
    {
        // do 10 i = 1, n - 1
        for(i = 1; i <= (n - 1); i++)
        {
            call_lartg(d(i), e(i), cs, sn, r);
            d(i) = r;
            e(i) = sn * d(i + 1);
            d(i + 1) = cs * d(i + 1);
            work(i) = cs;
            work(nm1 + i) = sn;
        }
    L10:

        //        ----------------------------------
        //        update singular vectors if desired
        //        ----------------------------------

        if(use_lasr_gpu_nocopy)
        {
            CHECK_HIP(hipStreamSynchronize(stream));

            if(need_update_singular_vectors)
            {
                // --------------
                // copy rotations
                // --------------
                size_t const nbytes = sizeof(S) * (n - 1);
                hipMemcpyKind const kind = hipMemcpyHostToDevice;

                {
                    void* const src = (void*)&(work(1));
                    void* const dst = (void*)&(dwork(1));
                    CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                }

                {
                    void* const src = (void*)&(work(n));
                    void* const dst = (void*)&(dwork(n));
                    CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                }
            }
            CHECK_HIP(hipStreamSynchronize(stream));
        }

        if(nru > 0)
        {
            // call_lasr( 'r', 'v', 'f', nru, n, work( 1 ), work( n ), u, ldu );
            char side = 'R';
            char pivot = 'V';
            char direct = 'F';
            if(use_gpu)
            {
                if(use_lasr_gpu_nocopy)
                {
                    call_lasr_gpu_nocopy(side, pivot, direct, nru, n, dwork(1), dwork(n), u(1, 1),
                                         ldu, stream);
                }
                else
                {
                    call_lasr_gpu(side, pivot, direct, nru, n, work(1), work(n), u(1, 1), ldu,
                                  dwork_, stream);
                }
            }
            else
            {
                call_lasr(side, pivot, direct, nru, n, work(1), work(n), u(1, 1), ldu);
            }
        }
        if(ncc > 0)
        {
            // call_lasr( 'l', 'v', 'f', n, ncc, work( 1 ), work( n ), c, ldc );
            char side = 'L';
            char pivot = 'V';
            char direct = 'F';
            if(use_gpu)
            {
                if(use_lasr_gpu_nocopy)
                {
                    call_lasr_gpu_nocopy(side, pivot, direct, n, ncc, dwork(1), dwork(n), c(1, 1),
                                         ldc, stream);
                }
                else
                {
                    call_lasr_gpu(side, pivot, direct, n, ncc, work(1), work(n), c(1, 1), ldc,
                                  dwork_, stream);
                }
            }
            else
            {
                call_lasr(side, pivot, direct, n, ncc, work(1), work(n), c(1, 1), ldc);
            }
        }
    }
    /*
   *     compute singular values to relative accuracy tol
   *     (by setting tol to be negative, algorithm will compute
   *     singular values to absolute accuracy abs(tol)*norm(input matrix))
   */

    tolmul = max(ten, min(hndrd, pow(eps, meight)));
    tol = tolmul * eps;

    /*
   *     compute approximate maximum, minimum singular values
   */

    /*
        smax = zero
        do 20 i = 1, n
           smax = max( smax, abs( d( i ) ) )
     L20:
        do 30 i = 1, n - 1
           smax = max( smax, abs( e( i ) ) )
     L30:
  */
    smax = zero;
    // do 20 i = 1, n
    for(i = 1; i <= n; i++)
    {
        smax = max(smax, abs(d(i)));
    }
L20:
    // do 30 i = 1, n - 1
    for(i = 1; i <= (n - 1); i++)
    {
        smax = max(smax, abs(e(i)));
    }
L30:

    sminl = zero;
    if(tol >= zero)
    {
        /*
     *        relative accuracy desired
     */

        sminoa = abs(d(1));
        if(sminoa == zero)
            goto L50;
        mu = sminoa;
        // do 40 i = 2, n
        for(i = 2; i <= n; i++)
        {
            mu = abs(d(i)) * (mu / (mu + abs(e(i - 1))));
            sminoa = min(sminoa, mu);
            if(sminoa == zero)
                goto L50;
        }
    L40:
    L50:

        sminoa = sminoa / sqrt(dble(n));
        thresh = max(tol * sminoa, ((unfl * n) * n) * maxitr);
    }
    else
    {
        /*
     *        absolute accuracy desired
     */

        thresh = max(abs(tol) * smax, ((unfl * n) * n) * maxitr);
    }
    /*
   *     prepare for main iteration loop for the singular values
   *     (maxit is the maximum number of passes through the inner
   *     loop permitted before nonconvergence signalled.)
   */
    maxitdivn = maxitr * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;
    /*
   *     m points to last element of unconverged part of matrix
   */
    m = n;
    /*
   *     begin main iteration loop
   */
L60:
    /*
   *     check for convergence or exceeding iteration count
   */
    if(m <= 1)
        goto L160;

    if(iter >= n)
    {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if(iterdivn >= maxitdivn)
            goto L200;
    }
    /*
   *     find diagonal block of matrix to work on
   */
    if(tol < zero && abs(d(m)) <= thresh)
        d(m) = zero;

    smax = abs(d(m));
    smin = smax;
    // do 70 lll = 1, m - 1
    for(lll = 1; lll <= (m - 1); lll++)
    {
        ll = m - lll;
        abss = abs(d(ll));
        abse = abs(e(ll));
        if(tol < zero && abss <= thresh)
            d(ll) = zero;
        if(abse <= thresh)
            goto L80;
        smin = min(smin, abss);
        smax = max(smax, max(abss, abse));
    }
L70:
    ll = 0;
    goto L90;
L80:
    e(ll) = zero;
    /*
   *     matrix splits since e(ll) = 0
   */
    if(ll == m - 1)
    {
        /*
     *        convergence of bottom singular value, return to top of loop
     */
        m = m - 1;
        goto L60;
    }
L90:
    ll = ll + 1;
    /*
   *     e(ll) through e(m-1) are nonzero, e(ll-1) is zero
   */
    if(ll == m - 1)
    {
        /*
     *        2 by 2 block, handle separately
     */
        call_lasv2(d(m - 1), e(m - 1), d(m), sigmn, sigmx, sinr, cosr, sinl, cosl);
        d(m - 1) = sigmx;
        e(m - 1) = zero;
        d(m) = sigmn;
        /*
     *        compute singular vectors, if desired
     */
        if(ncvt > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(ncvt, vt(m - 1, 1), ldvt, vt(m, 1), ldvt, cosr, sinr);
            }
            else
            {
                call_rot(ncvt, vt(m - 1, 1), ldvt, vt(m, 1), ldvt, cosr, sinr);
            }
        }
        if(nru > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(nru, u(1, m - 1), ione, u(1, m), ione, cosl, sinl);
            }
            else
            {
                call_rot(nru, u(1, m - 1), ione, u(1, m), ione, cosl, sinl);
            }
        }
        if(ncc > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(ncc, c(m - 1, 1), ldc, c(m, 1), ldc, cosl, sinl);
            }
            else
            {
                call_rot(ncc, c(m - 1, 1), ldc, c(m, 1), ldc, cosl, sinl);
            }
        }
        m = m - 2;
        goto L60;
    }
    /*
   *     if working on new submatrix, choose shift direction
   *     (from larger end diagonal element towards smaller)
   */
    if(ll > oldm || m < oldll)
    {
        if(abs(d(ll)) >= abs(d(m)))
        {
            /*
       *           chase bulge from top (big end) to bottom (small end)
       */
            idir = 1;
        }
        else
        {
            /*
       *           chase bulge from bottom (big end) to top (small end)
       */
            idir = 2;
        }
    }
    /*
   *     apply convergence tests
   */
    if(idir == 1)
    {
        /*
     *        run convergence test in forward direction
     *        first apply standard test to bottom of matrix
     */
        if(abs(e(m - 1)) <= abs(tol) * abs(d(m)) || (tol < zero && abs(e(m - 1)) <= thresh))
        {
            e(m - 1) = zero;
            goto L60;
        }

        if(tol >= zero)
        {
            /*
       *           if relative accuracy desired,
       *           apply convergence criterion forward
       */
            mu = abs(d(ll));
            sminl = mu;
            // do 100 lll = ll, m - 1
            for(lll = ll; lll <= (m - 1); lll++)
            {
                if(abs(e(lll)) <= tol * mu)
                {
                    e(lll) = zero;
                    goto L60;
                }
                mu = abs(d(lll + 1)) * (mu / (mu + abs(e(lll))));
                sminl = min(sminl, mu);
            }
            // L100:
        }
    }
    else
    {
        /*
     *        run convergence test in backward direction
     *        first apply standard test to top of matrix
     */
        if(abs(e(ll)) <= abs(tol) * abs(d(ll)) || (tol < zero && abs(e(ll)) <= thresh))
        {
            e(ll) = zero;
            goto L60;
        }

        if(tol >= zero)
        {
            /*
       *           if relative accuracy desired,
       *           apply convergence criterion backward
       */
            mu = abs(d(m));
            sminl = mu;
            // do 110 lll = m - 1, ll, -1
            for(lll = (m - 1); lll >= ll; lll--)
            {
                if(abs(e(lll)) <= tol * mu)
                {
                    e(lll) = zero;
                    goto L60;
                }
                mu = abs(d(lll)) * (mu / (mu + abs(e(lll))));
                sminl = min(sminl, mu);
            }
            // L110:
        }
    }
    oldll = ll;
    oldm = m;
    /*
   *     compute shift.  first, test if shifting would ruin relative
   *     accuracy, and if so set the shift to zero.
   */
    if(tol >= zero && n * tol * (sminl / smax) <= max(eps, hndrth * tol))
    {
        /*
     *        use a zero shift to avoid loss of relative accuracy
     */
        shift = zero;
    }
    else
    {
        /*
     *        compute the shift from 2-by-2 block at end of matrix
     */
        if(idir == 1)
        {
            sll = abs(d(ll));
            call_las2(d(m - 1), e(m - 1), d(m), shift, r);
        }
        else
        {
            sll = abs(d(m));
            call_las2(d(ll), e(ll), d(ll + 1), shift, r);
        }
        /*
     *        test if shift negligible, and if so set to zero
     */
        if(sll > zero)
        {
            if((shift / sll) * (shift / sll) < eps)
                shift = zero;
        }
    }
    /*
   *     increment iteration count
   */
    iter = iter + m - ll;
    /*
   *     if shift = 0, do simplified qr iteration
   */
    if(shift == zero)
    {
        if(idir == 1)
        {
            /*
       *           chase bulge from top to bottom
       *           save cosines and sines for later singular vector updates
       */
            cs = one;
            oldcs = one;
            // do 120 i = ll, m - 1
            for(i = ll; i <= (m - 1); i++)
            {
                auto di_cs = d(i) * cs;
                call_lartg(di_cs, e(i), cs, sn, r);
                if(i > ll)
                    e(i - 1) = oldsn * r;
                auto oldcs_r = oldcs * r;
                auto dip1_sn = d(i + 1) * sn;
                call_lartg(oldcs_r, dip1_sn, oldcs, oldsn, d(i));
                work(i - ll + 1) = cs;
                work(i - ll + 1 + nm1) = sn;
                work(i - ll + 1 + nm12) = oldcs;
                work(i - ll + 1 + nm13) = oldsn;
            }
        L120:
            h = d(m) * cs;
            d(m) = h * oldcs;
            e(m - 1) = h * oldsn;
            //
            //   -----------------------
            //   update singular vectors
            //   -----------------------

            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // --------------
                    // copy rotations
                    // --------------
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncvt, work( 1 ), work( n ), vt(
                // ll, 1 ), ldvt )
                char side = 'L';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(1), dwork(n),
                                             vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1),
                                      ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'f', nru, m-ll+1, work( nm12+1 ), work( nm13+1
                // ), u( 1, ll ), ldu )
                char side = 'R';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(nm12 + 1),
                                             dwork(nm13 + 1), u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                                      u(1, ll), ldu, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                              u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncc, work( nm12+1 ), work( nm13+1
                // ), c( ll, 1 ), ldc )
                char side = 'L';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(nm12 + 1),
                                             dwork(nm13 + 1), c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                                      c(ll, 1), ldc, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                              c(ll, 1), ldc);
                }
            }
            //
            //           test convergence
            //
            if(abs(e(m - 1)) <= thresh)
                e(m - 1) = zero;
        }
        else
        {
            /*
       *           chase bulge from bottom to top
       *           save cosines and sines for later singular vector updates
       */
            cs = one;
            oldcs = one;
            // do 130 i = m, ll + 1, -1
            for(i = m; i >= (ll + 1); i--)
            {
                auto di_cs = d(i) * cs;
                call_lartg(di_cs, e(i - 1), cs, sn, r);

                if(i < m)
                    e(i) = oldsn * r;

                auto oldcs_r = oldcs * r;
                auto dim1_sn = d(i - 1) * sn;
                call_lartg(oldcs_r, dim1_sn, oldcs, oldsn, d(i));

                work(i - ll) = cs;
                work(i - ll + nm1) = -sn;
                work(i - ll + nm12) = oldcs;
                work(i - ll + nm13) = -oldsn;
            }
        L130:
            h = d(ll) * cs;
            d(ll) = h * oldcs;
            e(ll) = h * oldsn;
            //
            //           update singular vectors
            //

            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // --------------
                    // copy rotations
                    // --------------
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncvt, work( nm12+1 ), work(
                // nm13+1
                // ), vt( ll, 1 ), ldvt );
                char side = 'L';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(nm12 + 1),
                                             dwork(nm13 + 1), vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                                      vt(ll, 1), ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                              vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'b', nru, m-ll+1, work( 1 ), work( n ), u( 1,
                // ll
                // ), ldu )
                char side = 'R';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(1), dwork(n),
                                             u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncc, work( 1 ), work( n ), c( ll,
                // 1
                // ), ldc )
                char side = 'L';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(1), dwork(n),
                                             c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc);
                }
            }
            //
            //           test convergence
            //
            if(abs(e(ll)) <= thresh)
                e(ll) = zero;
        }
    }
    else
    {
        //
        //        use nonzero shift
        //
        if(idir == 1)
        {
            //
            //           chase bulge from top to bottom
            //           save cosines and sines for later singular vector updates
            //
            f = (abs(d(ll)) - shift) * (sign(one, d(ll)) + shift / d(ll));
            g = e(ll);
            // do 140 i = ll, m - 1
            for(i = ll; i <= (m - 1); i++)
            {
                call_lartg(f, g, cosr, sinr, r);
                if(i > ll)
                    e(i - 1) = r;
                f = cosr * d(i) + sinr * e(i);
                e(i) = cosr * e(i) - sinr * d(i);
                g = sinr * d(i + 1);
                d(i + 1) = cosr * d(i + 1);
                call_lartg(f, g, cosl, sinl, r);
                d(i) = r;
                f = cosl * e(i) + sinl * d(i + 1);
                d(i + 1) = cosl * d(i + 1) - sinl * e(i);
                if(i < m - 1)
                {
                    g = sinl * e(i + 1);
                    e(i + 1) = cosl * e(i + 1);
                }
                work(i - ll + 1) = cosr;
                work(i - ll + 1 + nm1) = sinr;
                work(i - ll + 1 + nm12) = cosl;
                work(i - ll + 1 + nm13) = sinl;
            }
        L140:
            e(m - 1) = f;
            //
            //           update singular vectors
            //

            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // --------------
                    // copy rotations
                    // --------------
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncvt, work( 1 ), work( n ), vt(
                // ll, 1 ), ldvt )
                char side = 'L';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(1), dwork(n),
                                             vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1),
                                      ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1), ldvt);
                }
            }

            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'f', nru, m-ll+1, work( nm12+1 ), work( nm13+1
                // ), u( 1, ll ), ldu )
                char side = 'R';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(nm12 + 1),
                                             dwork(nm13 + 1), u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                                      u(1, ll), ldu, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                              u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncc, work( nm12+1 ), work( nm13+1
                // ), c( ll, 1 ), ldc )
                char side = 'L';
                char pivot = 'V';
                char direct = 'F';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(nm12 + 1),
                                             dwork(nm13 + 1), c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                                      c(ll, 1), ldc, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                              c(ll, 1), ldc);
                }
            }
            /*
       *           test convergence
       */
            if(abs(e(m - 1)) <= thresh)
                e(m - 1) = zero;
        }
        else
        {
            /*
       *           chase bulge from bottom to top
       *           save cosines and sines for later singular vector updates
       */
            f = (abs(d(m)) - shift) * (sign(one, d(m)) + shift / d(m));
            g = e(m - 1);
            // do 150 i = m, ll + 1, -1
            for(i = m; i >= (ll + 1); i--)
            {
                call_lartg(f, g, cosr, sinr, r);
                if(i < m)
                    e(i) = r;
                f = cosr * d(i) + sinr * e(i - 1);
                e(i - 1) = cosr * e(i - 1) - sinr * d(i);
                g = sinr * d(i - 1);
                d(i - 1) = cosr * d(i - 1);
                call_lartg(f, g, cosl, sinl, r);
                d(i) = r;
                f = cosl * e(i - 1) + sinl * d(i - 1);
                d(i - 1) = cosl * d(i - 1) - sinl * e(i - 1);
                if(i > ll + 1)
                {
                    g = sinl * e(i - 2);
                    e(i - 2) = cosl * e(i - 2);
                }
                work(i - ll) = cosr;
                work(i - ll + nm1) = -sinr;
                work(i - ll + nm12) = cosl;
                work(i - ll + nm13) = -sinl;
            }
        L150:
            e(ll) = f;
            //
            //           test convergence
            //
            if(abs(e(ll)) <= thresh)
                e(ll) = zero;
            //
            //           update singular vectors if desired
            //

            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // --------------
                    // copy rotations
                    // --------------
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncvt, work( nm12+1 ), work(
                // nm13+1
                // ), vt( ll, 1 ), ldvt )
                char side = 'L';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(nm12 + 1),
                                             dwork(nm13 + 1), vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                                      vt(ll, 1), ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                              vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'b', nru, m-ll+1, work( 1 ), work( n ), u( 1,
                // ll
                // ), ldu )
                char side = 'R';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(1), dwork(n),
                                             u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncc, work( 1 ), work( n ), c( ll,
                // 1
                // ), ldc )
                char side = 'L';
                char pivot = 'V';
                char direct = 'B';
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(1), dwork(n),
                                             c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc);
                }
            }
        }
    }
    CHECK_HIP(hipStreamSynchronize(stream));
    /*
   *     qr iteration finished, go back and check convergence
   */
    goto L60;

/*
 *     all singular values converged, so make them positive
 */
L160:
    // do 170 i = 1, n
    for(i = 1; i <= n; i++)
    {
        if(d(i) < zero)
        {
            d(i) = -d(i);
            //
            //           change sign of singular vectors, if desired
            //
            if(ncvt > 0)
            {
                if(use_gpu)
                {
                    call_scal_gpu(ncvt, negone, vt(i, 1), ldvt);
                }
                else
                {
                    call_scal(ncvt, negone, vt(i, 1), ldvt);
                }
            }
        }
    }
L170:
    //
    //     sort the singular values into decreasing order (insertion sort on
    //     singular values, but only one transposition per singular vector)
    //
    // do 190 i = 1, n - 1
    if(need_sort)
    {
        for(i = 1; i <= (n - 1); i++)
        {
            //
            //        scan for smallest d(i)
            //
            isub = 1;
            smin = d(1);
            // do 180 j = 2, n + 1 - i
            for(j = 2; j <= (n + 1 - i); j++)
            {
                if(d(j) <= smin)
                {
                    isub = j;
                    smin = d(j);
                }
            }
        L180:
            if(isub != n + 1 - i)
            {
                //
                //           swap singular values and vectors
                //
                d(isub) = d(n + 1 - i);
                d(n + 1 - i) = smin;
                if(ncvt > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(ncvt, vt(isub, 1), ldvt, vt(n + 1 - i, 1), ldvt);
                    }
                    else
                    {
                        call_swap(ncvt, vt(isub, 1), ldvt, vt(n + 1 - i, 1), ldvt);
                    }
                }
                if(nru > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(nru, u(1, isub), ione, u(1, n + 1 - i), ione);
                    }
                    else
                    {
                        call_swap(nru, u(1, isub), ione, u(1, n + 1 - i), ione);
                    }
                }
                if(ncc > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(ncc, c(isub, 1), ldc, c(n + 1 - i, 1), ldc);
                    }
                    else
                    {
                        call_swap(ncc, c(isub, 1), ldc, c(n + 1 - i, 1), ldc);
                    }
                }
            }
        }
    }
L190:
    goto L220;
//
//     maximum number of iterations exceeded, failure to converge
//
L200:
    info = 0;
    // do 210 i = 1, n - 1
    for(i = 1; i <= (n - 1); i++)
    {
        if(e(i) != zero)
            info = info + 1;
    }
L210:
L220:
    return;
    //
    //     end of dbdsqr
    //
}

template <typename T, typename S, typename W1, typename W2, typename W3, typename I = rocblas_int>
rocblas_status rocsolver_bdsqr_host_batch_template(rocblas_handle handle,
                                                   const rocblas_fill uplo_in,
                                                   const I n,
                                                   const I nv,
                                                   const I nu,
                                                   const I nc,
                                                   S* D,
                                                   const rocblas_stride strideD,
                                                   S* E,
                                                   const rocblas_stride strideE,
                                                   W1 V_arg,
                                                   const I shiftV,
                                                   const I ldv,
                                                   const rocblas_stride strideV,
                                                   W2 U_arg,
                                                   const I shiftU,
                                                   const I ldu,
                                                   const rocblas_stride strideU,
                                                   W3 C_arg,
                                                   const I shiftC,
                                                   const I ldc,
                                                   const rocblas_stride strideC,
                                                   I* info_array,
                                                   const I batch_count,
                                                   I* splits_map,
                                                   S* work)
{
    // -------------------------
    // copy D into hD, E into hE
    // -------------------------

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    W1 V = V_arg;
    W2 U = U_arg;
    W3 C = C_arg;

    auto is_device_pointer = [](void* ptr) -> bool {
        hipPointerAttribute_t dev_attributes;
        if(ptr == nullptr)
        {
            return (false);
        }

        auto istat = hipPointerGetAttributes(&dev_attributes, ptr);
        if(istat != hipSuccess)
        {
            std::cout << "is_device_pointer: istat = " << istat << " " << hipGetErrorName(istat)
                      << std::endl;
        }
        assert(istat == hipSuccess);
        return (dev_attributes.type == hipMemoryTypeDevice);
    };

    // ---------------------------------------------------
    // handle  batch case with array of pointers on device
    // ---------------------------------------------------
    std::vector<T*> Vp_array(batch_count);
    std::vector<T*> Up_array(batch_count);
    std::vector<T*> Cp_array(batch_count);

    if(nv > 0)
    {
        bool const is_device_V_arg = is_device_pointer((void*)V_arg);
        if(is_device_V_arg)
        {
            // ------------------------------------------------------------
            // note "T *" and "T * const" may be considered different types
            // ------------------------------------------------------------
            bool constexpr is_array_of_device_pointers
                = !(std::is_same<W1, T*>::value || std::is_same<W1, T* const>::value);
            bool constexpr need_copy_W1 = is_array_of_device_pointers;
            if constexpr(need_copy_W1)
            {
                size_t const nbytes = sizeof(T*) * batch_count;
                void* const dst = (void*)&(Vp_array[0]);
                void* const src = (void*)V_arg;
                HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                V = &(Vp_array[0]);
            }
        }
    }

    if(nu > 0)
    {
        bool const is_device_U_arg = is_device_pointer((void*)U_arg);
        if(is_device_U_arg)
        {
            bool constexpr is_array_of_device_pointers
                = !(std::is_same<W2, T*>::value || std::is_same<W2, T* const>::value);
            bool constexpr need_copy_W2 = is_array_of_device_pointers;
            if constexpr(need_copy_W2)
            {
                size_t const nbytes = sizeof(T*) * batch_count;
                void* const dst = (void*)&(Up_array[0]);
                void* const src = (void*)U_arg;
                HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                U = &(Up_array[0]);
            }
        }
    }

    if(nc > 0)
    {
        bool const is_device_C_arg = is_device_pointer((void*)C_arg);
        if(is_device_C_arg)
        {
            bool constexpr is_array_of_device_pointers
                = !(std::is_same<W3, T*>::value || std::is_same<W3, T* const>::value);
            bool constexpr need_copy_W3 = is_array_of_device_pointers;
            if constexpr(need_copy_W3)
            {
                size_t const nbytes = sizeof(T*) * batch_count;
                void* const dst = (void*)&(Cp_array[0]);
                void* const src = (void*)C_arg;
                HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                C = &(Cp_array[0]);
            }
        }
    }

    S* hD = nullptr;
    S* hE = nullptr;

    size_t const E_size = (n - 1);
    HIP_CHECK(hipHostMalloc(&hD, (sizeof(S) * std::max(1, batch_count)) * n));
    HIP_CHECK(hipHostMalloc(&hE, (sizeof(S) * std::max(1, batch_count)) * E_size));

    // ----------------------------------------------------
    // copy info_array[] on device to linfo_array[] on host
    // ----------------------------------------------------
    I* linfo_array = nullptr;
    HIP_CHECK(hipHostMalloc(&linfo_array, sizeof(I) * std::max(1, batch_count)));
    {
        void* const dst = &(linfo_array[0]);
        void* const src = &(info_array[0]);
        size_t const nbytes = sizeof(I) * batch_count;
        hipMemcpyKind const kind = hipMemcpyDeviceToHost;

        HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, kind, stream));
    }

    S* hwork = nullptr;
    HIP_CHECK(hipHostMalloc(&hwork, sizeof(S) * (4 * n)));

    // -------------------------------------------------
    // transfer arrays D(:) and E(:) from Device to Host
    // -------------------------------------------------

    bool const use_single_copy_for_D = (batch_count == 1) || (strideD == n);
    if(use_single_copy_for_D)
    {
        void* const dst = (void*)&(hD[0]);
        void* const src = (void*)&(D[0]);
        size_t const sizeBytes = sizeof(S) * n * batch_count;
        hipMemcpyKind const kind = hipMemcpyDeviceToHost;

        HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
    }
    else
    {
        for(I bid = 0; bid < batch_count; bid++)
        {
            void* const dst = (void*)&(hD[bid * n]);
            void* const src = (void*)&(D[bid * strideD]);
            size_t const sizeBytes = sizeof(S) * n;
            hipMemcpyKind const kind = hipMemcpyDeviceToHost;
            HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
        }
    }

    {
        for(I bid = 0; bid < batch_count; bid++)
        {
            void* const dst = (void*)&(hE[bid * E_size]);
            void* const src = (void*)&(E[bid * strideE]);
            size_t const sizeBytes = sizeof(S) * E_size;
            hipMemcpyKind const kind = hipMemcpyDeviceToHost;
            HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
        }
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    S* dwork_ = nullptr;
    HIP_CHECK(hipMalloc(&dwork_, sizeof(S) * (4 * n)));

    for(I bid = 0; bid < batch_count; bid++)
    {
        if(linfo_array[bid] != 0)
        {
            continue;
        };

        // std::vector<S> hwork(4 * n);

        char uplo = (uplo_in == rocblas_fill_lower) ? 'L' : 'U';
        S* d_ = &(hD[bid * n]);
        S* e_ = &(hE[bid * E_size]);

        T* v_ = (nv > 0) ? load_ptr_batch<T>(V, bid, shiftV, strideV) : nullptr;
        T* u_ = (nu > 0) ? load_ptr_batch<T>(U, bid, shiftU, strideU) : nullptr;
        T* c_ = (nc > 0) ? load_ptr_batch<T>(C, bid, shiftC, strideC) : nullptr;
        S* work_ = &(hwork[0]);
        // S* dwork = &(work[bid * (4 * n)]);

        I info = 0;

        I nru = nu;
        I ncc = nc;

        // -------------------------------------------------------
        // NOTE: lapack dbdsqr() accepts "VT" and "ldvt" for transpose of V
        // as input variable
        // However, rocsolver bdsqr() accepts variable called "V" and "ldv"
        // but may be  actually holding "VT"
        // -------------------------------------------------------
        T* vt_ = v_;
        I ldvt = ldv;

        I nrv = n;
        I ncvt = nv;
        bool const values_only = (ncvt == 0) && (nru == 0) && (ncc == 0);
        bool const use_lapack_bdsqr = false;

        if((use_lapack_bdsqr) && (values_only))
        {
            // --------------------------------
            // call the lapack version of bdsqr
            // --------------------------------
            auto ln = n;
            auto lncvt = ncvt;
            auto lnru = nru;
            auto lncc = ncc;
            S& d_arg = d_[0];
            S& e_arg = e_[0];
            T& vt_arg = vt_[0];
            T& u_arg = u_[0];
            T& c_arg = c_[0];
            S& work_arg = work_[0];
            auto ldvt_arg = ldvt;
            auto ldu_arg = ldu;
            auto ldc_arg = ldc;

            call_bdsqr(uplo, ln, lncvt, lnru, lncc, d_arg, e_arg, vt_arg, ldvt_arg, u_arg, ldu_arg,
                       c_arg, ldu_arg, work_arg, info);
        }
        else
        {
            bdsqr_single_template<S, T, I>(uplo, n, ncvt, nru, ncc,

                                           d_, e_,

                                           vt_, ldvt, u_, ldu, c_, ldc,

                                           work_, info, dwork_, stream);
        }

        if(info == 0)
        {
            // ----------------------------
            // explicitly zero out "E" array
            // to be compatible with rocsolver bdsqr
            // ----------------------------
            S const zero = S(0);
            for(I i = 0; i < (n - 1); i++)
            {
                e_[i] = zero;
            }
        }

        if(linfo_array[bid] == 0)
        {
            linfo_array[bid] = info;
        }
    } // end for bid

    // -------------------------------------------------
    // transfer arrays D(:) and E(:) from host to device
    // -------------------------------------------------

    if(use_single_copy_for_D)
    {
        void* const src = (void*)&(hD[0]);
        void* const dst = (void*)D;
        size_t const sizeBytes = sizeof(S) * n * batch_count;
        hipMemcpyKind const kind = hipMemcpyHostToDevice;

        HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
    }
    else
    {
        for(I bid = 0; bid < batch_count; bid++)
        {
            void* const src = (void*)&(hD[bid * n]);
            void* const dst = (void*)(D + bid * strideD);
            size_t const sizeBytes = sizeof(S) * n;
            hipMemcpyKind const kind = hipMemcpyHostToDevice;
            HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
        }
    }

    {
        for(I bid = 0; bid < batch_count; bid++)
        {
            void* const src = (void*)&(hE[bid * E_size]);
            void* const dst = (void*)&(E[bid * strideE]);
            size_t const sizeBytes = sizeof(S) * E_size;
            hipMemcpyKind const kind = hipMemcpyHostToDevice;
            HIP_CHECK(hipMemcpyAsync(dst, src, sizeBytes, kind, stream));
        }
    }

    {
        // ------------------------------------------------------
        // copy linfo_array[] from host to info_array[] on device
        // ------------------------------------------------------

        void* const src = (void*)&(linfo_array[0]);
        void* const dst = (void*)&(info_array[0]);
        size_t const nbytes = sizeof(I) * batch_count;
        hipMemcpyKind const kind = hipMemcpyHostToDevice;

        HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, kind, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    // ----------------------
    // free allocated storage
    // ----------------------

    HIP_CHECK(hipHostFree(hwork));
    hwork = nullptr;

    HIP_CHECK(hipHostFree(hD));
    hD = nullptr;

    HIP_CHECK(hipHostFree(hE));
    hE = nullptr;

    HIP_CHECK(hipFree(dwork_));
    dwork_ = nullptr;

    HIP_CHECK(hipHostFree(linfo_array));
    linfo_array = nullptr;

    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
#undef LASR_MAX_NTHREADS
#undef CHECK_HIP
