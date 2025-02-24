
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <complex>
#include <limits>

#include "hip/hip_cooperative_groups.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef LAUNCH_CHECK
#define LAUNCH_CHECK(fcn)                                                                  \
    {                                                                                      \
        auto const istat = (fcn);                                                          \
        bool const isok = (istat == hipSuccess);                                           \
        if(!isok)                                                                          \
        {                                                                                  \
            std::cerr << "Kernel launch error: " << hipGetErrorString(istat) << std::endl; \
        }                                                                                  \
        assert(isok);                                                                      \
    }
#endif

namespace cg = cooperative_groups;

template <typename T>
__device__ T reduce_sum(cg::thread_group g, T* temp, T val)
{
    auto const lane = g.thread_rank();
    g.sync();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for(auto i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane < i)
            val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

#if(0)
template <typename T>
__device__ int reduce_sum_shfl(cg::thread_block g, T const val)
{
    g.sync();
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for(auto i = g.size() / 2; i > 0; i /= 2)
    {
        val += __shfl_down(val, i);
        g.sync();
    }
    return val; // note: only thread 0 will return full sum
}
#endif

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static __device__ void gatomicAdd(double* const ptr, double const val)
{
    atomicAdd(ptr, val);
}

static __device__ void gatomicAdd(float* const ptr, float const val)
{
    atomicAdd(ptr, val);
}

static __device__ void gatomicAdd(rocblas_complex_num<float>* const ptr,
                                  rocblas_complex_num<float> const val)
{
    float* const p_real = (float*)ptr;
    float* const p_imag = p_real + 1;
    atomicAdd(p_real, val.real());
    atomicAdd(p_imag, val.imag());
}

static __device__ void gatomicAdd(rocblas_complex_num<double>* const ptr,
                                  rocblas_complex_num<double> const val)
{
    double* const p_real = (double*)ptr;
    double* const p_imag = p_real + 1;
    atomicAdd(p_real, val.real());
    atomicAdd(p_imag, val.imag());
}

template <typename I>
static __device__ __host__ I indxg2tile(I const ia, I const mb, I const myprow, I const nprow)
{
    I const itile = (ia / mb);
    return (itile);
}

// ------------------------------------------
// given a global index
// return the processor that holds this entry
// ------------------------------------------
template <typename I>
static __device__ __host__ I indxg2proc(I const ia, I const mb, I const myprow, I const nprow)
{
    I const itile = indxg2tile(ia, mb, myprow, nprow);
    I const iproc = (itile % nprow);

    return (iproc);
}

// --------------------------------------------
// given a global index
// return the first tile "jtile" that belongs to myprow
// --------------------------------------------
template <typename I>
static __device__ __host__ I first_tile(I const ia, I const mb, I const myprow, I const nprow)
{
    I const itile = (ia / mb);
    I const iproc = (itile % nprow);
    I const jtile = (itile + ((myprow + nprow - iproc) % nprow));

    assert((jtile % nprow) == myprow);

    return (jtile);
}

template <typename S>
static __device__ __host__ S lamch(char const ctype)
{
    if((ctype == 'E') || (ctype == 'e'))
    {
        return (std::numeric_limits<S>::epsilon() / 2);
    }
    if((ctype == 'S') || (ctype == 's'))
    {
        return (std::numeric_limits<S>::min());
    }
    if((ctype == 'B') || (ctype == 'b'))
    {
        return (std::numeric_limits<S>::base());
    }
    if((ctype == 'P') || (ctype == 'p'))
    {
        S const eps = std::numeric_limits<S>::epsilon() / 2;
        S const base = std::numeric_limits<S>::base();
        return (eps * base);
    }
    if((ctype == 'O') || (ctype == 'o'))
    {
        return (std::numeric_limits<S>::max());
    }
    return (0);
}

// --------------------------------------------------------------
// computes  sqrt( x^2 + y^2 + z^2 ) without unnecessary overflow
// assume x, y, z are not complex types
// --------------------------------------------------------------
template <typename S>
static __device__ S lapy3(S const x, S const y, S const z)
{
    assert(!rocblas_is_complex<S>);

    auto square = [](auto d) { return (d * d); };

    auto const xabs = std::abs(x);
    auto const yabs = std::abs(y);
    auto const zabs = std::abs(z);
    auto const w = std::max(xabs, std::max(yabs, zabs));
    auto const ans = (w == 0)
        ? (xabs + yabs + zabs)
        : w * std::sqrt(square(xabs / w) + square(yabs / w) + square(zabs / w));
    return (ans);
}

// -------------------------------------------------------
// computes sqrt( x^2 + y^2 ) without unnecessary overflow
// assume x, y are not complex types
// -------------------------------------------------------
template <typename S>
static __device__ S lapy2(S const x, S const y)
{
    S const one = 1;
    S const zero = 0;
    bool const x_is_nan = std::isnan(x);
    bool const y_is_nan = std::isnan(y);
    S dlapy2 = zero;
    if(x_is_nan)
    {
        dlapy2 = x;
    }
    if(y_is_nan)
    {
        dlapy2 = y;
    }

    auto const hugeval = lamch<S>('O'); // overflow

    auto square = [](auto d) { return (d * d); };

    if(!(x_is_nan || y_is_nan))
    {
        auto const xabs = std::abs(x);
        auto const yabs = std::abs(y);
        auto const w = std::max(xabs, yabs);
        auto const z = std::min(xabs, yabs);
        if((z == zero) || (w > hugeval))
        {
            dlapy2 = w;
        }
        else
        {
            dlapy2 = w * std::sqrt(one + square(z / w));
        }
    }
    return (dlapy2);
}

// ---------------------------------------
// Fortran sign intrinsic
// return  abs value of a times  sign of b
// ---------------------------------------
template <typename S>
static __device__ S sign(S const a, S const b)
{
    return (std::abs(a) * ((b < 0) ? -1 : 1));
}

// -------------------------------------------
// compute complex division in real arithmetic
// p + I * q = (a + I * b)/( c + I * d )
// -------------------------------------------
template <typename S>
static __device__ __host__ void ladiv(S const a, S const b, S const c, S const d, S& p, S& q)
{
    {
        assert(!rocblas_is_complex<S>);
    }

    S const bs = 2.0;
    S const half = 0.5;
    S const two = 2.0;
    S const one = 1.0;

    auto ladiv2 = [](S const a, S const b, S const c, S const d, S const r, S const t) -> S {
        S dladiv2 = 0;
        if(r != 0)
        {
            auto const br = b * r;
            if(br != 0)
            {
                dladiv2 = (a + br) * t;
            }
            else
            {
                dladiv2 = a * t + (b * t) * r;
            }
        }
        else
        {
            dladiv2 = (a + d * (b / c)) * t;
        }
        return (dladiv2);
    };

    auto ladiv1 = [](S& a, S& b, S& c, S& d, S& p, S& q) {
        S const one = 1.0;
        auto r = d / c;
        auto t = one / (c + d * r);
        p = ladiv2(a, b, c, d, r, t);
        a = -a;
        q = ladiv2(b, a, c, d, r, t);
    };

    S aa = a;
    S bb = b;
    S cc = c;
    S dd = d;

    S const ab = std::max(std::abs(a), std::abs(b));
    S const cd = std::max(std::abs(c), std::abs(d));
    S s = one;

    auto const ov = lamch<S>('O'); // overflow
    auto const un = lamch<S>('S'); // safe min
    auto const eps = lamch<S>('E'); // epsilon
    auto const be = bs / (eps * eps);

    if(cd >= half * ov)
    {
        cc = half * cc;
        dd = half * dd;
        s = half * s;
    }

    if(ab <= un * bs / eps)
    {
        aa *= be;
        bb *= be;
        s = s / be;
    }

    if(cd <= un * bs / eps)
    {
        cc *= be;
        dd *= be;
        s *= be;
    }

    if(std::abs(d) <= std::abs(c))
    {
        ladiv1(aa, bb, cc, dd, p, q);
    }
    else
    {
        ladiv1(bb, aa, dd, cc, p, q);
        q = -q;
    }

    p *= s;
    q *= s;

#ifdef NDEBUG
#else
    {
        // -------------------------------------
        // extra check
        //
        // p + I * q = (a + I * b)/( c + I * d )
        //
        // or
        //
        // (p + I * q) * (c + I * d ) == (a + I * b )
        // -------------------------------------
        double const tol = 20 * eps;
        auto const zpq = rocblas_complex_num<double>{double{p}, double{q}};
        auto const zcd = rocblas_complex_num<double>{double{c}, double{d}};
        auto const zab = rocblas_complex_num<double>{double{a}, double{b}};

        bool const isok = (std::abs(zpq * zcd - zab) <= tol * std::abs(zpq));
        assert(isok);
    }
#endif
}

template <typename T, typename I>
__device__ void Xscale_body(I const n,
                            T const alpha,
                            T* const X_,
                            I const ix,
                            I const jx,
                            I const ldx,
                            I const incx,
                            I const mb,
                            I const nb,
                            I const myprow,
                            I const mypcol,
                            I const nprow,
                            I const npcol)

{
    {
        bool const has_work = (n >= 1);
        if(!has_work)
        {
            return;
        }
    }

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    T const zero = 0;
    bool const is_alpha_zero = (alpha == zero);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };
    // ----------------------------------------
    // incx can only be 1 (column) or ldx (row)
    // ----------------------------------------
    bool const is_column_X = (incx == 1);
    {
        assert((incx == 1) || (incx == ldx));
    }

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto i) -> T& { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    if(is_column_X)
    {
        I const tile_start = first_tile(ix, mb, myprow, nprow);
        I const tile_end = first_tile(ix + n - 1, mb, myprow, nprow);
        I const tile_inc = nprow;

        I const ntiles = (tile_end - tile_start) / tile_inc;
        assert((ntiles * tile_inc) == (tile_end - tile_start));

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            I const tile = tile_start + it * tile_inc;
            I const i_start = std::max(ix, tile * mb);
            I const i_end = std::min(ix + n - 1, tile * mb + (mb - 1));

            if(is_alpha_zero)
            {
                for(auto i = (i_start + tix); i <= i_end; i += nx)
                {
                    Xvec(i - ix) = zero;
                }
            }
            else
            {
                for(auto i = (i_start + tix); i <= i_end; i += nx)
                {
                    Xvec(i - ix) *= alpha;
                }
            }
        }
    }
    else
    {
        auto const tile_start = first_tile(jx, nb, mypcol, npcol);
        auto const tile_end = first_tile(jx + n - 1, nb, mypcol, npcol);
        auto const tile_inc = npcol;

        auto const ntiles = (tile_end - tile_start) / tile_inc;
        assert((ntiles * tile_inc) == (tile_end - tile_start));

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const tile = tile_start + it * tile_inc;

            auto const j_start = std::max(jx, tile * nb);
            auto const j_end = std::min(jx + n - 1, tile * nb + (nb - 1));

            if(is_alpha_zero)
            {
                for(auto j = (j_start + tix); j <= j_end; j += nx)
                {
                    Xvec(j - jx) = zero;
                }
            }
            else
            {
                for(auto j = (j_start + tix); j <= j_end; j += nx)
                {
                    Xvec(j - jx) *= alpha;
                }
            }
        }
    }
}

template <typename T, typename I, typename Istride, typename UX>
static __global__ void Xscale_batch_kernel(I const n,
                                           T const* const p_alpha,
                                           Istride const stride_alpha,
                                           UX X_,
                                           Istride const shift_X,
                                           I const ix,
                                           I const jx,
                                           I const ldx,
                                           I const incx,
                                           Istride const stride_X,
                                           I const batch_count,
                                           I const mb,
                                           I const nb)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    for(I bid = 0; bid < batch_count; bid++)
    {
        T const alpha = *(p_alpha + bid * stride_alpha);

        auto const Xp = load_ptr_batch(X_, bid, shift_X, stride_X);
        Xscale_body(n, alpha, Xp, ix, jx, ldx, incx, mb, nb, myprow, mypcol, nprow, npcol);
    }
}

// -------------------------------
// compute the L2 norm of a vector
// and return in "ans"
// -------------------------------
template <typename T, typename I, typename S>
static void __device__ Xnrm2_body(cg::grid_group cg_grid,
                                  I const n,
                                  T const* const X_,
                                  I const ix,
                                  I const jx,
                                  I const ldx,
                                  I const incx,
                                  I const mb,
                                  I const nb,
                                  I const myprow,
                                  I const mypcol,
                                  I const nprow,
                                  I const npcol,
                                  S* const ans)
{
    if(n <= 0)
    {
        return;
    };

    bool constexpr is_complex = rocblas_is_complex<T>;

    {
        assert(cg_grid.is_valid());
    }

    if(cg_grid.thread_rank() == 0)
    {
        *ans = 0;
    }
    cg_grid.sync();

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    bool const is_column_X = (incx == 1);
    {
        assert((incx == 1) || (incx == ldx));
    }

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto const i) -> T& { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    extern __shared__ double lmem[];

    double* dnorm_sh = (double*)&(lmem[0]);

    double dnorm = 0;
    if(is_column_X)
    {
        auto const itile_start = first_tile(ix, mb, myprow, nprow);
        auto const itile_end = first_tile(ix + n - 1, mb, myprow, nprow);
        auto const ntiles = (itile_end - itile_start) / nprow;

        {
            assert(itile_start <= itile_end);
            assert(ntiles * nprow == (itile_end - itile_start));
        }

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const itile = itile_start + it * nprow;
            auto const istart = std::max(ix, itile * mb);
            auto const iend = std::min(ix + n - 1, itile * mb + (mb - 1));

            for(auto iix = (istart + tix); iix <= iend; iix += nx)
            {
                T const xi = Xvec((iix - ix));
                if(is_complex)
                {
                    dnorm += std::norm(xi);
                }
                else
                {
                    dnorm += xi * xi;
                }
            }
        }
    }
    else
    {
        auto const jtile_start = first_tile(jx, nb, mypcol, npcol);
        auto const jtile_end = first_tile(jx + n - 1, nb, mypcol, npcol);
        auto const ntiles = (jtile_end - jtile_start) / npcol;

        {
            assert(jtile_start <= jtile_end);
            assert(ntiles * npcol == (jtile_end - jtile_start));
        }

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const jtile = jtile_start + it * npcol;
            auto const jstart = std::max(jx, it * nb);
            auto const jend = std::min(jx + n - 1, it * nb + (nb - 1));

            for(auto jjx = (jstart + tix); jjx <= jend; jjx += nx)
            {
                T const xj = Xvec((jjx - jx));
                if(is_complex)
                {
                    dnorm += std::norm(xj);
                }
                else
                {
                    dnorm += xj * xj;
                }
            }
        }
    }

    auto const cg_block = cg::this_thread_block();
    dnorm = reduce_sum(cg_block, dnorm_sh, dnorm);

    bool const need_atomic_update = (is_column_X) ? (nprow > 1) : (npcol > 1);
    if(cg_block.thread_rank() == 0)
    {
        if(need_atomic_update)
        {
            atomicAdd(ans, static_cast<S>(dnorm));
        }
        else
        {
            *ans += static_cast<S>(dnorm);
        }
    }

    cg_grid.sync();
}

// -------------------------------
// compute the dot product
// and return in "ans"
//
//
// NOTE: assume *ans has been set to zero
// -------------------------------
template <typename T, typename I>
static void __device__ Xdot_body(I const n,

                                 T const* const X_,
                                 I const ix,
                                 I const jx,
                                 I const ldx,
                                 I const incx,

                                 T const* const Y_,
                                 I const iy,
                                 I const jy,
                                 I const ldy,
                                 I const incy,

                                 T* const ans,

                                 I const mb,
                                 I const nb,
                                 I const myprow,
                                 I const mypcol,
                                 I const nprow,
                                 I const npcol)
{
    if(n <= 0)
    {
        return;
    };

    bool constexpr is_complex = rocblas_is_complex<T>;

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    bool const is_column_X = (incx == 1);
    bool const is_column_Y = (incy == 1);
    {
        assert((incx == 1) || (incx == ldx));
        assert((incy == 1) || (incy == ldy));
    }

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto const i) -> T { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    auto const ip_Y = idx2D(iy, jy, ldy);
    auto Yvec = [=](auto const i) -> T { return (Y_[ip_Y + i * static_cast<int64_t>(incy)]); };

    extern __shared__ double lmem[];

    T* dsum_sh = (T*)&(lmem[0]);

    T dsum = 0;
    if(is_column_X)
    {
        auto const itile_start = first_tile(ix, mb, myprow, nprow);
        auto const itile_end = first_tile(ix + n - 1, mb, myprow, nprow);
        auto const ntiles = (itile_end - itile_start) / nprow;

        {
            assert(itile_start <= itile_end);
            assert(ntiles * nprow == (itile_end - itile_start));
        }

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const itile = itile_start + it * nprow;
            auto const istart = std::max(ix, itile * mb);
            auto const iend = std::min(ix + n - 1, itile * mb + (mb - 1));

            for(auto iix = (istart + tix); iix <= iend; iix += nx)
            {
                T const xi = Xvec((iix - ix));
                T const yi = Yvec((iix - ix));
                if constexpr(is_complex)
                {
                    dsum += conj(xi) * yi;
                }
                else
                {
                    dsum += xi * yi;
                }
            }
        }
    }
    else
    {
        auto const jtile_start = first_tile(jx, nb, mypcol, npcol);
        auto const jtile_end = first_tile(jx + n - 1, nb, mypcol, npcol);
        auto const ntiles = (jtile_end - jtile_start) / npcol;

        {
            assert(jtile_start <= jtile_end);
            assert(ntiles * npcol == (jtile_end - jtile_start));
        }

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const jtile = jtile_start + it * npcol;
            auto const jstart = std::max(jx, it * nb);
            auto const jend = std::min(jx + n - 1, it * nb + (nb - 1));

            for(auto jjx = (jstart + tix); jjx <= jend; jjx += nx)
            {
                T const xj = Xvec((jjx - jx));
                T const yj = Yvec((jjx - jx));
                if constexpr(is_complex)
                {
                    dsum += conj(xj) * yj;
                }
                else
                {
                    dsum += xj * yj;
                }
            }
        }
    }

    auto const cg_block = cg::this_thread_block();
    dsum = reduce_sum(cg_block, dsum_sh, dsum);

    bool const need_atomic_update = (is_column_X) ? (nprow > 1) : (npcol > 1);
    if(cg_block.thread_rank() == 0)
    {
        if(need_atomic_update)
        {
            atomicAdd(ans, (dsum));
        }
        else
        {
            *ans += (dsum);
        }
    }
}

template <typename T, typename I, typename Istride, typename UX, typename UY>
__global__ static void Xdot_batch_kernel(I const n,

                                         UX X_,
                                         Istride const shiftX,
                                         I const ix,
                                         I const jx,
                                         I const ldx,
                                         I const incx,
                                         Istride const strideX,

                                         UY Y_,
                                         Istride const shiftY,
                                         I const iy,
                                         I const jy,
                                         I const ldy,
                                         I const incy,
                                         Istride const strideY,

                                         I const batch_count,
                                         I const mb,
                                         I const nb,
                                         T* const ans,
                                         T* const work)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    for(I bid = 0; bid < batch_count; bid++)
    {
        T const* const Xp = load_ptr_batch(X_, bid, shiftX, strideX);
        T const* const Yp = load_ptr_batch(Y_, bid, shiftY, strideY);
        T* const ansp = ans + bid;

        Xdot_body<T, I>(n, Xp, ix, jx, ldx, incx, Yp, iy, jy, ldy, incy, ansp,

                        mb, nb, myprow, mypcol, nprow, npcol);
    }
}

template <typename T, typename I, typename Istride, typename UX, typename UY, typename Tex>
static void rocsolverCall_dot(rocblas_handle handle,
                              I const n,

                              UX X_,
                              Istride const shiftX,
                              I const ix,
                              I const jx,
                              I const ldx,
                              I const incx,
                              Istride const strideX,

                              UY Y_,
                              Istride const shiftY,
                              I const iy,
                              I const jy,
                              I const ldy,
                              I const incy,
                              Istride const strideY,

                              I const batch_count,
                              T* const norms,
                              I const mb,
                              I const nb,
                              Tex* const workspace,
                              T** const workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto const nx = 32;
    auto const ny = 32;
    auto const num_cu = get_num_cu();
    size_t const ld_size = 64 * 1024;

    {
        auto const istat = hipMemset(norms, 0, sizeof(T) * batch_count);
        assert(istat == hipSuccess);
    }

    Xdot_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(nx, ny, 1), ld_size, stream>>>(
        n, X_, shiftX, ix, jx, ldx, incx, strideX, Y_, shiftY, iy, jy, ldy, incy, strideY,
        batch_count, mb, nb, norms, workspace);
}

template <typename T, typename I>
__device__ void Xlacgv_body(I const n,
                            T* const X_,
                            I const ix,
                            I const jx,
                            I const ldx,
                            I const incx,
                            I const mb,
                            I const nb,
                            I const myprow,
                            I const mypcol,
                            I const nprow,
                            I const npcol)

{
    {
        bool const has_work = (n >= 1);
        if(!has_work)
        {
            return;
        }
    }

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };
    // ----------------------------------------
    // incx can only be 1 (column) or ldx (row)
    // ----------------------------------------
    bool const is_column_X = (incx == 1);
    assert((incx == 1) || (incx == ldx));

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto i) -> T& { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    if(is_column_X)
    {
        I const tile_start = first_tile(ix, mb, myprow, nprow);
        I const tile_end = first_tile(ix + n - 1, mb, myprow, nprow);
        I const tile_inc = nprow;

        I const ntiles = (tile_end - tile_start) / tile_inc;

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            I const tile = tile_start + it * tile_inc;
            I const i_start = std::max(ix, tile * mb);
            I const i_end = std::min(ix + n - 1, tile * mb + (mb - 1));

            {
                for(auto i = (i_start + tix); i <= i_end; i += nx)
                {
                    Xvec(i - ix) = conj(Xvec(i - ix));
                }
            }
        }
    }
    else
    {
        auto const tile_start = first_tile(jx, nb, mypcol, npcol);
        auto const tile_end = first_tile(jx + n - 1, nb, mypcol, npcol);
        auto const tile_inc = npcol;

        auto const ntiles = (tile_end - tile_start) / tile_inc;

        for(auto it = (0 + tiy); it <= ntiles; it += ny)
        {
            auto const tile = tile_start + it * tile_inc;

            auto const j_start = std::max(jx, tile * nb);
            auto const j_end = std::min(jx + n - 1, tile * nb + (nb - 1));

            {
                for(auto j = (j_start + tix); j <= j_end; j += nx)
                {
                    Xvec(j - jx) = conj(Xvec(j - jx));
                }
            }
        }
    }
}

template <typename T, typename I, typename Istride, typename UX>
static __global__ void Xlacgv_batch_kernel(I const n,
                                           UX X_,
                                           Istride const shift_X,
                                           I const ix,
                                           I const jx,
                                           I const ldx,
                                           I const incx,
                                           Istride const stride_X,
                                           I const batch_count,
                                           I const mb,
                                           I const nb)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    for(I bid = 0; bid < batch_count; bid++)
    {
        auto const Xp = load_ptr_batch(X_, bid, shift_X, stride_X);
        Xlacgv_body(n, Xp, ix, jx, ldx, incx, mb, nb, myprow, mypcol, nprow, npcol);
    }
}

template <typename T, typename I, typename Istride, typename UX>
static void rocsolverCall_lacgv_template(rocblas_handle handle,
                                         I const n,
                                         UX X_,
                                         Istride const shiftX,
                                         I const ix,
                                         I const jx,
                                         I const ldx,
                                         I const incx,
                                         Istride strideX,
                                         I const batch_count,
                                         I const mb,
                                         I const nb)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    size_t const lds_size = 64 * 1024;
    auto const NX = 32;
    auto const NY = 32;
    auto const num_cu = get_num_cu();

    Xlacgv_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(NX, NY, 1), lds_size, stream>>>(
        n, X_, shiftX, ix, jx, ldx, incx, strideX, batch_count, mb, nb);
}

// ----------------------
// matrix vector multiply
// Yvec = alpha * op(A(0:(m-1),0:(n-1)) * Xvec
// where
// op(A) can be  A or
//               transpose(A) or
//               conj(transpose(A))
// ----------------------
template <typename T, typename I>
static __device__ void Xgemv_body(char const trans,
                                  I const m,
                                  I const n,
                                  T const alpha,

                                  T const* const A_,
                                  I const ia,
                                  I const ja,
                                  I const lda,

                                  T const* const X_,
                                  I const ix,
                                  I const jx,
                                  I const ldx,
                                  I const incx,

                                  T* const Y_,
                                  I const iy,
                                  I const jy,
                                  I const ldy,
                                  I const incy,

                                  I const mb,
                                  I const nb,
                                  I const myprow,
                                  I const mypcol,
                                  I const nprow,
                                  I const npcol)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (alpha != 0);
        if(!has_work)
        {
            return;
        }
    }

    {
        bool const is_valid_lda = (lda >= 1) && (lda >= m);
        assert(is_valid_lda);
    }

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_no_transpose = (trans == 'N') || (trans == 'n');
    {
        bool const is_valid_trans = is_transpose || is_conj_transpose || is_no_transpose;
        assert(is_valid_trans);
    }

    bool const is_column_Y = (incy == 1);
    bool const is_column_X = (incx == 1);

    {
        bool const is_valid_incx = ((incx == 1) || (incx == ldx));
        bool const is_valid_incy = ((incy == 1) || (incy == ldy));

        assert(is_valid_incx);
        assert(is_valid_incy);
    }

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto const ip_Y = idx2D(iy, jy, ldy);
    auto Yvec = [=](auto i) -> T& {
        assert((0 <= i) && (i < ((is_no_transpose) ? m : n)));
        return (Y_[ip_Y + i * static_cast<int64_t>(incy)]);
    };

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto i) {
        assert((0 <= i) && (i < ((is_no_transpose) ? n : m)));
        return (X_[ip_X + i * static_cast<int64_t>(incx)]);
    };

    auto A = [=](auto i, auto j) {
        assert((ia <= i) && (i <= (ia + m - 1)));
        assert((ja <= j) && (j <= (ja + n - 1)));
        return (A_[idx2D(i, j, lda)]);
    };

    // bool const need_atomic_update = (is_no_transpose) ? (npcol > 1) : (nprow > 1);
    bool const need_atomic_update = true;

    I const itileA_start = first_tile(ia, mb, myprow, nprow);
    I const itileA_end = first_tile(ia + m - 1, mb, myprow, nprow);
    I const itile_inc = nprow;

    I const jtileA_start = first_tile(ja, nb, mypcol, npcol);
    I const jtileA_end = first_tile(ja + n - 1, nb, mypcol, npcol);
    I const jtile_inc = npcol;

    I const tixy = tix + tiy * nx;
    I const nxny = nx * ny;
    I const mbnb = (is_no_transpose) ? nb : mb;

    extern __shared__ double lmem[];
    T* const Xvec_sh = (T*)&(lmem[0]);
    T* const rsum_sh_ = Xvec_sh + mbnb;

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    auto const ld_rsum_sh = is_even(nx) ? nx + 1 : nx;

    auto rsum_sh = [=](auto tx, auto ty) -> T& { return (rsum_sh_[tx + ty * ld_rsum_sh]); };

    for(auto i = (0 + tixy); i < (ld_rsum_sh)*ny; i += nxny)
    {
        rsum_sh_[i] = 0;
    }

    for(auto i = (0 + tixy); i < mbnb; i += nxny)
    {
        Xvec_sh[i] = 0;
    }

    __syncthreads();

    auto cgwave = cg::tiled_partition(cg::this_thread_block(), nx);

#if(0)
    {
        if(cg::this_grid().thread_rank() == 0)
        {
            printf("trans=%c, m=%d, n=%d, ia=%d, ja=%d\n", trans, m, n, ia, ja);
        }
    }
#endif

    if(is_no_transpose)
    {
        // -----------------
        // Y += alpha * A * X
        // -----------------
        for(auto jtileA = jtileA_start; jtileA <= jtileA_end; jtileA += jtile_inc)
        {
            for(auto itileA = itileA_start; itileA <= itileA_end; itileA += itile_inc)
            {
                // ------------------------------
                // process tile "(itileA,jtileA)"
                // ------------------------------

                auto const jstart = std::max(ja, jtileA * nb);
                auto const jend = std::min(ja + n - 1, jtileA * nb + (nb - 1));

                auto const istart = std::max(ia, itileA * mb);
                auto const iend = std::min(ia + m - 1, itileA * mb + (mb - 1));

                for(auto jja = (jstart + tixy); jja <= jend; jja += nxny)
                {
                    Xvec_sh[(jja - jstart)] = Xvec(jja - ja);
                }

                __syncthreads();

                for(auto iia = (istart + tiy); iia <= iend; iia += ny)
                {
                    T rsum = 0;
                    for(auto jja = (jstart + tix); jja <= jend; jja += nx)
                    {
                        auto const xj = Xvec_sh[(jja - jstart)];
                        auto const aij = A(iia, jja);
                        rsum += aij * xj;
                    }
                    rsum_sh(tix, tiy) = rsum;
                    rsum = reduce_sum(cgwave, &(rsum_sh(0, tiy)), rsum);

                    if(cgwave.thread_rank() == 0)
                    {
                        auto const alpha_rsum = alpha * rsum;
                        auto const ioff = (iia - ia);
                        if(need_atomic_update)
                        {
                            gatomicAdd(&(Yvec(ioff)), alpha_rsum);
                        }
                        else
                        {
                            Yvec(ioff) += alpha_rsum;
                        }
                    }
                }

                __syncthreads();

            } // end for itileA
        } // end for jtileA
    }
    else
    {
        // -----------------
        // Yvec(0:(n-1)) += alpha * trans(A(0:(m-1),0:(n-1)) * Xvec(0:(m-1))
        // -----------------
        for(auto jtileA = jtileA_start; jtileA <= jtileA_end; jtileA += jtile_inc)
        {
            for(auto itileA = itileA_start; itileA <= itileA_end; itileA += itile_inc)
            {
                // ------------------------------
                // process tile "(itileA,jtileA)"
                // ------------------------------
                auto const jstart = std::max(ja, jtileA * nb);
                auto const jend = std::min(ja + n - 1, jtileA * nb + (nb - 1));

                auto const istart = std::max(ia, itileA * mb);
                auto const iend = std::min(ia + m - 1, itileA * mb + (mb - 1));

                for(auto iia = (istart + tixy); iia <= iend; iia += nxny)
                {
                    Xvec_sh[(iia - istart)] = Xvec((iia - ia));
                }

                __syncthreads();

                for(auto jja = (jstart + tiy); jja <= jend; jja += ny)
                {
                    T rsum = 0;
                    for(auto iia = (istart + tix); iia <= iend; iia += nx)
                    {
                        T const aij = A(iia, jja);
                        T const xi = Xvec_sh[(iia - istart)];

                        if(is_complex && is_conj_transpose)
                        {
                            rsum += conj(aij) * xi;
                        }
                        else
                        {
                            rsum += aij * xi;
                        }
                    }

                    rsum_sh(tix, tiy) = rsum;
                    rsum = reduce_sum(cgwave, &(rsum_sh(0, tiy)), rsum);

                    if(cgwave.thread_rank() == 0)
                    {
                        auto const alpha_rsum = alpha * rsum;
                        auto const joff = (jja - ja);
                        if(need_atomic_update)
                        {
                            gatomicAdd(&(Yvec(joff)), alpha_rsum);
                        }
                        else
                        {
                            Yvec(joff) += alpha_rsum;
                        }
                    }
                } // end for jja

                __syncthreads();

            } // end for itileA
        } // end for jtileA
    }
    __syncthreads();
}

template <typename T, typename I, typename Istride, typename UA, typename UX, typename UY>
static __global__ void Xgemv_batch_kernel(char const trans,
                                          I const m,
                                          I const n,
                                          T const* const p_alpha,
                                          Istride const stride_alpha,
                                          UA A_,
                                          Istride const shift_A,
                                          I const ia,
                                          I const ja,
                                          I const lda,
                                          Istride const stride_A,
                                          UX X_,
                                          Istride const shift_X,
                                          I const ix,
                                          I const jx,
                                          I const ldx,
                                          I const incx,
                                          Istride const stride_X,
                                          UY Y_,
                                          Istride const shift_Y,
                                          I const iy,
                                          I const jy,
                                          I const ldy,
                                          I const incy,
                                          Istride const stride_Y,
                                          I const batch_count,
                                          I const mb,
                                          I const nb)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    for(I bid = 0; bid < batch_count; bid++)
    {
        T const alpha = *(p_alpha + bid * stride_alpha);

        T const* const Ap = load_ptr_batch(A_, bid, shift_A, stride_A);
        T const* const Xp = load_ptr_batch(X_, bid, shift_X, stride_X);
        T* const Yp = load_ptr_batch(Y_, bid, shift_Y, stride_Y);

        Xgemv_body(trans, m, n, alpha, Ap, ia, ja, lda, Xp, ix, jx, ldx, incx, Yp, iy, jy, ldy,
                   incy, mb, nb, myprow, mypcol, nprow, npcol);
    }
}

template <typename T, typename I, typename Istride, typename UA, typename UX, typename UY>
static void rocsolverCall_gemv(rocblas_handle handle,
                               rocblas_operation trans,
                               I const mm,
                               I const nn,
                               T const* const p_alpha,
                               Istride const stride_alpha,
                               UA A,
                               I const shiftA,
                               I const ia,
                               I const ja,
                               I const lda,
                               Istride const strideA,
                               UX X,
                               I const shiftX,
                               I const ix,
                               I const jx,
                               I const ldx,
                               I const incx,
                               Istride const strideX,
                               T const* const p_beta,
                               Istride const stride_beta,
                               UY Y,
                               I const shiftY,
                               I const iy,
                               I const jy,
                               I const ldy,
                               I const incy,
                               Istride const strideY,
                               I const batch_count,
                               I const mb,
                               I const nb,
                               void* work_Arr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto const num_cu = get_num_cu();
    size_t const lds_size = 64 * 1024;

    // -----------------------------
    // scale output Y vector by beta
    // -----------------------------
    auto const nx = 32;
    auto const ny = 32;

    {
        bool const is_valid_trans = (trans == rocblas_operation_none)
            || (trans == rocblas_operation_transpose)
            || (trans == rocblas_operation_conjugate_transpose);
        assert(is_valid_trans);
    }

    char const ctrans = (trans == rocblas_operation_none)  ? 'N'
        : (trans == rocblas_operation_transpose)           ? 'T'
        : (trans == rocblas_operation_conjugate_transpose) ? 'C'
                                                           : '?';

    bool const is_no_transpose = (trans == rocblas_operation_none);
    auto const len_Yvec = is_no_transpose ? mm : nn;

    Xscale_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(nx, ny, 1), lds_size, stream>>>(
        len_Yvec, p_beta, stride_beta, Y, shiftY, iy, jy, ldy, incy, strideY, batch_count, mb, nb);

    Xgemv_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(nx, ny, 1), lds_size, stream>>>(
        ctrans, mm, nn, p_alpha, stride_alpha, A, shiftA, ia, ja, lda, strideA, X, shiftX, ix, jx,
        ldx, incx, strideX, Y, shiftY, iy, jy, ldy, incy, strideY, batch_count, mb, nb);
}

// ----------------------
// matrix vector multiple
// y = alpha * op(A) * x
// where
// op(A) can be  A
//     or transpose(A)
//     or conj(transpose(A))
// ----------------------
template <typename T, typename I>
static __device__ void Xsymv_body(char const c_uplo,
                                  I const n,
                                  T const alpha,

                                  T const* const A_,
                                  I const ia,
                                  I const ja,
                                  I const lda,

                                  T const* const X_,
                                  I const ix,
                                  I const jx,
                                  I const ldx,
                                  I const incx,

                                  T* const Y_,
                                  I const iy,
                                  I const jy,
                                  I const ldy,
                                  I const incy,

                                  I const mb,
                                  I const nb,
                                  I const myprow,
                                  I const mypcol,
                                  I const nprow,
                                  I const npcol)
{
    bool const is_complex = rocblas_is_complex<T>;
    auto const m = n;

    auto const tix = hipThreadIdx_x;
    auto const tiy = hipThreadIdx_y;
    auto const nx = hipBlockDim_x;
    auto const ny = hipBlockDim_y;

    bool const is_upper = (c_uplo == 'U') || (c_uplo == 'u');
    bool const is_lower = (c_uplo == 'L') || (c_uplo == 'l');
    {
        bool const is_valid = (is_lower || is_upper);
        assert(is_valid);
    }

    bool const is_column_X = (incx == 1);
    bool const is_column_Y = (incy == 1);
    {
        assert((incx == 1) || (incx == ldx));
        assert((incy == 1) || (incy == ldy));
    }

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto const ip_Y = idx2D(iy, jy, ldy);
    auto Yvec = [=](auto i) -> T& { return (Y_[ip_Y + i * static_cast<int64_t>(incy)]); };

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto i) { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    auto A = [=](auto i, auto j) { return (A_[idx2D(i, j, lda)]); };

    auto const itileA_start = first_tile(ia, mb, myprow, nprow);
    auto const jtileA_start = first_tile(ja, nb, mypcol, npcol);
    auto const itileA_end = first_tile(ia + n - 1, mb, myprow, nprow);
    auto const jtileA_end = first_tile(ja + n - 1, nb, mypcol, npcol);
    auto const itile_inc = nprow;
    auto const jtile_inc = npcol;

    extern __shared__ double lmem[];
    T* const ytmp_row = (T*)&(lmem[0]);
    T* const ytmp_col = ytmp_row + mb;

    {
        size_t const lds_size = 64 * 1024;
        bool const lds_ok = ((sizeof(T) * (mb + nb)) <= lds_size);
        assert(lds_ok);
    }

    auto const tixy = tix + tiy * nx;
    auto const nxny = nx * ny;

    for(auto i = (0 + tixy); i < mb; i += nxny)
    {
        ytmp_row[i] = 0;
    }
    for(auto i = (0 + tixy); i < nb; i += nxny)
    {
        ytmp_col[i] = 0;
    }
    __syncthreads();

    // -----------------
    // Y1 = [D1  L21' ]  * X1
    // Y2 = [L21   D2 ]    X2
    //
    //
    // Y1 = (D1 * X1) + (L21' * X2)
    // Y2 = (L21 * X1) + ( D2 * X2 )
    // -----------------
    for(auto jtileA = jtileA_start; jtileA <= jtileA_end; jtileA += jtile_inc)
    {
        for(auto itileA = itileA_start; itileA <= itileA_end; itileA += itile_inc)
        {
            // ------------------------------
            // process tile "(itileA,jtileA)"
            // ------------------------------

            auto const jstart = std::max(ja, jtileA * nb);
            auto const jend = std::min(ja + n - 1, jtileA * nb + (nb - 1));

            auto const istart = std::max(ia, itileA * mb);
            auto const iend = std::min(ia + m - 1, itileA * mb + (mb - 1));

            bool const is_strictly_lower = ((istart - ia) > (jend - ja));
            bool const is_strictly_upper = ((jstart - ja) > (iend - ia));
            bool const can_skip_tile
                = (is_lower && is_strictly_upper) || (is_upper && is_strictly_lower);
            if(can_skip_tile)
            {
                continue;
            }

            for(auto jja = (jstart + tiy); jja <= jend; jja += ny)
            {
                auto const xj = Xvec((jja - ja));
                for(auto iia = (istart + tix); iia <= iend; iia += nx)
                {
                    auto const ioff = (iia - istart);
                    auto const joff = (jja - jstart);
                    bool const is_diagonal = ((iia - ia) == (jja - ja));
                    bool const is_strictly_lower = ((iia - ia) > (jja - ja));
                    bool const is_strictly_upper = ((iia - ia) < (jja - ja));

                    bool const can_skip
                        = (is_lower && is_strictly_upper) || (is_upper && is_strictly_lower);

                    if(can_skip)
                    {
                        continue;
                    };

                    if(is_diagonal)
                    {
                        auto const ajj = std::real(A(iia, jja));
                        gatomicAdd(&(ytmp_row[ioff]), ajj * xj);
                    }
                    else
                    {
                        // ------------------
                        // off-diagonal entry
                        // ------------------
                        T const aij = A(iia, jja);
                        T const atji = (is_complex) ? conj(aij) : aij;

                        T const xi = Xvec((iia - ia));

                        gatomicAdd(&(ytmp_row[ioff]), aij * xj);
                        gatomicAdd(&(ytmp_col[joff]), atji * xi);
                    }
                }
            }

            __syncthreads();

            if(tiy == 0)
            {
                for(auto iia = (istart + tix); iia <= iend; iia += nx)
                {
                    auto const iiy = (iia - ia);
                    auto const ioff = (iia - istart);
                    auto const alpha_ytmp_row = alpha * ytmp_row[ioff];

                    bool const need_atomic_update = true;
                    if(need_atomic_update)
                    {
                        gatomicAdd(&(Yvec(iiy)), alpha_ytmp_row);
                    }
                    else
                    {
                        Yvec(iiy) += alpha_ytmp_row;
                    }
                    ytmp_row[ioff] = 0;
                }

                for(auto jja = (jstart + tix); jja <= jend; jja += nx)
                {
                    auto const jjy = (jja - ja);
                    auto const joff = (jja - jstart);
                    auto const alpha_ytmp_col = alpha * ytmp_col[joff];

                    bool const need_atomic_update = true;
                    if(need_atomic_update)
                    {
                        gatomicAdd(&(Yvec(jjy)), alpha_ytmp_col);
                    }
                    else
                    {
                        Yvec(jjy) += alpha_ytmp_col;
                    }
                    ytmp_col[joff] = 0;
                }
            }
            __syncthreads();

        } // end for itileA
    } // end for jtileA

    __syncthreads();
}

// ----------------------
// matrix vector multiple
// y = alpha * op(A) * x
// where
// op(A) can be  A
//     or transpose(A)
//     or conj(transpose(A))
// ----------------------
template <typename T, typename I>
static __device__ void Xsymv_body_v2(char const c_uplo,
                                     I const n,
                                     T const alpha,

                                     T const* const A_,
                                     I const ia,
                                     I const ja,
                                     I const lda,

                                     T const* const X_,
                                     I const ix,
                                     I const jx,
                                     I const ldx,
                                     I const incx,

                                     T* const Y_,
                                     I const iy,
                                     I const jy,
                                     I const ldy,
                                     I const incy,

                                     I const mb,
                                     I const nb,
                                     I const myprow,
                                     I const mypcol,
                                     I const nprow,
                                     I const npcol)
{
    bool const is_complex = rocblas_is_complex<T>;
    auto const m = n;
    {
        // assume operate on symmetric diagonal
        // (ia == ja)
        assert((ia == ja));
        assert((mb == nb));
    }

    auto const tix = hipThreadIdx_x;
    auto const tiy = hipThreadIdx_y;
    auto const nx = hipBlockDim_x;
    auto const ny = hipBlockDim_y;

    bool const is_upper = (c_uplo == 'U') || (c_uplo == 'u');
    bool const is_lower = (c_uplo == 'L') || (c_uplo == 'l');

    bool const is_column_X = (incx == 1);
    bool const is_column_Y = (incy == 1);
    {
        assert((is_lower) || (is_upper));
        assert((incx == 1) || (incx == ldx));
        assert((incy == 1) || (incy == ldy));
    }

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto const ip_Y = idx2D(iy, jy, ldy);
    auto Yvec = [=](auto i) -> T& { return (Y_[ip_Y + i * static_cast<int64_t>(incy)]); };

    auto const ip_X = idx2D(ix, jx, ldx);
    auto Xvec = [=](auto i) { return (X_[ip_X + i * static_cast<int64_t>(incx)]); };

    auto A = [=](auto i, auto j) { return (A_[idx2D(i, j, lda)]); };

    auto const itileA_start = first_tile(ia, mb, myprow, nprow);
    auto const jtileA_start = first_tile(ja, nb, mypcol, npcol);
    auto const itileA_end = first_tile(ia + n - 1, mb, myprow, nprow);
    auto const jtileA_end = first_tile(ja + n - 1, nb, mypcol, npcol);
    auto const itile_inc = nprow;
    auto const jtile_inc = npcol;

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };
    auto const ldtmp = (is_even(nx)) ? nx + 1 : nx;

    extern __shared__ double lmem[];
    T* pfree = (T*)&(lmem[0]);

    size_t total_len = 0;
    T* const tmp_ = pfree;
    pfree += ldtmp * ny;
    total_len += ldtmp * ny;
    T* const ysh_j = pfree;
    pfree += nb;
    total_len += nb;
    T* const xsh_j = pfree;
    pfree += nb;
    total_len += nb;

    auto tmp = [=](auto tx, auto ty) -> T& { return (tmp_[tx + ty * ldtmp]); };

    {
        size_t const lds_size = 64 * 1024;
        bool const lds_ok = (sizeof(T) * total_len <= lds_size);
        assert(lds_ok);
    }

    auto const tixy = tix + tiy * nx;
    auto const nxny = nx * ny;

    // -----------------
    // Y1 = [D1  L21' ]  * X1
    // Y2 = [L21   D2 ]    X2
    //
    //
    // Y1 = (D1 * X1) + (L21' * X2)
    // Y2 = (L21 * X1) + ( D2 * X2 )
    // -----------------

    auto const cg_wave = cg::tiled_partition(cg::this_thread_block(), nx);

    for(auto jtileA = jtileA_start; jtileA <= jtileA_end; jtileA += jtile_inc)
    {
        auto const jstart = std::max(ja, jtileA * nb);
        auto const jend = std::min(ja + n - 1, jtileA * nb + (nb - 1));

        for(auto jja = (jstart + tixy); jja <= jend; jja += nxny)
        {
            ysh_j[(jja - jstart)] = 0;
            xsh_j[(jja - jstart)] = Xvec((jja - ja));
        }

        __syncthreads();

        for(auto itileA = itileA_start; itileA <= itileA_end; itileA += itile_inc)
        {
            auto const istart = std::max(ia, itileA * mb);
            auto const iend = std::min(ia + m - 1, itileA * mb + (mb - 1));

            bool const is_lower_tile = (istart > jend);
            bool const is_upper_tile = (jstart > iend);

            bool const do_work = (is_lower && is_lower_tile) || (is_upper && is_upper_tile);
            if(!do_work)
            {
                continue;
            }

            // ------------------------------
            // process tile "(itileA,jtileA)"
            // ------------------------------

            for(auto iia = (istart + tiy); iia <= iend; iia += ny)
            {
                auto const ioff = (iia - istart);
                auto const xi = Xvec((iia - ia));

                T y_i = 0;
                for(auto jja = (jstart + tix); jja <= jend; jja += nx)
                {
                    auto const joff = (jja - jstart);
                    auto const xj = xsh_j[joff];

                    bool const is_diagonal = (iia == jja);
                    bool const is_strictly_lower = (iia > jja);
                    bool const is_strictly_upper = (jja > iia);

                    if(is_diagonal)
                    {
                        auto const aii = std::real(A(iia, iia));
                        y_i += aii * xi;
                    }
                    else
                    {
                        // ------------------
                        // off-diagonal entry
                        // ------------------

                        bool const do_work
                            = (is_lower && is_strictly_lower) || (is_upper && is_strictly_upper);

                        if(!do_work)
                        {
                            continue;
                        };

                        T const aij = A(iia, jja);
                        y_i += aij * xj;

                        if constexpr(is_complex)
                        {
                            T const atji = conj(aij);
                            gatomicAdd(&(ysh_j[joff]), atji * xi);
                        }
                        else
                        {
                            T const atji = aij;
                            gatomicAdd(&(ysh_j[joff]), atji * xi);
                        }
                    }
                } // end for jja

                y_i = reduce_sum(cg_wave, &(tmp(0, tiy)), y_i);
                if(cg_wave.thread_rank() == 0)
                {
                    gatomicAdd(&(Yvec(iia - ia)), alpha * y_i);
                }
            } // end for iia
        } // end for itileA

        __syncthreads();

        for(auto jja = (jstart + tixy); jja <= jend; jja += nxny)
        {
            auto const joff = (jja - jstart);
            gatomicAdd(&(Yvec((jja - ja))), alpha * ysh_j[joff]);
            ysh_j[joff] = 0;
        }
        __syncthreads();

    } // end for jtileA
}

// compute Yvec(0:(n-1)) += alpha * op(A) * Xvec(0:(n-1))
// where A is symmetric/hermitian matrix
// op(A) is lower triangular or upper triangular
//
template <typename T, typename I, typename Istride, typename UA, typename UX, typename UY>
static __global__ void Xsymv_batch_kernel(char c_uplo,
                                          I const n,
                                          T const* const p_alpha,
                                          Istride const stride_alpha,
                                          UA A_,
                                          Istride const shiftA,
                                          I const ia,
                                          I const ja,
                                          I const lda,
                                          Istride const strideA,
                                          UX X_,
                                          Istride const shiftX,
                                          I const ix,
                                          I const jx,
                                          I const ldx,
                                          I const incx,
                                          Istride const strideX,
                                          UY Y_,
                                          Istride const shiftY,
                                          I const iy,
                                          I const jy,
                                          I const ldy,
                                          I const incy,
                                          Istride const strideY,
                                          I const batch_count,
                                          I const mb,
                                          I const nb)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    for(I bid = 0; bid < batch_count; bid++)
    {
        T const alpha = *(p_alpha + bid * stride_alpha);
        T const* const Ap = load_ptr_batch(A_, bid, shiftA, strideA);
        T const* const Xp = load_ptr_batch(X_, bid, shiftX, strideX);
        T* const Yp = load_ptr_batch(Y_, bid, shiftY, strideY);

        Xsymv_body(c_uplo, n, alpha, Ap, ia, ja, lda, Xp, ix, jx, ldx, incx, Yp, iy, jy, ldy, incy,
                   mb, nb, myprow, mypcol, nprow, npcol);
    }
}

template <typename T, typename I, typename Istride, typename UA, typename UX, typename UY>
rocblas_status rocsolverCall_symv_hemv(rocblas_handle handle,
                                       rocblas_fill const uplo,
                                       I const n,
                                       T const* const p_alpha,
                                       Istride const stride_alpha,
                                       UA A,
                                       Istride const shiftA,
                                       I const ia,
                                       I const ja,
                                       I const lda,
                                       Istride const strideA,
                                       UX X,
                                       Istride const shiftX,
                                       I const ix,
                                       I const jx,
                                       I const ldx,
                                       I const incx,
                                       Istride const strideX,
                                       T const* const p_beta,
                                       Istride stride_beta,
                                       UY Y,
                                       Istride const shiftY,
                                       I const iy,
                                       I const jy,
                                       I const ldy,
                                       I const incy,
                                       Istride const strideY,
                                       I const batch_count,
                                       I const mb,
                                       I const nb,
                                       T* work,
                                       T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto const num_cu = get_num_cu();
    size_t const lds_size = 64 * 1024;

    // -----------------------------
    // scale output Y vector by beta
    // -----------------------------
    auto const nx = 32;
    auto const ny = 32;

    Xscale_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(nx, ny, 1), lds_size, stream>>>(
        n, p_beta, stride_beta, Y, shiftY, iy, jy, ldy, incy, strideY, batch_count, mb, nb);

    char const c_uplo = (uplo == rocblas_fill_upper) ? 'U'
        : (uplo == rocblas_fill_lower)               ? 'L'
                                                     : '?';

    Xsymv_batch_kernel<T, I, Istride><<<dim3(num_cu, 1, 1), dim3(nx, ny, 1), lds_size, stream>>>(
        c_uplo, n, p_alpha, stride_alpha, A, shiftA, ia, ja, lda, strideA, X, shiftX, ix, jx, ldx,
        incx, strideX, Y, shiftY, iy, jy, ldy, incy, strideY, batch_count, mb, nb);

    return (rocblas_status_success);
}

/** set_offdiag kernel copies the off-diagonal element of A, which is the non-zero element
    resulting by applying the Householder reflector to the working column, to E. Then set it
    to 1 to prepare for the application of the Householder reflector to the rest of the matrix **/

template <typename T, typename I, typename Istride, typename UA, typename UE>
static __global__ void set_offdiag_batch_kernel(const rocblas_int batch_count,
                                                UA A_,
                                                I const shiftA,
                                                Istride const strideA,
                                                UE E_,
                                                Istride const strideE)
{
    I const nthreads_per_block = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;
    I const nblocks = hipGridDim_x * hipGridDim_y * hipGridDim_z;
    I const bid_inc = nthreads_per_block * nblocks;

    I const ithread = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    I const iblock = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x
        + hipBlockIdx_z * (hipGridDim_x * hipGridDim_y);

    I const bid_start = ithread + iblock * nthreads_per_block;

    Istride const shiftE = 0;

    bool constexpr is_complex = rocblas_is_complex<T>;

    for(I bid = (0 + bid_start); bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch(A_, bid, shiftA, strideA);
        auto const E = load_ptr_batch(E_, bid, shiftE, strideE);

        if constexpr(is_complex)
        {
            E[0] = std::real(A[0]);
        }
        else
        {
            E[0] = A[0];
        }
        A[0] = T(1);
    }
}

template <typename T, typename I, typename Istride, typename UA>
static __global__ void lower_stage1(I const n,
                                    I const j,

                                    T* scalars,

                                    T* const W_,
                                    Istride const shiftW,
                                    I const ldw,
                                    Istride const strideW,

                                    T* const A_,
                                    Istride const shiftA,
                                    I const lda,
                                    Istride const strideA,

                                    I const batch_count,

                                    I const mb,
                                    I const nb)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    auto cg_grid = cg::this_grid();

    for(I bid = 0; bid < batch_count; bid++)
    {
        T* const A = load_ptr_batch(A_, bid, shiftA, strideA);
        T* const W = load_ptr_batch(W_, bid, shiftW, strideW);

#if(0)

        if(COMPLEX)
        {
            rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                        batch_count);
        }
#else
        if(is_complex)
        {
            I const nn = j;
            Xlacgv_body(nn,

                        W, j, 0, ldw, ldw,

                        mb, nb, myprow, mypcol, nprow, npcol);
            cg_grid.sync();
        }

#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none,

                            n - j, j,

                            cast2constType<T>(scalars), 0,

                            A, shiftA + idx2D(j, 0, lda), lda, strideA,

                            W, shiftW + idx2D(j, 0, ldw), ldw, strideW,

                            cast2constType<T>(scalars + 2), 0,

                            A, shiftA + idx2D(j, j, lda), 1, strideA,

                            batch_count, workArr);
#else
        {
            char const trans = 'N';
            T const alpha = *scalars;
            T const beta = *(scalars + 2);
            I const mm = n - j;
            I const nn = j;

            I const len_Y = mm;

            Xscale_body(mm, beta,

                        A, j, j, lda, 1,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();

            Xgemv_body(trans, mm, nn, alpha,

                       A, j, 0, lda,

                       W, j, 0, ldw, ldw,

                       A, j, j, lda, 1,

                       mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif

#if(0)

        if(COMPLEX)
        {
            rocsolver_lacgv_template<T>(handle, j,

                                        W, shiftW + idx2D(j, 0, ldw), ldw, strideW,

                                        batch_count);

            rocsolver_lacgv_template<T>(handle, j,

                                        A, shiftA + idx2D(j, 0, lda), lda, strideA,

                                        batch_count);
        }
#else
        if(is_complex)
        {
            I const nn = j;
            Xlacgv_body(nn, W, j, 0, ldw, ldw,

                        mb, nb, myprow, mypcol, nprow, npcol);

            Xlacgv_body(nn, A, j, 0, lda, lda,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j, cast2constType<T>(scalars), 0,

                            W, shiftW + idx2D(j, 0, ldw), ldw, strideW,

                            A, shiftA + idx2D(j, 0, lda), lda, strideA,

                            cast2constType<T>(scalars + 2), 0,

                            A, shiftA + idx2D(j, j, lda), 1, strideA,

                            batch_count, workArr);
#else
        {
            char trans = 'N';
            I const mm = n - j;
            I const nn = j;
            I const len_Y = mm;

            T const alpha = *scalars;
            T const beta = *(scalars + 2);

            Xscale_body(len_Y, beta,

                        A, j, j, lda, 1,

                        mb, mb, myprow, mypcol, nprow, npcol);
            cg_grid.sync();

            Xgemv_body(trans, mm, nn, alpha,

                       W, j, 0, ldw,

                       A, j, 0, lda, lda,

                       A, j, j, lda, 1,

                       mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif

#if(0)
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                        batch_count);

#else
        if(is_complex)
        {
            I const nn = j;
            Xlacgv_body(nn,

                        A, j, 0, lda, lda,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA>
static __global__ void upper_stage1(I const n,
                                    I const j,
                                    I const jw,

                                    T* const scalars,

                                    T* const W_,
                                    Istride const shiftW,
                                    I const ldw,
                                    Istride const strideW,

                                    UA A_,
                                    Istride const shiftA,
                                    I const lda,
                                    Istride const strideA,

                                    I const batch_count,
                                    I const mb,
                                    I const nb)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    auto cg_grid = cg::this_grid();

    for(I bid = 0; bid < batch_count; bid++)
    {
        T* const A = load_ptr_batch(A_, bid, shiftA, strideA);
        T* const W = load_ptr_batch(W_, bid, shiftW, strideW);

#if(0)
        // update column j of A with reflector computed in step j-1
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw), ldw,
                                        strideW, batch_count);
#else
        if(is_complex)
        {
            I const nn = n - 1 - j;
            Xlacgv_body<T>(nn,

                           W, j, jw + 1, ldw, ldw,

                           mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,

                            cast2constType<T>(scalars), 0,

                            A, shiftA + idx2D(0, j + 1, lda), lda, strideA,

                            W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,

                            cast2constType<T>(scalars + 2), 0,

                            A, shiftA + idx2D(0, j, lda), 1, strideA,

                            batch_count, workArr);
#else
        {
            char const trans = 'N';
            I const mm = j + 1;
            I const nn = n - 1 - j;
            auto const alpha = *scalars;
            auto const beta = *(scalars + 2);

            I const len_Y = mm;
            Xscale_body(len_Y, beta,

                        A, 0, j, lda, 1,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();

            Xgemv_body(trans, mm, nn, alpha,

                       A, 0, j + 1, lda,

                       W, j, jw + 1, ldw, ldw,

                       A, 0, j, lda, 1,

                       mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif

#if(0)
        if(COMPLEX)
        {
            rocsolver_lacgv_template<T>(handle, n - 1 - j,

                                        W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,

                                        batch_count);
            rocsolver_lacgv_template<T>(handle, n - 1 - j,

                                        A, shiftA + idx2D(j, j + 1, lda), lda, strideA,

                                        batch_count);
        }
#else
        if(is_complex)
        {
            auto const nn = n - 1 - j;

            Xlacgv_body(nn,

                        W, j, jw + 1, ldw, ldw,

                        mb, nb, myprow, mypcol, nprow, npcol);

            Xlacgv_body(nn, A, j, j + 1, lda, lda,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                            cast2constType<T>(scalars), 0,

                            W, shiftW + idx2D(0, jw + 1, ldw), ldw, strideW,

                            A, shiftA + idx2D(j, j + 1, lda), lda, strideA,

                            cast2constType<T>(scalars + 2), 0,

                            A, shiftA + idx2D(0, j, lda), 1, strideA,

                            batch_count, workArr);
#else
        {
            char const trans = 'N';
            I const mm = j + 1;
            I const nn = n - 1 - j;
            auto const len_Y = mm;

            auto const alpha = *scalars;
            auto const beta = *(scalars + 2);

            Xscale_body(len_Y, beta, A, 0, j, lda, 1, mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();

            Xgemv_body(trans, mm, nn, alpha,

                       W, 0, jw + 1, ldw,

                       A, j, j + 1, lda, lda,

                       A, 0, j, lda, 1,

                       mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif

#if(0)
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - 1 - j,

                                        A, shiftA + idx2D(j, j + 1, lda), lda, strideA,

                                        batch_count);
#else
        if(is_complex)
        {
            auto const nn = n - 1 - j;
            Xlacgv_body(nn, A, j, j + 1, lda, lda,

                        mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA, typename UW>
static __global__ void upper_stage2(I const n,
                                    I const j,
                                    I const jw,

                                    T const* const scalars,
                                    T* const tau,
                                    Istride const strideP,
                                    T* const norms,

                                    UW W_,
                                    Istride const shiftW,
                                    I const ldw,
                                    Istride const strideW,

                                    UA A_,
                                    Istride const shiftA,
                                    I const lda,
                                    Istride const strideA,

                                    I const batch_count,
                                    I const mb,
                                    I const nb)
{
    auto const myprow = hipBlockIdx_x;
    auto const mypcol = hipBlockIdx_y;
    auto const nprow = hipGridDim_x;
    auto const npcol = hipGridDim_y;

    auto cg_grid = cg::this_grid();

    for(I bid = 0; bid < batch_count; bid++)
    {
        T* const A = load_ptr_batch(A_, bid, shiftA, strideA);
        T* const W = load_ptr_batch(W_, bid, shiftW, strideW);

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                            cast2constType<T>(scalars + 2), 0,

                            W, shiftW + idx2D(0, jw + 1, ldw), ldw, strideW,

                            A, shiftA + idx2D(0, j, lda), 1, strideA,

                            cast2constType<T>(scalars + 1), 0,

                            W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,

                            batch_count, workArr);
#else
        {
            char const trans = 'C';
            I const mm = j;
            I const nn = n - 1 - j;
            I const len_Y = nn;

            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            Xscale_body<T, I>(len_Y, beta, W, j + 1, jw, ldw, 1, mb, nb, myprow, mypcol, nprow,
                              npcol);

            cg_grid.sync();

            Xgemv_body<T, I>(trans, mm, nn, alpha,

                             W, 0, jw + 1, ldw,

                             A, 0, j, lda, 1,

                             W, j + 1, jw, ldw, 1,

                             mb, nb, myprow, mypcol, nprow, npcol);
            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                            cast2constType<T>(scalars), 0,

                            A, shiftA + idx2D(0, j + 1, lda), lda, strideA,

                            W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,

                            cast2constType<T>(scalars + 2), 0,

                            W, shiftW + idx2D(0, jw, ldw), 1, strideW,

                            batch_count, workArr);
#else
        {
            char const trans = 'N';
            I const mm = j;
            I const nn = n - 1 - j;
            I const len_Y = mm;

            T const alpha = *scalars;
            T const beta = *(scalars + 2);

            Xscale_body<T, I>(len_Y, beta,

                              W, 0, jw, ldw, 1,

                              mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();

            Xgemv_body<T, I>(trans, mm, nn, alpha,

                             A, 0, j + 1, lda,

                             W, j + 1, jw, ldw, 1,

                             W, 0, jw, ldw, 1,

                             mb, nb, myprow, mypcol, nprow, npcol);
            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                            cast2constType<T>(scalars + 2), 0,

                            A, shiftA + idx2D(0, j + 1, lda), lda, strideA,

                            A, shiftA + idx2D(0, j, lda), 1, strideA,

                            cast2constType<T>(scalars + 1), 0,

                            W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,

                            batch_count, workArr);
#else
        {
            char const trans = 'C';
            I const mm = j;
            I const nn = n - 1 - j;
            I const len_Y = nn;

            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            Xscale_body<T, I>(len_Y, beta, W, j + 1, jw, ldw, 1, mb, nb, myprow, mypcol, nprow,
                              npcol);
            cg_grid.sync();

            Xgemv_body<T, I>(trans, mm, nn, alpha,

                             A, 0, j + 1, lda,

                             A, 0, j, lda, 1,

                             W, j + 1, jw, ldw, 1,

                             mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                            cast2constType<T>(scalars), 0,

                            W, shiftW + idx2D(0, jw + 1, ldw), ldw, strideW,

                            W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,

                            cast2constType<T>(scalars + 2), 0,

                            W, shiftW + idx2D(0, jw, ldw), 1, strideW,

                            batch_count, workArr);
#else
        {
            char const trans = 'N';
            I const mm = j;
            I const nn = n - 1 - j;
            I const len_Y = mm;

            T const alpha = *(scalars);
            T const beta = *(scalars + 2);

            Xscale_body<T, I>(len_Y, beta,

                              W, 0, jw, ldw, 1,

                              mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();

            Xgemv_body<T, I>(trans, mm, nn, alpha,

                             W, 0, jw + 1, ldw,

                             W, j + 1, jw, ldw, 1,

                             W, 0, jw, ldw, 1,

                             mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }

#endif

#if(0)
        rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W, shiftW + idx2D(0, jw, ldw), 1,
                            strideW, batch_count);
#else
        {
            I const nn = j;
            T const* const p_alpha = (tau + j - 1);
            T const alpha = *(p_alpha + bid * strideP);

            Xscale_body<T, I>(nn, alpha,

                              W, 0, jw, ldw, 1,

                              mb, nb, myprow, mypcol, nprow, npcol);

            cg_grid.sync();
        }
#endif

#if(0)
        rocblasCall_dot<COMPLEX, T>(handle, j,

                                    W, shiftW + idx2D(0, jw, ldw), 1, strideW,

                                    A, shiftA + idx2D(0, j, lda), 1, strideA,

                                    batch_count, norms, work, workArr);
#else
        {
            I const nn = j;
            Xdot_body<T, I>(nn,

                            W, 0, jw, ldw, 1,

                            A, 0, j, lda, 1,

                            &(norms[bid]),

                            mb, nb, myprow, mypcol, nprow, npcol);
            cg_grid.sync();
        }
#endif

    } // end for bid
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_coop_template(rocblas_handle handle,
                                             const rocblas_fill uplo,
                                             const rocblas_int n,
                                             const rocblas_int k,
                                             U A,
                                             const rocblas_int shiftA,
                                             const rocblas_int lda,
                                             const rocblas_stride strideA,
                                             S* E,
                                             const rocblas_stride strideE,
                                             T* tau,
                                             const rocblas_stride strideP,
                                             T* W,
                                             const rocblas_int shiftW,
                                             const rocblas_int ldw,
                                             const rocblas_stride strideW,
                                             const rocblas_int batch_count,
                                             T* scalars,
                                             T* work,
                                             T* norms,
                                             T** workArr)
{
    ROCSOLVER_ENTER("latrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // configure kernels
    rocblas_int blocks = (batch_count - 1) / BS1 + 1;
    dim3 grid_b(blocks, 1);
    dim3 threads(BS1, 1, 1);
    blocks = (n - 1) / BS1 + 1;
    dim3 grid_n(blocks, batch_count);

    auto const nx = 32;
    auto const ny = 32;
    bool const use_org = true;
    bool const use_coop = true;
    size_t lds_size = 64 * 1024;
    auto const mb = k;
    auto const nb = k;

    auto const num_cu = get_num_cu();

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            if(use_coop)
            {
#if(0)

                template <typename T, typename I, typename Istride, typename UA>
                static __global__ void lower_stage1(
                    I const n, I const j,

                    T* scalars,

                    T* const W_, Istride const shiftW, I const ldw, Istride const strideW,

                    T* const A_, Istride const shiftA, I const lda, Istride const strideA,

                    I const batch_count,

                    I const mb, I const nb)
#else
                rocblas_stride const lshiftW = shiftW;
                rocblas_stride const lshiftA = shiftA;

                void* args[]
                    = {(void*)&n,           (void*)&j,       (void*)&scalars,

                       (void*)&W,           (void*)&lshiftW, (void*)&ldw,     (void*)&strideW,

                       (void*)&A,           (void*)&lshiftA, (void*)&lda,     (void*)&strideA,

                       (void*)&batch_count,

                       (void*)&mb,          (void*)&nb};

                auto const nx = 32;
                auto const ny = 32;
                auto const lds_size = 64 * 1024;

                LAUNCH_CHECK(hipLaunchCooperativeKernel(
                    (void*)(lower_stage1<T, rocblas_int, rocblas_stride, U>), dim3(num_cu, 1, 1),
                    dim3(nx, ny, 1), args, lds_size, stream));

#endif
            }
            else
            {
                if(COMPLEX)
                {
                    rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw,
                                                strideW, batch_count);
                }

                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda),
                                    lda, strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda),
                                    1, strideA, batch_count, workArr);

                if(COMPLEX)
                {
                    rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw,
                                                strideW, batch_count);
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);
                }

                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                    cast2constType<T>(scalars), 0, W, shiftW + idx2D(j, 0, ldw),
                                    ldw, strideW, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda),
                                    1, strideA, batch_count, workArr);

                if(COMPLEX)
                {
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);
                }
            }

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                     shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1, strideA,
                                     (tau + j), strideP, batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            if(use_org)
            {
                ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                        shiftA + idx2D(j + 1, j, lda), strideA, (E + j), strideE);
            }
            else
            {
                set_offdiag_batch_kernel<T, rocblas_int, rocblas_stride>
                    <<<dim3(num_cu, 1, 1), dim3(32, 32, 1), 0, stream>>>(
                        batch_count, A, shiftA + idx2D(j + 1, j, lda), strideA, (E + j), strideE);
            }

            // compute/update column j of W
            if(use_org)
            {
                rocblasCall_symv_hemv<T>(
                    handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                    lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                    shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);
            }
            else
            {
                T const* const p_alpha = (scalars + 2);
                rocblas_stride const stride_alpha = 0;
                T const* const p_beta = (scalars + 1);
                rocblas_stride const stride_beta = 0;

                rocblas_int const incx = 1;
                rocblas_int const incy = 1;
                rocsolverCall_symv_hemv<T, rocblas_int, rocblas_stride>(
                    handle, uplo, n - 1 - j, p_alpha, stride_alpha, A, shiftA, j + 1, j + 1, lda,
                    strideA, A, shiftA, j + 1, j, lda, incx, strideA, p_beta, stride_beta, W,
                    shiftW, j + 1, j, ldw, incy, strideW, batch_count, mb, nb, work, workArr);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, n - j - 1, (tau + j), strideP, W,
                                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count);

            rocblasCall_dot<COMPLEX, T>(handle, n - 1 - j, W, shiftW + idx2D(j + 1, j, ldw), 1,
                                        strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                        batch_count, norms, work, workArr);

            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, n - 1 - j, norms,
                                    tau + j, strideP, A, shiftA + idx2D(j + 1, j, lda), strideA, W,
                                    shiftW + idx2D(j + 1, j, ldw), strideW);
        } // end for j
    }
    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;
            // update column j of A with reflector computed in step j-1
            if(use_coop)
            {
#if(0)
                template <typename T, typename I, typename Istride, typename UA>
                static __global__ void upper_stage1(
                    I const n, I const j, I const jw,

                    T* const scalars,

                    T* const W_, Istride const shiftW, I const ldw, Istride const strideW,

                    UA A_, Istride const shiftA, I const lda, Istride const strideA,

                    I const batch_count, I const mb, I const nb)
                {
#else
                rocblas_stride lshiftW = shiftW;
                rocblas_stride lshiftA = shiftA;
                void* args[] = {

                    (void*)&n,           (void*)&j,       (void*)&jw,

                    (void*)&scalars,

                    (void*)&W,           (void*)&lshiftW, (void*)&ldw, (void*)&strideW,

                    (void*)&A,           (void*)&lshiftA, (void*)&lda, (void*)&strideA,

                    (void*)&batch_count, (void*)&mb,      (void*)&nb};

                auto const nx = 32;
                auto const ny = 32;
                auto const lds_size = 64 * 1024;

                LAUNCH_CHECK(hipLaunchCooperativeKernel(
                    (void*)(upper_stage1<T, rocblas_int, rocblas_stride, U>), dim3(num_cu, 1, 1),
                    dim3(nx, ny, 1), args, lds_size, stream));

#endif
                }
                else
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - 1 - j, W,
                                                    shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                                    batch_count);

                    rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                        cast2constType<T>(scalars), 0, A,
                                        shiftA + idx2D(0, j + 1, lda), lda, strideA, W,
                                        shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                        cast2constType<T>(scalars + 2), 0, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, batch_count, workArr);

                    if(COMPLEX)
                    {
                        rocsolver_lacgv_template<T>(handle, n - 1 - j, W,
                                                    shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                                    batch_count);
                        rocsolver_lacgv_template<T>(handle, n - 1 - j, A,
                                                    shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                                    batch_count);
                    }

                    rocblasCall_gemv<T>(
                        handle, rocblas_operation_none, j + 1, n - 1 - j, cast2constType<T>(scalars),
                        0, W, shiftW + idx2D(0, jw + 1, ldw), ldw, strideW, A,
                        shiftA + idx2D(j, j + 1, lda), lda, strideA, cast2constType<T>(scalars + 2),
                        0, A, shiftA + idx2D(0, j, lda), 1, strideA, batch_count, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - 1 - j, A,
                                                    shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                                    batch_count);
                }
                // generate Householder reflector to work on column j
                rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), A,
                                         shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1),
                                         strideP, batch_count, work, norms);

                // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
                ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                        shiftA + idx2D(j - 1, j, lda), strideA, (E + j - 1), strideE);

                // compute/update column j of W
                rocblasCall_symv_hemv<T>(handle, uplo, j, (scalars + 2), 0, A, shiftA, lda, strideA,
                                         A, shiftA + idx2D(0, j, lda), 1, strideA, (scalars + 1), 0,
                                         W, shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count,
                                         work, workArr);

                if(use_coop)
                {
#if(0)
                    template <typename T, typename I, typename Istride, typename UA, typename UW>
                    static __device__ void upper_stage2(
                        I const n, I const j, I const jw,

                        T const* const scalars, T* const tau, Istride const strideP, T* const norms,

                        UW W_, Istride const shiftW, I const ldw, Istride const strideW,

                        UA A_, Istride const shiftA, I const lda, Istride const strideA,

                        I const batch_count, I const mb, I const nb)
#else
                rocblas_stride const lshiftA = shiftA;
                rocblas_stride const lshiftW = shiftW;

                void* args[]
                    = {(void*)&n,           (void*)&j,       (void*)&jw,

                       (void*)&scalars,     (void*)&tau,     (void*)&strideP, (void*)&norms,

                       (void*)&W,           (void*)&lshiftW, (void*)&ldw,     (void*)&strideW,

                       (void*)&A,           (void*)&lshiftA, (void*)&lda,     (void*)&strideA,

                       (void*)&batch_count, (void*)&mb,      (void*)&nb};

                LAUNCH_CHECK(hipLaunchCooperativeKernel(
                    (void*)(upper_stage2<T, rocblas_int, rocblas_stride, U, T*>),
                    dim3(num_cu, 1, 1), dim3(nx, ny, 1), args, lds_size, stream));

#endif
                }
                else
                {
                    rocblasCall_gemv<T>(
                        handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                        cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw), ldw,
                        strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                        cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(j + 1, jw, ldw), 1,
                        strideW, batch_count, workArr);

                    rocblasCall_gemv<T>(
                        handle, rocblas_operation_none, j, n - 1 - j, cast2constType<T>(scalars), 0,
                        A, shiftA + idx2D(0, j + 1, lda), lda, strideA, W,
                        shiftW + idx2D(j + 1, jw, ldw), 1, strideW, cast2constType<T>(scalars + 2),
                        0, W, shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, workArr);

                    rocblasCall_gemv<T>(
                        handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                        cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda), lda,
                        strideA, A, shiftA + idx2D(0, j, lda), 1, strideA,
                        cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(j + 1, jw, ldw), 1,
                        strideW, batch_count, workArr);

                    rocblasCall_gemv<T>(
                        handle, rocblas_operation_none, j, n - 1 - j, cast2constType<T>(scalars), 0,
                        W, shiftW + idx2D(0, jw + 1, ldw), ldw, strideW, W,
                        shiftW + idx2D(j + 1, jw, ldw), 1, strideW, cast2constType<T>(scalars + 2),
                        0, W, shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, workArr);

                    rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W,
                                        shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count);

                    rocblasCall_dot<COMPLEX, T>(handle, j, W, shiftW + idx2D(0, jw, ldw), 1,
                                                strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                                batch_count, norms, work, workArr);
                }

                // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
                //  available, we can use it instead of the scale_axpy kernel, if it provides
                //  better performance.)
                ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, j, norms,
                                        tau + j - 1, strideP, A, shiftA + idx2D(0, j, lda), strideA,
                                        W, shiftW + idx2D(0, jw, ldw), strideW);
            } // end for  j
        }

        rocblas_set_pointer_mode(handle, old_mode);
        return rocblas_status_success;
    }

    ROCSOLVER_END_NAMESPACE
