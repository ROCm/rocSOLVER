
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

#include "hip/amd_detail/amd_warp_sync_functions.h"
#include "hip/device_functions.h"
#include "hip/hip_common.h"
#include "hip/hip_cooperative_groups.h"

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#define PRINT_PROFILE
#ifdef PRINT_PROFILE
#define CLOCK64() clock64()
#else
#define CLOCK64() (0)
#endif

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

template <typename T, typename I>
__device__ T reduce_sum_shfl_wsize(I const wsize, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for(auto offset = wsize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down(val, offset);
        // g.sync();
    }
    return val; // note: only thread 0 will return full sum
}

static bool get_cooperative_launch(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeCooperativeLaunch;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_lds_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_warp_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeWarpSize;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_max_threads_per_block(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMaxThreadsPerBlock;
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

// -------------------------------------------------
// assume launch as dim3( num_cu,1,1), dim3(nx,ny,1)
// -------------------------------------------------
template <typename T, typename I>
static __device__ void Xscale_coop_body(I const n, T const alpha, T* const X, I const incx)
{
    {
        bool const has_work = (n >= 1) && (alpha != 1);
        if(!has_work)
        {
            return;
        }
    }

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    I const nblocks = (hipGridDim_x * hipGridDim_y * hipGridDim_z);
    I const nthreads_per_block = (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z);

    I const nx = nthreads_per_block;

    I const tix = (hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
                   + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y));

    I const bix = (hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x
                   + hipBlockIdx_z * (hipGridDim_x * hipGridDim_y));

    I const mb = ceil(n, nblocks);
    I const ix_start = bix * mb;
    I const ix_end = std::min(n, ix_start + mb);

    if(incx == 1)
    {
        for(I ix = (ix_start + tix); ix < ix_end; ix += nx)
        {
            if(alpha == 0)
            {
                X[ix] = 0;
            }
            else
            {
                X[ix] *= alpha;
            }
        }
    }
    else
    {
        for(I ix = (ix_start + tix); ix < ix_end; ix += nx)
        {
            if(alpha == 0)
            {
                X[ix * static_cast<int64_t>(incx)] = 0;
            }
            else
            {
                X[ix * static_cast<int64_t>(incx)] *= alpha;
            }
        }
    }
}

template <typename T, typename I>
static __device__ void Xgemv_sh(char const trans,
                                I const mm,
                                I const nn,

                                T const alpha,

                                T const* const A_sh_,
                                I const ldAsh,

                                T const* const X_,
                                I const incx,

                                T* const Y_,
                                I const incy)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    I nx = hipBlockDim_x;
    I ny = hipBlockDim_y;
    I tix = hipThreadIdx_x;
    I tiy = hipThreadIdx_y;

    I const myprow = hipBlockIdx_x;
    I const nprow = hipGridDim_x;

    {
        bool const has_work = (mm >= 1) && (nn >= 1) && (alpha != 0);
        if(!has_work)
        {
            return;
        }
    }

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    auto const mb = ceil(mm, nprow);

    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_no_transpose = (!is_transpose) && (!is_conj_transpose);

    I const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;
    I const warp_size = hipBlockDim_x;

    auto setup_nxny
        = [=](bool const is_no_transpose, auto const mm, auto const nn, auto& nx, auto& ny) {
              if(is_no_transpose)
              {
                  nx = (nn >= warp_size)       ? warp_size
                      : (nn >= warp_size / 2)  ? warp_size / 2
                      : (nn >= warp_size / 4)  ? warp_size / 4
                      : (nn >= warp_size / 8)  ? warp_size / 8
                      : (nn >= warp_size / 16) ? warp_size / 16
                      : (nn >= warp_size / 32) ? warp_size / 32
                                               : 1;
                  ny = nthreads / nx;
                  assert((nx * ny) == nthreads);
              }
              else
              {
                  auto const lny = nthreads / warp_size;
                  ny = (nn >= lny)       ? lny
                      : (nn >= lny / 2)  ? lny / 2
                      : (nn >= lny / 4)  ? lny / 4
                      : (nn >= lny / 8)  ? lny / 8
                      : (nn >= lny / 16) ? lny / 16
                      : (nn >= lny / 32) ? lny / 32
                                         : 1;

                  nx = nthreads / ny;
                  assert((nx * ny) == nthreads);
              }
          };

    I const wsize = std::min(warp_size, nx);

    I const tixy = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    //  ---------------------
    //  tixy = tix + tiy * nx
    //  ---------------------
    tix = (tixy % nx);
    tiy = (tixy - tix) / nx;
    bool const is_wave_rank0 = ((tixy % nx) == 0);

    auto A_sh = [=](auto i, auto j) {
        assert((0 <= i) && (i < mb));
        assert((0 <= j) && (j < nn));
        return (A_sh_[i + j * ldAsh]);
    };

    auto const len_X = (is_no_transpose) ? nn : mm;
    auto const len_Y = (is_no_transpose) ? mm : nn;

    auto Xvec = [=](auto i) -> T {
        assert((0 <= i) && (i < len_X));
        if(incx == 1)
        {
            return (X_[i]);
        }
        else
        {
            return (X_[i * static_cast<int64_t>(incx)]);
        }
    };

    auto Yvec = [=](auto i) -> T& {
        assert((0 <= i) && (i < len_Y));
        if(incy == 1)
        {
            return (Y_[i]);
        }
        else
        {
            return (Y_[i * static_cast<int64_t>(incy)]);
        }
    };

    I const ia_start = myprow * mb;
    I const ia_end = std::min(mm, (myprow + 1) * mb);

    if(is_no_transpose)
    {
        //  ------------------------------------------
        //  Y(1:mm) += alpha * A(1:mm, 1:nn) * X(1:nn)
        //  ------------------------------------------

        for(auto ia = (ia_start + tiy); ia < ia_end; ia += ny)
        {
            T y_i = 0;
            for(auto ja = (0 + tix); ja < nn; ja += nx)
            {
                T const xj = Xvec(ja);
                T const aij = A_sh(ia - ia_start, ja);
                y_i += aij * xj;
            }
            if(wsize > 1)
            {
                y_i = reduce_sum_shfl_wsize(wsize, y_i);
            }

            // ------------------------------
            // note: assume atomicAdd not required
            // ------------------------------
            if(is_wave_rank0 && (y_i != 0))
            {
                gatomicAdd(&(Yvec(ia)), alpha * y_i);
            }
        }
    }
    else
    {
        // ------------------------------------------------------
        // Y(1:nn) += alpha * tranpose( A(1:mm, 1:nn) ) * X(1:mm)
        // ------------------------------------------------------

        for(auto ja = (0 + tiy); ja < nn; ja += ny)
        {
            T y_j = 0;
            for(auto ia = (ia_start + tix); ia < ia_end; ia += nx)
            {
                T const xi = Xvec(ia);
                T const aij = A_sh(ia - ia_start, ja);
                if(is_complex && is_conj_transpose)
                {
                    y_j += xi * conj(aij);
                }
                else
                {
                    y_j += xi * aij;
                }
            }
            y_j = reduce_sum_shfl_wsize(wsize, y_j);
            if(is_wave_rank0)
            {
                gatomicAdd(&(Yvec(ja)), alpha * y_j);
            }
        }
    }
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
static __global__ void lower_stage2(I const n,
                                    I const j,

                                    T* const scalars,
                                    T* const tau,
                                    Istride const strideP,
                                    T* const norms,

                                    T* const W_,
                                    Istride const shiftW,
                                    I const ldw,
                                    Istride const strideW,

                                    UA A_,
                                    Istride const shiftA,
                                    I const lda,
                                    Istride const strideA,

                                    I const batch_count,
                                    I const lds_size)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    I tix = hipThreadIdx_x;
    I tiy = hipThreadIdx_y;
    I nx = hipBlockDim_x;
    I ny = hipBlockDim_y;
    I const tixy = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    I const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    // ----------------------------------------
    // assume launch dimension encode warp_size
    // ----------------------------------------
    I const warp_size = hipBlockDim_x;

    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    auto time_gemv_c1 = CLOCK64() * 0;
    auto time_gemv_n1 = CLOCK64() * 0;
    auto time_gemv_c2 = CLOCK64() * 0;
    auto time_gemv_n2 = CLOCK64() * 0;

    auto time_scale = CLOCK64() * 0;
    auto time_load = CLOCK64() * 0;
    auto tic = CLOCK64();

    auto cg_grid = cg::this_grid();

    I const mm = (n - j - 1);
    I const nn = j;

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };
    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    auto setup_nxny = [=](auto mb, auto nn, auto& nx, auto& ny) {
        auto const lny = (nn >= warp_size) ? warp_size
            : (nn >= warp_size / 2)        ? warp_size / 2
            : (nn >= warp_size / 4)        ? warp_size / 4
            : (nn >= warp_size / 8)        ? warp_size / 8
            : (nn >= warp_size / 16)       ? warp_size / 16
                                           : 1;
        if((nthreads % lny) == 0)
        {
            ny = lny;
            nx = nthreads / ny;
        }
    };

    // ------------------------------------
    // repartition configuration of threads
    // ------------------------------------
    bool const use_setup_nxny = true;
    if(use_setup_nxny)
    {
        setup_nxny(mm, nn, nx, ny);
        // ---------------------
        // tixy = tix + tiy * nx
        // ---------------------
        tix = (tixy % nx);
        tiy = (tixy - tix) / nx;
        assert(nx * ny == nthreads);
    }

    I const mb = ceil(mm, nprow);
    I const ldAWsh = is_even(mb) ? mb + 1 : mb;
    I const nb = nn;

    extern __shared__ double lmem[];

    size_t total_bytes = 0;
    std::byte* pfree = (std::byte*)&(lmem[0]);
    T* const W_sh_ = (T*)pfree;

    size_t const size_mat = sizeof(T) * ldAWsh * nn;

    pfree += size_mat;
    total_bytes += size_mat;

    T* const A_sh_ = (T*)pfree;
    pfree += size_mat;
    total_bytes += size_mat;

    {
        assert(total_bytes <= lds_size);
    }

    auto W_sh = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < mb));
        assert((0 <= j) && (j < nn));
        return (W_sh_[i + j * ldAWsh]);
    };

    auto A_sh = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < mb));
        assert((0 <= j) && (j < nn));
        return (A_sh_[i + j * ldAWsh]);
    };

    for(I bid = 0; bid < batch_count; bid++)
    {
        T* const __restrict__ W = load_ptr_batch(W_, bid, shiftW, strideW);
        T* const __restrict__ A = load_ptr_batch(A_, bid, shiftA, strideA);

        // --------------------------------
        // load data into LDS shared memory
        // --------------------------------

        I const iw = j + 1;
        I const jw = 0;
        I const ia = j + 1;
        I const ja = 0;

        auto Amat = [=](auto i, auto j) -> T& {
            assert((0 <= i) && (i < mm));
            assert((0 <= i) && (j < nn));
            assert((ia + i) < lda);

            return (A[idx2D(ia + i, ja + j, lda)]);
        };

        auto Wmat = [=](auto i, auto j) -> T& {
            assert((0 <= i) && (i < mm));
            assert((0 <= i) && (j < nn));
            assert((iw + i) < ldw);

            return (W[idx2D(iw + i, jw + j, ldw)]);
        };

        I const ii_start = (myprow)*mb;
        I const ii_end = std::min(mm, (myprow + 1) * mb);

        __syncthreads();
        tic = CLOCK64();

        bool const use_load_2D = false;
        if(use_load_2D)
        {
            for(I jj = (0 + tiy); jj < nn; jj += ny)
            {
                for(I ii = (ii_start + tix); ii < ii_end; ii += nx)
                {
                    A_sh((ii - ii_start), jj) = Amat(ii, jj);
                    W_sh((ii - ii_start), jj) = Wmat(ii, jj);
                }
            }
        }
        else
        {
            // ----------------------------
            // use one-dimensional indexing
            // ----------------------------

            I const ii_size = ii_end - ii_start;
            for(I ixy = (0 + tixy); ixy < (ii_size * nn); ixy += nthreads)
            {
                // ixy = ii + jj * ii_size
                I const jj = ixy / ii_size;
                I const ii = ixy % ii_size;

                A_sh(ii, jj) = Amat(ii + ii_start, jj);
                W_sh(ii, jj) = Wmat(ii + ii_start, jj);
            }
        }

        __syncthreads();
        time_load += CLOCK64() - tic;

        {
            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            char const trans = 'C';
            I const len_Y = nn;

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                tic = CLOCK64();
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(0, j, ldw), 1);
                cg_grid.sync();
                time_scale += CLOCK64() - tic;
            }

            if(alpha == 0)
            {
                Xscale_coop_body(len_Y, alpha, W + idx2D(0, j, ldw), 1);
            }
            else
            {
                tic = CLOCK64();
                Xgemv_sh<T, I>(trans, mm, nn, alpha,

                               W_sh_, ldAWsh,

                               A + idx2D(j + 1, j, lda), 1,

                               W + idx2D(0, j, ldw), 1);

                tic = CLOCK64();

                cg_grid.sync();
                time_gemv_c1 += CLOCK64() - tic;
            }
        }

        {
            char const trans = 'N';

            T const alpha = *scalars;
            T const beta = *(scalars + 2);

            I const len_Y = mm;

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                tic = CLOCK64();
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(j + 1, j, ldw), 1);

                // cg_grid.sync();
                // NOTE: grid sync is not needed since the part of vector
                // is updated by the same thread block
                __syncthreads();

                time_scale += CLOCK64() - tic;
            }

            if(alpha == 0)
            {
                Xscale_coop_body(len_Y, alpha, W + idx2D(j + 1, j, ldw), 1);
            }
            else
            {
                tic = CLOCK64();
                Xgemv_sh<T, I>(trans, mm, nn, alpha,

                               A_sh_, ldAWsh,

                               W + idx2D(0, j, ldw), 1,

                               W + idx2D(j + 1, j, ldw), 1);

                cg_grid.sync();
                time_gemv_n1 += CLOCK64() - tic;
            }
        }

        {
            char const trans = 'C';
            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            I const len_Y = nn;

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                tic = CLOCK64();

                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(0, j, ldw), 1);

                cg_grid.sync();

                time_scale += CLOCK64() - tic;
            }

            tic = CLOCK64();
            Xgemv_sh<T, I>(trans, mm, nn, alpha,

                           A_sh_, ldAWsh,

                           A + idx2D(j + 1, j, lda), 1,

                           W + idx2D(0, j, ldw), 1);

            cg_grid.sync();

            time_gemv_c2 += CLOCK64() - tic;
        }

        {
            char const trans = 'N';
            T const alpha = *scalars;
            T const beta = *(scalars + 2);

            I const len_Y = mm;

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                tic = CLOCK64();
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(j + 1, j, ldw), 1);

                // cg_grid.sync();
                // NOTE: grid sync is not needed since the part of vector
                // is updated by the same thread block
                __syncthreads();

                time_scale += CLOCK64() - tic;
            }

            tic = CLOCK64();
            Xgemv_sh<T, I>(trans, mm, nn, alpha,

                           W_sh_, ldAWsh,

                           W + idx2D(0, j, ldw), 1,

                           W + idx2D(j + 1, j, ldw), 1);

            cg_grid.sync();
            time_gemv_n2 += CLOCK64() - tic;
        }

    } // end for bid

#ifdef PRINT_PROFILE
    if(cg_grid.thread_rank() == 0)
    {
        printf("mm=%d, nn=%d, time_scale=%le, time_load=%le, time_gemv_c1=%le, time_gemv_c2=%le, "
               "time_gemv_n1=%le, time_gemv_n2=%le\n",
               (int)mm, (int)nn, (double)time_scale, (double)time_load, (double)time_gemv_c1,
               (double)time_gemv_c2, (double)time_gemv_n1, (double)time_gemv_n2);
    }
#endif
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
                                    I const lds_size)
{
    I const myprow = hipBlockIdx_x;
    I const mypcol = hipBlockIdx_y;
    I const nprow = hipGridDim_x;
    I const npcol = hipGridDim_y;

    auto cg_grid = cg::this_grid();

    I const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    I const mm = j;
    I const nn = n - 1 - j;

    I const mb = ceil(mm, nprow);
    I const ldAWsh = is_even(mb) ? mb + 1 : mb;

    I nx = hipBlockDim_x;
    I ny = hipBlockDim_y;
    I tix = hipThreadIdx_x;
    I tiy = hipThreadIdx_y;
    I const tixy = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    // ----------------------------------------
    // assume warp_size in launch configuration
    // ----------------------------------------
    I const warp_size = hipBlockDim_x;

    bool const use_setup_nxny = true;
    if(use_setup_nxny)
    {
        auto setup_nxny = [=](auto mb, auto nn, auto& nx, auto& ny) {
            auto const lny = (nn >= ny) ? ny
                : (nn >= ny / 2)        ? ny / 2
                : (nn >= ny / 4)        ? ny / 4
                : (nn >= ny / 8)        ? ny / 8
                : (nn >= ny / 16)       ? ny / 16
                : (nn >= ny / 32)       ? ny / 32
                                        : 1;
            if((nthreads % lny) == 0)
            {
                ny = lny;
                nx = nthreads / ny;
            }

            assert((nx * ny) == nthreads);
        };

        //  ----------------------
        //  tixy = tix + tiy * nx
        //  ----------------------
        tix = (tixy % nx);
        tiy = (tixy - tix) / nx;
    }

    extern __shared__ double lmem[];
    std::byte* pfree = (std::byte*)&(lmem[0]);

    size_t const mat_size = sizeof(T) * ldAWsh * nn;
    size_t total_bytes = 0;

    T* const A_sh_ = (T*)pfree;
    pfree += mat_size;
    total_bytes += mat_size;

    T* const W_sh_ = (T*)pfree;
    pfree += mat_size;
    total_bytes += mat_size;

    {
        size_t const ld_size = 64 * 1024;
        assert(total_bytes <= ld_size);
    }

    auto A_sh = [=](auto i, auto j) -> T& { return (A_sh_[i + j * ldAWsh]); };

    auto W_sh = [=](auto i, auto j) -> T& { return (W_sh_[i + j * ldAWsh]); };

    I const iiw = 0;
    I const jjw = jw + 1;
    I const ia = 0;
    I const ja = j + 1;

    for(I bid = 0; bid < batch_count; bid++)
    {
        T* const __restrict__ A = load_ptr_batch(A_, bid, shiftA, strideA);
        T* const __restrict__ W = load_ptr_batch(W_, bid, shiftW, strideW);

        auto Amat = [=](auto i, auto j) -> T& { return (A[idx2D(ia + i, ja + j, lda)]); };

        auto Wmat = [=](auto i, auto j) -> T& { return (W[idx2D(iiw + i, jjw + j, ldw)]); };

        // -----------------------
        // load into shared memory
        // -----------------------

        I const ia_start = myprow * mb;
        I const ia_end = std::min(mm, (myprow + 1) * mb);

        __syncthreads();

        bool const use_load_2D = false;
        if(use_load_2D)
        {
            for(I jj = (0 + tiy); jj < nn; jj += ny)
            {
                for(I ii = (ia_start + tix); ii < ia_end; ii += nx)
                {
                    A_sh((ii - ia_start), jj) = Amat(ii, jj);
                    W_sh((ii - ia_start), jj) = Wmat(ii, jj);
                }
            }
        }
        else
        {
            I const ia_size = ia_end - ia_start;

            for(I ixy = (0 + tixy); ixy < (ia_size * nn); ixy += nthreads)
            {
                // ------------------
                // ixy = ii + jj * ia_size
                // ------------------
                I const jj = ixy / ia_size;
                I const ii = ixy % ia_size;

                A_sh(ii, jj) = Amat(ii + ia_start, jj);
                W_sh(ii, jj) = Wmat(ii + ia_start, jj);
            }
        }

        __syncthreads();

        {
            char const trans = 'C';
            I const len_Y = nn;

            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(j + 1, jw, ldw), 1);

                cg_grid.sync();
            }

            Xgemv_sh(trans, mm, nn, alpha,

                     W_sh_, ldAWsh,

                     A + idx2D(0, j, lda), 1,

                     W + idx2D(j + 1, jw, ldw), 1);
            cg_grid.sync();
        }

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
            I const len_Y = mm;

            T const alpha = *scalars;
            T const beta = *(scalars + 2);

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(0, jw, ldw), 1);

                // cg_grid.sync();
                // NOTE: grid sync is not needed since the part of vector
                // is updated by the same thread block
                __syncthreads();
            }

            Xgemv_sh(trans, mm, nn, alpha,

                     A_sh_, ldAWsh,

                     W + idx2D(j + 1, jw, ldw), 1,

                     W + idx2D(0, jw, ldw), 1);
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
            I const len_Y = nn;

            T const alpha = *(scalars + 2);
            T const beta = *(scalars + 1);

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(j + 1, jw, ldw), 1);
                cg_grid.sync();
            }

            Xgemv_sh(trans, mm, nn, alpha,

                     A_sh_, ldAWsh,

                     A + idx2D(0, j, lda), 1,

                     W + idx2D(j + 1, jw, ldw), 1);

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
            I const len_Y = mm;

            T const alpha = *(scalars);
            T const beta = *(scalars + 2);

            bool const has_work = (len_Y >= 1) && (beta != 1);
            if(has_work)
            {
                Xscale_coop_body(len_Y, beta,

                                 W + idx2D(0, jw, ldw), 1);

                // cg_grid.sync();
                // NOTE: grid sync is not needed since the part of vector
                // is updated by the same thread block
                __syncthreads();
            }

            Xgemv_sh(trans, mm, nn, alpha,

                     W_sh_, ldAWsh,

                     W + idx2D(j + 1, jw, ldw), 1,

                     W + idx2D(0, jw, ldw), 1);

            cg_grid.sync();
        }

#endif

        {
            I const nn = j;
            T const* const p_alpha = (tau + j - 1);
            T const alpha = *(p_alpha + bid * strideP);

            Xscale_coop_body(nn, alpha,

                             W + idx2D(0, jw, ldw), 1);

            cg_grid.sync();
        }

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
    auto const num_cu = get_num_cu();
    auto const warp_size = get_warp_size();
    auto const max_threads_per_block = get_max_threads_per_block();
    rocblas_int const lds_size = get_lds_size();

    rocblas_int blocks = (batch_count - 1) / BS1 + 1;
    dim3 grid_b(blocks, 1);
    dim3 threads(BS1, 1, 1);
    blocks = (n - 1) / BS1 + 1;
    dim3 grid_n(blocks, batch_count);

    auto const nx = warp_size;
    auto const ny = max_threads_per_block / nx;

    bool const use_org = true;
    bool const use_coop = true;
    auto const mb = k;
    auto const nb = k;

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    auto need_lds_size = [=](auto mm, auto nn, auto sizeof_T) {
        auto const mb = ceil(mm, num_cu);

        auto ld = is_even(mb) ? mb + 1 : mb;
        return (sizeof_T * (ld * nn) * 2);
    };

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j, 0, ldw), ldw,
                                strideW, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
            }

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                     shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1, strideA,
                                     (tau + j), strideP, batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, (E + j), strideE);

            // compute/update column j of W
            {
                rocblasCall_symv_hemv<T>(
                    handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                    lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                    shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);
            }

            bool use_lower_stage2
                = get_cooperative_launch() && (need_lds_size(n - j - 1, j, sizeof(T)) <= lds_size);
#ifdef NDEBUG
#else
            printf("use_lower_stage2=%d, mm=%d, nn=%d\n", (int)use_lower_stage2, (int)(n - j - 1),
                   (int)j);
#endif

            if(use_lower_stage2)
            {
                rocblas_stride lshiftA = shiftA;
                rocblas_stride lshiftW = shiftW;

                void* args[]
                    = {(void*)&n,           (void*)&j,

                       (void*)&scalars,     (void*)&tau,     (void*)&strideP, (void*)&norms,

                       (void*)&W,           (void*)&lshiftW, (void*)&ldw,     (void*)&strideW,

                       (void*)&A,           (void*)&lshiftA, (void*)&lda,     (void*)&strideA,

                       (void*)&batch_count, (void*)&lds_size};

                auto const nx = warp_size;
                auto const ny = max_threads_per_block / nx;
                auto const lds_size = 64 * 1024;

                LAUNCH_CHECK(hipLaunchCooperativeKernel(
                    (void*)(lower_stage2<T, rocblas_int, rocblas_stride, U>), dim3(num_cu, 1, 1),
                    dim3(nx, ny, 1), args, lds_size, stream));
            }
            else
            {
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                    cast2constType<T>(scalars + 2), 0, W,
                                    shiftW + idx2D(j + 1, 0, ldw), ldw, strideW, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw),
                                    1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                    lda, strideA, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                    cast2constType<T>(scalars + 2), 0, W,
                                    shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                    cast2constType<T>(scalars + 2), 0, A,
                                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw),
                                    1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                    cast2constType<T>(scalars), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                    ldw, strideW, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                    cast2constType<T>(scalars + 2), 0, W,
                                    shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, workArr);
            }

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
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1), strideP,
                                     batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j - 1, j, lda), strideA, (E + j - 1), strideE);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(handle, uplo, j, (scalars + 2), 0, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, (scalars + 1), 0, W,
                                     shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, work,
                                     workArr);

            bool const use_upper_stage2
                = get_cooperative_launch() && (need_lds_size(j, n - 1 - j, sizeof(T)) <= lds_size);
#ifdef NDEBUG
#else
            printf("use_upper_stage2=%d, mm=%d, nn=%d\n", (int)use_upper_stage2, (int)j,
                   (int)n - 1 - j);
#endif

            if(use_upper_stage2)
            {
                rocblas_stride lshiftA = shiftA;
                rocblas_stride lshiftW = shiftW;

                void* args[]
                    = {(void*)&n,           (void*)&j,       (void*)&jw,

                       (void*)&scalars,     (void*)&tau,     (void*)&strideP, (void*)&norms,

                       (void*)&W,           (void*)&lshiftW, (void*)&ldw,     (void*)&strideW,

                       (void*)&A,           (void*)&lshiftA, (void*)&lda,     (void*)&strideA,

                       (void*)&batch_count, (void*)&lds_size};

                LAUNCH_CHECK(hipLaunchCooperativeKernel(
                    (void*)(upper_stage2<T, rocblas_int, rocblas_stride, U, T*>),
                    dim3(num_cu, 1, 1), dim3(nx, ny, 1), args, lds_size, stream));
            }
            else
            {
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                    cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw), ldw,
                    strideW, A, shiftA + idx2D(0, j, lda), 1, strideA, cast2constType<T>(scalars + 1),
                    0, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                    lda, strideA, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                    cast2constType<T>(scalars + 2), 0, W,
                                    shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA,
                    A, shiftA + idx2D(0, j, lda), 1, strideA, cast2constType<T>(scalars + 1), 0, W,
                    shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

                rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                    cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                    ldw, strideW, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                    cast2constType<T>(scalars + 2), 0, W,
                                    shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, workArr);

                rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W,
                                    shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count);
            }
            rocblasCall_dot<COMPLEX, T>(handle, j, W, shiftW + idx2D(0, jw, ldw), 1, strideW, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, batch_count, norms,
                                        work, workArr);

            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, j, norms,
                                    tau + j - 1, strideP, A, shiftA + idx2D(0, j, lda), strideA, W,
                                    shiftW + idx2D(0, jw, ldw), strideW);
        } // end for  j
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
