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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <cmath>

ROCSOLVER_BEGIN_NAMESPACE

/************** Kernels and device functions *******************/
/***************************************************************/

/** BDSQR_ESTIMATE device function computes an estimate of the smallest
    singular value of a n-by-n upper bidiagonal matrix given by D and E
    It also applies convergence test if conver = 1 **/
template <typename T>
__device__ T bdsqr_estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver)
{
    T smin = t2b ? std::abs(D[0]) : std::abs(D[n - 1]);
    T t = smin;

    rocblas_int je, jd;

    for(rocblas_int i = 1; i < n; ++i)
    {
        jd = t2b ? i : n - 1 - i;
        je = jd - t2b;
        if((std::abs(E[je]) <= tol * t) && conver)
        {
            E[je] = 0;
            smin = -1;
            break;
        }
        t = std::abs(D[jd]) * t / (t + std::abs(E[je]));
        smin = (t < smin) ? t : smin;
    }

    return smin;
}

/** BDSQR_T2BQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from top to bottom **/
template <typename T, typename S>
__device__ void bdsqr_t2bQRstep(const rocblas_int tid,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                S* D,
                                S* E,
                                T* V,
                                const rocblas_int ldv,
                                T* U,
                                const rocblas_int ldu,
                                T* C,
                                const rocblas_int ldc,
                                const S sh,
                                S* rots)
{
    S f, g, c, s, r;
    T temp1, temp2;
    rocblas_int nr = nv ? 2 * n : 0;

    if(tid == 0)
    {
        int sgn = (S(0) < D[0]) - (D[0] < S(0));
        if(D[0] == 0)
            f = 0;
        else
            f = (std::abs(D[0]) - sh) * (S(sgn) + sh / D[0]);
        g = E[0];

        for(rocblas_int k = 0; k < n - 1; ++k)
        {
            // first apply rotation by columns
            lartg(f, g, c, s, r);
            if(k > 0)
                E[k - 1] = r;
            f = c * D[k] - s * E[k];
            E[k] = c * E[k] + s * D[k];
            g = -s * D[k + 1];
            D[k + 1] = c * D[k + 1];
            // save rotations to update singular vectors
            if(nv)
            {
                rots[k] = c;
                rots[k + n] = -s;
            }

            // then apply rotation by rows
            lartg(f, g, c, s, r);
            D[k] = r;
            f = c * E[k] - s * D[k + 1];
            D[k + 1] = c * D[k + 1] + s * E[k];
            if(k < n - 2)
            {
                g = -s * E[k + 1];
                E[k + 1] = c * E[k + 1];
            }
            // save rotations to update singular vectors
            if(nu || nc)
            {
                rots[k + nr] = c;
                rots[k + nr + n] = -s;
            }
        }
        E[n - 2] = f;
    }
    __syncthreads();

    // update singular vectors
    if(nv)
    {
        // rotate from the left (forward direction)
        for(rocblas_int i = 0; i < n - 1; i++)
        {
            for(rocblas_int j = tid; j < nv; j += hipBlockDim_x)
            {
                temp1 = V[i + j * ldv];
                temp2 = V[i + 1 + j * ldv];
                c = rots[i];
                s = rots[i + n];
                V[i + j * ldv] = c * temp1 + s * temp2;
                V[i + 1 + j * ldv] = c * temp2 - s * temp1;
            }
        }
    }
    if(nu)
    {
        // rotate from the right (forward direction)
        for(rocblas_int j = 0; j < n - 1; j++)
        {
            for(rocblas_int i = tid; i < nu; i += hipBlockDim_x)
            {
                temp1 = U[i + j * ldu];
                temp2 = U[i + (j + 1) * ldu];
                c = rots[j + nr];
                s = rots[j + nr + n];
                U[i + j * ldu] = c * temp1 + s * temp2;
                U[i + (j + 1) * ldu] = c * temp2 - s * temp1;
            }
        }
    }
    if(nc)
    {
        // rotate from the left (forward direction)
        for(rocblas_int i = 0; i < n - 1; i++)
        {
            for(rocblas_int j = tid; j < nc; j += hipBlockDim_x)
            {
                temp1 = C[i + j * ldc];
                temp2 = C[i + 1 + j * ldc];
                c = rots[i + nr];
                s = rots[i + nr + n];
                C[i + j * ldc] = c * temp1 + s * temp2;
                C[i + 1 + j * ldc] = c * temp2 - s * temp1;
            }
        }
    }
}

/** BDSQR_B2TQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from bottom to top **/
template <typename T, typename S>
__device__ void bdsqr_b2tQRstep(const rocblas_int tid,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                S* D,
                                S* E,
                                T* V,
                                const rocblas_int ldv,
                                T* U,
                                const rocblas_int ldu,
                                T* C,
                                const rocblas_int ldc,
                                const S sh,
                                S* rots)
{
    S f, g, c, s, r;
    T temp1, temp2;
    rocblas_int nr = nv ? 2 * n : 0;

    if(tid == 0)
    {
        int sgn = (S(0) < D[n - 1]) - (D[n - 1] < S(0));
        if(D[n - 1] == 0)
            f = 0;
        else
            f = (std::abs(D[n - 1]) - sh) * (S(sgn) + sh / D[n - 1]);
        g = E[n - 2];

        for(rocblas_int k = n - 1; k > 0; --k)
        {
            // first apply rotation by rows
            lartg(f, g, c, s, r);
            if(k < n - 1)
                E[k] = r;
            f = c * D[k] - s * E[k - 1];
            E[k - 1] = c * E[k - 1] + s * D[k];
            g = -s * D[k - 1];
            D[k - 1] = c * D[k - 1];
            // save rotations to update singular vectors
            if(nu || nc)
            {
                rots[(k - 1) + nr] = c;
                rots[(k - 1) + nr + n] = s;
            }

            // then apply rotation by columns
            lartg(f, g, c, s, r);
            D[k] = r;
            f = c * E[k - 1] - s * D[k - 1];
            D[k - 1] = c * D[k - 1] + s * E[k - 1];
            if(k > 1)
            {
                g = -s * E[k - 2];
                E[k - 2] = c * E[k - 2];
            }
            // save rotations to update singular vectors
            if(nv)
            {
                rots[k - 1] = c;
                rots[(k - 1) + n] = s;
            }
        }
        E[0] = f;
    }
    __syncthreads();

    // update singular vectors
    if(nv)
    {
        // rotate from the left (backward direction)
        for(rocblas_int i = n - 1; i > 0; i--)
        {
            for(rocblas_int j = tid; j < nv; j += hipBlockDim_x)
            {
                temp1 = V[i + j * ldv];
                temp2 = V[i - 1 + j * ldv];
                c = rots[i - 1];
                s = rots[(i - 1) + n];
                V[i + j * ldv] = c * temp1 - s * temp2;
                V[i - 1 + j * ldv] = c * temp2 + s * temp1;
            }
        }
    }
    if(nu)
    {
        // rotate from the right (backward direction)
        for(rocblas_int j = n - 1; j > 0; j--)
        {
            for(rocblas_int i = tid; i < nu; i += hipBlockDim_x)
            {
                temp1 = U[i + j * ldu];
                temp2 = U[i + (j - 1) * ldu];
                c = rots[(j - 1) + nr];
                s = rots[(j - 1) + nr + n];
                U[i + j * ldu] = c * temp1 - s * temp2;
                U[i + (j - 1) * ldu] = c * temp2 + s * temp1;
            }
        }
    }
    if(nc)
    {
        // rotate from the left (backward direction)
        for(rocblas_int i = n - 1; i > 0; i--)
        {
            for(rocblas_int j = tid; j < nc; j += hipBlockDim_x)
            {
                temp1 = C[i + j * ldc];
                temp2 = C[i - 1 + j * ldc];
                c = rots[(i - 1) + nr];
                s = rots[(i - 1) + nr + n];
                C[i + j * ldc] = c * temp1 - s * temp2;
                C[i - 1 + j * ldc] = c * temp2 + s * temp1;
            }
        }
    }
}

/** BDSQR_KERNEL implements the main loop of the bdsqr algorithm
    to compute the SVD of an upper bidiagonal matrix given by D and E **/
template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_kernel(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   W1 VV,
                                   const rocblas_int shiftV,
                                   const rocblas_int ldv,
                                   const rocblas_stride strideV,
                                   W2 UU,
                                   const rocblas_int shiftU,
                                   const rocblas_int ldu,
                                   const rocblas_stride strideU,
                                   W3 CC,
                                   const rocblas_int shiftC,
                                   const rocblas_int ldc,
                                   const rocblas_stride strideC,
                                   rocblas_int* info,
                                   const rocblas_int maxiter,
                                   const S eps,
                                   const S sfm,
                                   const S tol,
                                   const S minshift,
                                   rocblas_int* splitsA,
                                   S* workA,
                                   const rocblas_int incW,
                                   const rocblas_stride strideW)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int sid = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;

    // if a NaN or Inf was detected in the input, return
    if(info[bid] != 0)
        return;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T *V, *U, *C;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    if(VV)
        V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
    if(UU)
        U = load_ptr_batch<T>(UU, bid, shiftU, strideU);
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    rocblas_int* splits = splitsA + bid * n;
    S* work = workA + bid * strideW;
    S* rots = work + 2;

    // shared variables
    __shared__ bool applyqr;
    __shared__ int t2b;
    __shared__ S smin, smax, sh, thresh;
    __shared__ rocblas_int i, k, start;
    __shared__ rocblas_int iter;

    // get convergence threshold
    if(tid == 0)
    {
        smin = work[0];
        thresh = work[1];
    }

    // main loop
    // iterate over each diagonal block
    while(2 * sid + 1 < n && splits[2 * sid + 1] > 0)
    {
        if(tid == 0)
        {
            // read diagonal block endpoints
            i = start = splits[2 * sid];
            k = splits[2 * sid + 1];

            // number of iterations (QR steps) applied to current block
            iter = 0;
        }
        __syncthreads();

        // iterate while diagonal block has not converged
        while(k > start && iter < maxiter)
        {
            if(tid == 0)
            {
                applyqr = false;

                // current block goes from i until k
                // determine shift for the QR step
                // (apply convergence test to find gaps)
                if(std::abs(D[i]) >= std::abs(D[k]))
                {
                    t2b = 1;
                    sh = std::abs(D[i]);
                }
                else
                {
                    t2b = 0;
                    sh = std::abs(D[k]);
                }

                // shift
                smin = bdsqr_estimate<S>(k - i + 1, D + i, E + i, t2b, tol, 1);
                // estimate of the largest singular value in the block
                smax = find_max_tridiag(i, k, D, E);

                // check for gaps, if none then continue
                if(smin >= 0)
                {
                    if(smin / smax <= minshift)
                        smin = 0; // shift set to zero if less than accepted value
                    else if(sh > 0)
                    {
                        if(smin * smin / sh / sh < eps)
                            smin = 0; // shift set to zero if negligible
                    }

                    applyqr = true;
                }
            }
            __syncthreads();

            // apply QR step
            if(applyqr)
            {
                if(tid == 0)
                    iter += k - i;

                if(t2b)
                    bdsqr_t2bQRstep(tid, k - i + 1, nv, nu, nc, D + i, E + i, V + i, ldv,
                                    U + i * ldu, ldu, C + i, ldc, smin, rots + incW * i);
                else
                    bdsqr_b2tQRstep(tid, k - i + 1, nv, nu, nc, D + i, E + i, V + i, ldv,
                                    U + i * ldu, ldu, C + i, ldc, smin, rots + incW * i);

                __syncthreads();
            }

            // update current block endpoints
            if(tid == 0)
            {
                while(k - 1 >= start && std::abs(E[k - 1]) < thresh)
                {
                    E[k - 1] = 0;
                    k--;
                }

                for(i = k - 1; i >= start; i--)
                {
                    if(std::abs(E[i]) < thresh)
                    {
                        E[i] = 0;
                        break;
                    }
                }
                i++;
            }
        }

        sid += hipGridDim_y;
    }
}

/** BDSQR_LOWER2UPPER kernel transforms a lower bidiagonal matrix given by D and E
    into an upper bidiagonal matrix via givens rotations **/
template <typename T, typename S, typename W1, typename W2>
ROCSOLVER_KERNEL void bdsqr_lower2upper(const rocblas_int n,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* DD,
                                        const rocblas_stride strideD,
                                        S* EE,
                                        const rocblas_stride strideE,
                                        W1 UU,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W2 CC,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        S* workA,
                                        const rocblas_stride strideW)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j;
    S f, g, c, s, r;
    T temp1, temp2;

    // if a NaN or Inf was detected in the input, return
    if(info[bid] != 0)
        return;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T *U, *C;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    if(UU)
        U = load_ptr_batch<T>(UU, bid, shiftU, strideU);
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* rots = workA + bid * strideW + 2;

    if(tid == 0)
    {
        f = D[0];
        g = E[0];
        for(i = 0; i < n - 1; ++i)
        {
            // apply rotations by rows
            lartg(f, g, c, s, r);
            D[i] = r;
            E[i] = -s * D[i + 1];
            f = c * D[i + 1];
            g = E[i + 1];

            // save rotation to update singular vectors
            if(nu || nc)
            {
                rots[i] = c;
                rots[i + n] = -s;
            }
        }
        D[n - 1] = f;
    }
    __syncthreads();

    // update singular vectors
    if(nu)
    {
        // rotate from the right (forward direction)
        for(j = 0; j < n - 1; j++)
        {
            for(i = tid; i < nu; i += hipBlockDim_x)
            {
                temp1 = U[i + j * ldu];
                temp2 = U[i + (j + 1) * ldu];
                c = rots[j];
                s = rots[j + n];
                U[i + j * ldu] = c * temp1 + s * temp2;
                U[i + (j + 1) * ldu] = c * temp2 - s * temp1;
            }
        }
    }
    if(nc)
    {
        // rotate from the left (forward direction)
        for(i = 0; i < n - 1; i++)
        {
            for(j = tid; j < nc; j += hipBlockDim_x)
            {
                temp1 = C[i + j * ldc];
                temp2 = C[(i + 1) + j * ldc];
                c = rots[i];
                s = rots[i + n];
                C[i + j * ldc] = c * temp1 + s * temp2;
                C[(i + 1) + j * ldc] = c * temp2 - s * temp1;
            }
        }
    }
}

/** BDSQR_INIT kernel checks if there are any NaNs or Infs in the input, calculates the
    convergence threshold and initial estimate for the smallest singular value, and splits
    the matrix into diagonal blocks. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_init(const rocblas_int n,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 S* EE,
                                 const rocblas_stride strideE,
                                 rocblas_int* info,
                                 const rocblas_int maxiter,
                                 const S sfm,
                                 const S tol,
                                 rocblas_int* splitsA,
                                 S* workA,
                                 const rocblas_int incW,
                                 const rocblas_stride strideW)
{
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* splits = splitsA + bid * n;
    S* work = workA + bid * strideW;

    bool found = false;
    rocblas_int ii = 0;
    rocblas_int start = 0;

    // calculate threshold for zeroing elements (convergence threshold)
    // direction
    int t2b = (D[0] >= D[n - 1]) ? 1 : 0;
    // estimate of the smallest singular value
    S smin = bdsqr_estimate<S>(n, D, E, t2b, tol, 0);
    // threshold
    S thresh = std::max(tol * smin / S(std::sqrt(n)), S(maxiter) * sfm);

    work[0] = smin;
    work[1] = thresh;

    // search for NaNs, Infs, and splits in the input
    for(rocblas_int i = 0; i < n - 1; i++)
    {
        if(2 * i + 1 < n)
        {
            splits[2 * i] = 0;
            splits[2 * i + 1] = 0;
            __threadfence();
        }

        if(!std::isfinite(D[i]) || !std::isfinite(E[i]))
            found = true;

        if(std::abs(E[i]) < thresh)
        {
            E[i] = 0;
            if(start < i)
            {
                // save diagonal block endpoints
                splits[2 * ii] = start;
                splits[2 * ii + 1] = i;
                ii++;
            }
            start = i + 1;
        }
    }

    if(!std::isfinite(D[n - 1]))
        found = true;

    if(start < n - 1)
    {
        // save diagonal block endpoints
        splits[2 * ii] = start;
        splits[2 * ii + 1] = n - 1;
    }

    // update output
    if(found)
    {
        for(rocblas_int i = 0; i < n - 1; i++)
        {
            D[i] = nan("");
            E[i] = nan("");
        }
        D[n - 1] = nan("");
        info[bid] = n;
    }
    else
        info[bid] = 0;
}

/**
 --------------------------------------------
 perform swaps to implement permutation vector
 the permutation vector will be restored to the
 identity permutation 0,1,2,...

 Note: this routine works in a thread block
 --------------------------------------------
 **/
template <typename T, typename I>
__device__ static void bdsqr_permute_swap(const I n,
                                          const I nv,
                                          T* V,
                                          const I ldv,
                                          const I nu,
                                          T* U,
                                          const I ldu,
                                          const I nc,
                                          T* C,
                                          const I ldc,
                                          I* map)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    bool const is_root_thread = (tid == 0);

    if(n <= 0)
    {
        return;
    };

    assert(map != nullptr);

    for(I i = 0; i < n; i++)
    {
        __syncthreads();

        while(map[i] != i)
        {
            auto const map_i = map[i];
            auto const map_ii = map[map[i]];

            __syncthreads();

            if(is_root_thread)
            {
                map[map_i] = map_i;
                map[i] = map_ii;
            };

            __syncthreads();

            // -----------
            // swap arrays
            // -----------

            auto const j_start = tid;
            auto const j_inc = nthreads;

            __syncthreads();

            if(nv > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nv; j += j_inc)
                {
                    swap(V[m + j * ((int64_t)ldv)], V[i + j * ((int64_t)ldv)]);
                };
            }
            __syncthreads();

            if(nu > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nu; j += j_inc)
                {
                    swap(U[j + m * ((int64_t)ldu)], U[j + i * ((int64_t)ldu)]);
                };
            }
            __syncthreads();

            if(nc > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nc; j += j_inc)
                {
                    swap(C[m + j * ((int64_t)ldc)], C[i + j * ((int64_t)ldc)]);
                };
            }
            __syncthreads();
        };
    };
}

/**
 * BDSQR_SORT sorts the singular values and vectors by
 * shell sort or selection sort if applicable.
 * **/

template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_sort(const rocblas_int n,
                                 const rocblas_int nv,
                                 const rocblas_int nu,
                                 const rocblas_int nc,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 S* EE,
                                 const rocblas_stride strideE,
                                 W1 VV,
                                 const rocblas_int shiftV,
                                 const rocblas_int ldv,
                                 const rocblas_stride strideV,
                                 W2 UU,
                                 const rocblas_int shiftU,
                                 const rocblas_int ldu,
                                 const rocblas_stride strideU,
                                 W3 CC,
                                 const rocblas_int shiftC,
                                 const rocblas_int ldc,
                                 const rocblas_stride strideC,
                                 rocblas_int* info,
                                 rocblas_int* splits_map)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j_start = tid;
    auto const j_inc = nthreads;

    rocblas_int bid = hipBlockIdx_y;

    // if a NaN or Inf was detected in the input, return
    if(info[bid] != 0)
        return;

    // local variables
    rocblas_int i, j, m;
    rocblas_int local_info = 0;

    // array pointers
    S* const D = (DD != nullptr) ? DD + bid * strideD : nullptr;
    S* const E = (EE != nullptr) ? EE + bid * strideE : nullptr;
    T* const V = (nv > 0) ? load_ptr_batch<T>(VV, bid, shiftV, strideV) : nullptr;
    T* const U = (nu > 0) ? load_ptr_batch<T>(UU, bid, shiftU, strideU) : nullptr;
    T* const C = (nc > 0) ? load_ptr_batch<T>(CC, bid, shiftC, strideC) : nullptr;

    rocblas_int* map = (splits_map == nullptr) ? nullptr : splits_map + bid * n;

    __syncthreads();

    // ensure all singular values converged and are positive
    for(rocblas_int i = 0; i < n; i++)
    {
        if(i < n - 1 && E[i] != 0)
            local_info++;

        if(D[i] < 0)
        {
            if(nv)
            {
                for(auto j = j_start; j < nv; j += j_inc)
                    V[i + j * ((int64_t)ldv)] = -V[i + j * ((int64_t)ldv)];
                __syncthreads();
            }

            if(tid == 0)
                D[i] = -D[i];
        }
    }

    if(local_info > 0)
    {
        if(tid == 0)
            info[bid] = local_info;
        return;
    }
    __syncthreads();

    // sort singular values & vectors
    bool const use_map = (map != nullptr);
    if(use_map)
    {
        bool const need_swap = (nv > 0) || (nu > 0) || (nc > 0);
        if(need_swap)
        {
            shell_sort_descending(n, D, map);
            __syncthreads();
            bdsqr_permute_swap(n, nv, V, ldv, nu, U, ldu, nc, C, ldc, map);
        }
        else
        {
            rocblas_int* const null_map = nullptr;
            shell_sort_descending(n, D, null_map);
        };

        __syncthreads();
    }
    else
    {
        S p;
        for(i = 0; i < n - 1; i++)
        {
            m = i;
            p = D[i];
            for(j = i + 1; j < n; j++)
            {
                if(D[j] > p)
                {
                    m = j;
                    p = D[j];
                }
            }
            __syncthreads();

            if(m != i)
            {
                if(tid == 0)
                {
                    D[m] = D[i];
                    D[i] = p;
                }

                if(nv)
                {
                    for(j = tid; j < nv; j += hipBlockDim_x)
                        swap(V[m + j * ldv], V[i + j * ldv]);
                    __syncthreads();
                }
                if(nu)
                {
                    for(j = tid; j < nu; j += hipBlockDim_x)
                        swap(U[j + m * ldu], U[j + i * ldu]);
                    __syncthreads();
                }
                if(nc)
                {
                    for(j = tid; j < nc; j += hipBlockDim_x)
                        swap(C[m + j * ldc], C[i + j * ldc]);
                    __syncthreads();
                }
            }
        }
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

template <typename T>
void rocsolver_bdsqr_getMemorySize(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   const rocblas_int batch_count,
                                   size_t* size_splits_map,
                                   size_t* size_work)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_splits_map = 0;
        *size_work = 0;
        return;
    }

    // size of split indices array
    *size_splits_map = sizeof(rocblas_int) * n * batch_count;

    // size of workspace
    rocblas_int incW = 0;
    if(nv)
        incW += 2;
    if(nu || nc)
        incW += 2;
    *size_work = sizeof(T) * (2 + incW * n) * batch_count;
}

template <typename S, typename W>
rocblas_status rocsolver_bdsqr_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        const rocblas_int ldv,
                                        const rocblas_int ldu,
                                        const rocblas_int ldc,
                                        S D,
                                        S E,
                                        W V,
                                        W U,
                                        W C,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((nv > 0 && ldv < n) || (nc > 0 && ldc < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (n && nv && !V) || (n && nu && !U) || (n && nc && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename W1, typename W2, typename W3>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        W1 V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        W2 U,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W3 C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* splits_map,
                                        S* work)
{
    ROCSOLVER_ENTER("bdsqr", "uplo:", uplo, "n:", n, "nv:", nv, "nu:", nu, "nc:", nc,
                    "shiftV:", shiftV, "ldv:", ldv, "shiftU:", shiftU, "ldu:", ldu,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set tolerance and max number of iterations:
    // machine precision (considering rounding strategy)
    S eps = get_epsilon<S>() / 2;
    // safest minimum value such that 1/sfm does not overflow
    S sfm = get_safemin<S>();
    // max number of iterations (QR steps) before declaring not convergence
    rocblas_int maxiter = 6 * n * n;
    // relative accuracy tolerance
    S tol = std::max(S(10.0), std::min(S(100.0), S(pow(eps, -0.125)))) * eps;
    //(minimum accepted shift to not ruin relative accuracy) / (max singular
    // value)
    S minshift = std::max(eps, tol / S(100)) / (n * tol);

    rocblas_int incW = 0;
    if(nv)
        incW += 2;
    if(nu || nc)
        incW += 2;
    rocblas_stride strideW = 2 + incW * n;

    // grid dimensions
    rocblas_int nuc_max = std::max(nu, nc);
    rocblas_int nvuc_max = std::max(nv, nuc_max);

    dim3 grid1(1, batch_count, 1);
    dim3 grid2(1, BDSQR_SPLIT_GROUPS, batch_count);
    dim3 threads1(1, 1, 1);
    dim3 threads2((nu || nc ? std::min(nuc_max, BS1) : 1), 1, 1);
    dim3 threads3((nv || nu || nc ? std::min(nvuc_max, BS1) : 1), 1, 1);

    // check for NaNs and Infs in input
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_init<T>), grid1, threads1, 0, stream, n, D, strideD, E, strideE,
                            info, maxiter, sfm, tol, splits_map, work, incW, strideW);

    if(n > 1)
    {
        // rotate to upper bidiagonal if necessary
        if(uplo == rocblas_fill_lower)
        {
            ROCSOLVER_LAUNCH_KERNEL((bdsqr_lower2upper<T>), grid1, threads2, 0, stream, n, nu, nc,
                                    D, strideD, E, strideE, U, shiftU, ldu, strideU, C, shiftC, ldc,
                                    strideC, info, work, strideW);
        }

        // main computation of SVD
        ROCSOLVER_LAUNCH_KERNEL((bdsqr_kernel<T>), grid2, threads3, 0, stream, n, nv, nu, nc, D,
                                strideD, E, strideE, V, shiftV, ldv, strideV, U, shiftU, ldu,
                                strideU, C, shiftC, ldc, strideC, info, maxiter, eps, sfm, tol,
                                minshift, splits_map, work, incW, strideW);
    }

    // sort the singular values and vectors
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_sort<T>), grid1, threads3, 0, stream, n, nv, nu, nc, D, strideD,
                            E, strideE, V, shiftV, ldv, strideV, U, shiftU, ldu, strideU, C, shiftC,
                            ldc, strideC, info, splits_map);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
