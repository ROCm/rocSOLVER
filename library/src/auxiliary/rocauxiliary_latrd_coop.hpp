
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

#ifndef ROCSOLVER_LATRD_COOP_H
#define ROCSOLVER_LATRD_COOP_H 1

#include <algorithm>
#include <cmath>
#include <complex>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

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
        bool const has_work = (n >= 1) && (alpha != 1);
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

    auto X = [=](auto iix, auto jjx) -> T& { return (X_[idx2D(iix, jjx, ldx)]); };

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

            for(auto i = (i_start + tix); i <= i_end; i += nx)
            {
                X(i, jx) = (is_alpha_zero) ? zero : alpha * X(i, jx);
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

            for(auto j = (j_start + tix); j <= j_end; j += nx)
            {
                X(ix, j) = (is_alpha_zero) ? zero : alpha * X(ix, j);
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

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;
    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_no_transpose = (!is_transpose) && (!is_conj_transpose);

    bool const is_column_Y = (incy == 1);
    bool const is_column_X = (incx == 1);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto Yvec = [=](auto i) -> T& {
        auto const iiy = (is_column_Y) ? iy + i : iy;
        auto const jjy = (is_column_Y) ? jy : jy + i;
        return (Y_[idx2D(iiy, jjy, ldy)]);
    };

    auto Xvec = [=](auto i) {
        auto const iix = (is_column_X) ? ix + i : ix;
        auto const jjx = (is_column_X) ? jx : jx + i;
        return (X_[idx2D(iix, jjx, ldx)]);
    };

    auto A = [=](auto i, auto j) { return (A_[idx2D(i, j, lda)]); };

    bool const need_atomic_update = (is_no_transpose) ? (npcol > 1) : (nprow > 1);

    I const itileA_start = first_tile(ia, mb, myprow, nprow);
    I const jtileA_start = first_tile(ja, nb, mypcol, npcol);
    I const itileA_end = first_tile(ia + m - 1, mb, myprow, nprow);
    I const jtileA_end = first_tile(ja + n - 1, nb, mypcol, npcol);
    I const itile_inc = nprow;
    I const jtile_inc = npcol;

    extern __shared__ double lmem[];
    T* const ytmp = (T*)&(lmem[0]);

    I const tixy = tix + tiy * nx;
    I const nxny = nx * ny;
    I const mbnb = (is_no_transpose) ? mb : nb;

    for(auto i = (0 + tixy); i < mbnb; i += nxny)
    {
        ytmp[i] = 0;
    }
    __syncthreads();

    if(is_no_transpose)
    {
        // -----------------
        // Y = alpha * A * X
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

                for(auto jja = (jstart + tiy); jja <= jend; jja += ny)
                {
                    auto const xj = Xvec((jja - ja));
                    for(auto iia = (istart + tix); iia <= iend; iia += nx)
                    {
                        auto const aij = A(iia, jja);
                        gatomicAdd(&(ytmp[(iia - istart)]), aij * xj);
                    }
                }
                __syncthreads();

                for(auto iia = (istart + tixy); iia <= iend; iia += nxny)
                {
                    auto const iiy = (iia - ia);
                    auto const ioff = (iia - istart);
                    auto const alpha_ytmp = alpha * ytmp[ioff];

                    if(need_atomic_update)
                    {
                        gatomicAdd(&(Yvec(iiy)), alpha_ytmp);
                    }
                    else
                    {
                        Yvec(iiy) += alpha_ytmp;
                    }
                    ytmp[ioff] = 0;
                }

                __syncthreads();

            } // end for itileA
        } // end for jtileA
    }
    else
    {
        // -----------------
        // Y = alpha * op(A) * X
        // -----------------

        // -----------------
        // Y = alpha * A * X
        // -----------------
        for(auto jtileA = jtileA_start; jtileA <= jtileA_end; jtileA += jtile_inc)
        {
            for(auto itileA = itileA_start; itileA <= itileA_end; itileA += itile_inc)
            {
                // ------------------------------
                // process tile "(itileA,jtileA)"
                // ------------------------------

                auto const jstart = std::max(ja, jtileA * nb);
                auto const jend = std::min(ja + n - 1, jstart + nb - 1);

                auto const istart = std::max(ia, itileA * mb);
                auto const iend = std::min(ia + m - 1, istart + mb - 1);

                for(auto jja = (jstart + tiy); jja <= jend; jja += ny)
                {
                    for(auto iia = (istart + tix); iia <= iend; iia += nx)
                    {
                        T const aij = A(iia, jja);
                        T const atji = (is_complex && is_conj_transpose) ? conj(aij) : aij;
                        T const xi = Xvec((iia - ia));

                        gatomicAdd(&(ytmp[(jja - jstart)]), atji * xi);
                    }
                }
                __syncthreads();

                for(auto jja = (jstart + tixy); jja <= jend; jja += nxny)
                {
                    auto const jjy = (jja - ja);
                    auto const joff = (jja - jstart);
                    auto const alpha_ytmp = alpha * ytmp[joff];

                    if(need_atomic_update)
                    {
                        gatomicAdd(&(Yvec(jjy)), alpha_ytmp);
                    }
                    else
                    {
                        Yvec(jjy) += alpha_ytmp;
                    }
                    ytmp[joff] = 0;
                }

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

    bool const use_rocblas = false;
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
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);

            if(use_rocblas)
            {
                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda),
                                    lda, strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda),
                                    1, strideA, batch_count, workArr);
            }
            else
            {
                auto const mm = n - j;
                auto const nn = j;
                auto const p_alpha = cast2constType<T>(scalars);
                rocblas_stride const stride_alpha = 0;
                auto const p_beta = cast2constType<T>(scalars + 2);
                rocblas_stride const stride_beta = 0;

                auto const Y = A;
                auto const shift_Y = shiftA;
                auto const iy = j;
                auto const jy = j;
                auto const ldy = lda;
                auto const incy = 1;
                auto const stride_Y = strideA;

                Xscale_batch_kernel<T, rocblas_int, rocblas_stride>
                    <<<dim3(num_cu, 1, 1), dim3(32, 32, 1), lds_size, stream>>>(
                        mm, p_beta, stride_beta, Y, shift_Y, iy, jy, ldy, incy, stride_Y,
                        batch_count, mb, nb);

                // rocblasCall_gemv<T>(handle, rocblas_operation_none, mm, nn,
                //            cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                //            strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                //            cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                //            strideA, batch_count, workArr);

                char const ctrans = 'N';
                auto const ia = j;
                auto const ja = 0;
                auto const iw = j;
                auto const jw = 0;
                auto const incw = ldw;

                Xgemv_batch_kernel<T, rocblas_int, rocblas_stride>
                    <<<dim3(num_cu, 1, 1), dim3(32, 32, 1), lds_size, stream>>>(
                        ctrans, mm, nn, p_alpha, stride_alpha, A, shiftA, ia, ja, lda, strideA, W,
                        shiftW, iw, jw, ldw, incw, strideW, Y, shift_Y, iy, jy, ldy, incy, stride_Y,
                        batch_count, mb, nb);
            }

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
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                     shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1, strideA,
                                     (tau + j), strideP, batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, (E + j), strideE);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(
                handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);

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
        }
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

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count);

            rocblasCall_dot<COMPLEX, T>(handle, j, W, shiftW + idx2D(0, jw, ldw), 1, strideW, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, batch_count, norms,
                                        work, workArr);

            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, j, norms,
                                    tau + j - 1, strideP, A, shiftA + idx2D(0, j, lda), strideA, W,
                                    shiftW + idx2D(0, jw, ldw), strideW);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
#endif
