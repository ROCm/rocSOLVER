/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
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

#include "rocblas.hpp"
#include "roclapack_getf2_nopiv.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"
#include <type_traits>

template <typename I>
static I partition_mn(I const m, I const n)
{
    auto const nb = std::min(m, n) / 2;
    return (nb);
}

static size_t get_lds_size()
{
    rocblas_int const default_lds_size = 64 * 1024;

    int lds_size = 0;
    int deviceId = 0;
    auto istat_device = hipGetDevice(&deviceId);
    if(istat_device != hipSuccess)
    {
        return (default_lds_size);
    };
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto istat_attr = hipDeviceGetAttribute(&lds_size, attr, deviceId);
    if(istat_attr != hipSuccess)
    {
        return (default_lds_size);
    };

    return (lds_size);
}

template <typename T>
static rocblas_int get_getrf_nopiv_blocksize(rocblas_int n)
{
    auto iceil = [](auto n, auto base) { return ((n - 1) / base + 1); };
    rocblas_int const nb_max = GETRF_NOPIV_BLOCKSIZE(T);
#if NDEBUG
#else
    {
        auto const lds_size = get_lds_size();
        bool const isok = (sizeof(T) * (nb_max * nb_max) <= lds_size);
        assert(isok);
    }
#endif
    rocblas_int const npass = iceil(n, nb_max);
    rocblas_int nb = iceil(n, npass);
    nb = std::max(1, std::min(nb_max, nb));
    return (nb);
}

template <typename I>
ROCSOLVER_KERNEL void chk_singular(I* iinfo, I* info, I j, I batch_count)
{
    I const id_start = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    I const id_inc = hipBlockDim_x * hipGridDim_x;

    for(I id = id_start; id < batch_count; id += id_inc)
    {
        if((info[id] == 0) && (iinfo[id] > 0))
        {
            info[id] = iinfo[id] + j;
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getrf_nopiv_rightlooking_getMemorySize(const rocblas_int m,
                                                      const rocblas_int n,
                                                      const rocblas_int batch_count,
                                                      size_t* size_work1,
                                                      size_t* size_work2,
                                                      size_t* size_work3,
                                                      size_t* size_work4,
                                                      size_t* size_iinfo,
                                                      bool* optim_mem)
{
    rocblas_int const nb = get_getrf_nopiv_blocksize<T>(n);

    *size_iinfo = sizeof(rocblas_int) * batch_count;
    *optim_mem = true;

    {
        auto is_even = [](auto n) { return ((n % 2) == 0); };
        // ------------------------
        // storage for rocblas TRSM
        // ------------------------

        size_t w1_max = 0;
        size_t w2_max = 0;
        size_t w3_max = 0;
        size_t w4_max = 0;

        size_t w1a = 0;
        size_t w2a = 0;
        size_t w3a = 0;
        size_t w4a = 0;

        rocblas_int const min_mn = std::min(m, n);

        // --------------------------------------
        //  L21 * U11 = A21 => L21 = A21 / U11
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_right;
            rocblas_operation const trans = rocblas_operation_none;

            {
                // -----------------------------------
                // avoid magic numbers in rocblas TRSM
                // by inflating (mm,nn) to odd numbers
                // -----------------------------------

                rocblas_int const mm = is_even(m) ? m + 1 : m;
                rocblas_int const nn = is_even(nb) ? nb + 1 : nb;

                {
                    rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &w1a,
                                                            &w2a, &w3a, &w4a, optim_mem);
                    w1_max = std::max(w1_max, w1a);
                    w2_max = std::max(w2_max, w2a);
                    w3_max = std::max(w3_max, w3a);
                    w4_max = std::max(w4_max, w4a);
                }

                {
                    bool const inblocked = true;
                    const rocblas_int inca = 1;
                    const rocblas_int incb = 1;

                    rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &w1a,
                                                            &w2a, &w3a, &w4a, optim_mem, inblocked,
                                                            inca, incb);
                };

                w1_max = std::max(w1_max, w1a);
                w2_max = std::max(w2_max, w2a);
                w3_max = std::max(w3_max, w3a);
                w4_max = std::max(w4_max, w4a);
            }

            for(rocblas_int j = 0; j < min_mn; j += nb)
            {
                rocblas_int const jb = std::min(min_mn - j, nb);
                rocblas_int const mm = m - (j + jb);
                rocblas_int const nn = jb;

                bool const has_work = (mm >= 1) && (nn >= 1);
                if(has_work)
                {
                    {
                        rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count,
                                                                &w1a, &w2a, &w3a, &w4a, optim_mem);

                        w1_max = std::max(w1_max, w1a);
                        w2_max = std::max(w2_max, w2a);
                        w3_max = std::max(w3_max, w3a);
                        w4_max = std::max(w4_max, w4a);
                    }

                    {
                        bool const inblocked = true;
                        const rocblas_int inca = 1;
                        const rocblas_int incb = 1;

                        rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count,
                                                                &w1a, &w2a, &w3a, &w4a, optim_mem,
                                                                inblocked, inca, incb);

                        w1_max = std::max(w1_max, w1a);
                        w2_max = std::max(w2_max, w2a);
                        w3_max = std::max(w3_max, w3a);
                        w4_max = std::max(w4_max, w4a);
                    }
                }
            }
        }

        // --------------------------------------
        //  L11 * U12 = A12 => U12 = L11 \ A12
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_left;
            rocblas_operation const trans = rocblas_operation_none;

            {
                rocblas_int const mm = is_even(nb) ? nb + 1 : nb;
                rocblas_int const nn = is_even(n) ? n + 1 : n;

                {
                    rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &w1a,
                                                            &w2a, &w3a, &w4a, optim_mem);

                    w1_max = std::max(w1_max, w1a);
                    w2_max = std::max(w2_max, w2a);
                    w3_max = std::max(w3_max, w3a);
                    w4_max = std::max(w4_max, w4a);
                }

                {
                    bool const inblocked = true;
                    const rocblas_int inca = 1;
                    const rocblas_int incb = 1;

                    rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &w1a,
                                                            &w2a, &w3a, &w4a, optim_mem, inblocked,
                                                            inca, incb);

                    w1_max = std::max(w1_max, w1a);
                    w2_max = std::max(w2_max, w2a);
                    w3_max = std::max(w3_max, w3a);
                    w4_max = std::max(w4_max, w4a);
                }
            }

            for(rocblas_int j = 0; j < min_mn; j += nb)
            {
                rocblas_int const jb = std::min(min_mn - j, nb);
                rocblas_int const mm = jb;
                rocblas_int const nn = n - (j + jb);

                bool const has_work = (mm >= 1) && (nn >= 1);
                if(has_work)
                {
                    {
                        rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count,
                                                                &w1a, &w2a, &w3a, &w4a, optim_mem);

                        w1_max = std::max(w1_max, w1a);
                        w2_max = std::max(w2_max, w2a);
                        w3_max = std::max(w3_max, w3a);
                        w4_max = std::max(w4_max, w4a);
                    }

                    {
                        rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count,
                                                                &w1a, &w2a, &w3a, &w4a, optim_mem);

                        bool const inblocked = true;
                        const rocblas_int inca = 1;
                        const rocblas_int incb = 1;
                        rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count,
                                                                &w1a, &w2a, &w3a, &w4a, optim_mem,
                                                                inblocked, inca, incb);

                        w1_max = std::max(w1_max, w1a);
                        w2_max = std::max(w2_max, w2a);
                        w3_max = std::max(w3_max, w3a);
                        w4_max = std::max(w4_max, w4a);
                    }
                }
            }
        }

        *size_work1 = std::max(*size_work1, w1_max);
        *size_work2 = std::max(*size_work2, w2_max);
        *size_work3 = std::max(*size_work3, w3_max);
        *size_work4 = std::max(*size_work4, w4_max);
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getrf_nopiv_recursive_getMemorySize(const rocblas_int m,
                                                   const rocblas_int n,
                                                   const rocblas_int batch_count,
                                                   size_t* size_work1,
                                                   size_t* size_work2,
                                                   size_t* size_work3,
                                                   size_t* size_work4,
                                                   size_t* size_iinfo,
                                                   bool* optim_mem)
{
    size_t lsize_work1 = 0;
    size_t lsize_work2 = 0;
    size_t lsize_work3 = 0;
    size_t lsize_work4 = 0;
    size_t lsize_iinfo = 0;

    auto const nsmall = GETRF_NOPIV_STOPPING_NB(T);
    bool const is_nsmall = (std::min(m, n) <= nsmall);

    if(is_nsmall)
    {
        rocsolver_getrf_nopiv_rightlooking_getMemorySize<BATCHED, STRIDED, T>(
            m, n, batch_count, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, &lsize_iinfo,
            optim_mem);

        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
    }
    else
    {
        auto const nb = partition_mn(m, n);
        auto const m1 = nb;
        auto const n1 = nb;

        auto const m2 = m - m1;
        auto const n2 = n - n1;

        // -----------------------------------------------
        // [A11  A12] = [L11   0 ] * [U11  U12]
        // [A21  A22]   [L21  L22]   [0    U22]
        //
        // (1) A11 = L11 * U11,   recursive factorization
        //
        // (2) L21 * U11 = A21,   triangular solve
        // or
        // (2) L21 = A21 / U11
        //
        // (3) L11 * U12 = A12,   triangular solve
        // or
        // (3) U12 = L11 \ A12
        //
        // (4) (A22 - L21 * U12) = L22 * U22
        // or
        // (4a) A22 <- A22 - L21 * U12
        // (4b) A22 = L22 * U22,  recursive factorization
        // -----------------------------------------------

        {
            // ----------------------------------------------
            // (1) A11 = L11 * U11,   recursive factorization
            // ----------------------------------------------
            auto const mm = m1;
            auto const nn = n1;

            rocsolver_getrf_nopiv_recursive_getMemorySize<BATCHED, STRIDED, T>(
                mm, nn, batch_count, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4,
                &lsize_iinfo, optim_mem);

            *size_work1 = std::max(*size_work1, lsize_work1);
            *size_work2 = std::max(*size_work2, lsize_work2);
            *size_work3 = std::max(*size_work3, lsize_work3);
            *size_work4 = std::max(*size_work4, lsize_work4);
            *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
        }

        {
            // -------------------
            // (2) L21 = A21 / U11
            // -------------------

            rocblas_side const side = rocblas_side_right;
            rocblas_operation const trans = rocblas_operation_none;

            auto const mm = m2;
            auto const nn = n1;

            rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &lsize_work1,
                                                    &lsize_work2, &lsize_work3, &lsize_work4,
                                                    optim_mem);

            *size_work1 = std::max(*size_work1, lsize_work1);
            *size_work2 = std::max(*size_work2, lsize_work2);
            *size_work3 = std::max(*size_work3, lsize_work3);
            *size_work4 = std::max(*size_work4, lsize_work4);
        }

        {
            // -------------------
            // (3) U12 = L11 \ A12
            // -------------------
            rocblas_side const side = rocblas_side_left;
            rocblas_operation const trans = rocblas_operation_none;

            auto const mm = m1;
            auto const nn = n2;
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(side, trans, mm, nn, batch_count, &lsize_work1,
                                                    &lsize_work2, &lsize_work3, &lsize_work4,
                                                    optim_mem);

            *size_work1 = std::max(*size_work1, lsize_work1);
            *size_work2 = std::max(*size_work2, lsize_work2);
            *size_work3 = std::max(*size_work3, lsize_work3);
            *size_work4 = std::max(*size_work4, lsize_work4);
        }

        {
            // ----------------------------------------------
            // (4) (A22 - L21 * U12) = L22 * U22
            // ----------------------------------------------
            auto const mm = m2;
            auto const nn = n2;

            rocsolver_getrf_nopiv_recursive_getMemorySize<BATCHED, STRIDED, T>(
                mm, nn, batch_count, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4,
                &lsize_iinfo, optim_mem);

            *size_work1 = std::max(*size_work1, lsize_work1);
            *size_work2 = std::max(*size_work2, lsize_work2);
            *size_work3 = std::max(*size_work3, lsize_work3);
            *size_work4 = std::max(*size_work4, lsize_work4);
            *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getrf_nopiv_getMemorySize(const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_work1,
                                         size_t* size_work2,
                                         size_t* size_work3,
                                         size_t* size_work4,
                                         size_t* size_iinfo,
                                         bool* optim_mem)
{
    assert(size_work1 != nullptr);
    assert(size_work2 != nullptr);
    assert(size_work3 != nullptr);
    assert(size_work4 != nullptr);
    assert(size_iinfo != nullptr);
    assert(optim_mem != nullptr);

    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_iinfo = 0;
    *optim_mem = true;
    // if quick return no need of workspace
    if((m == 0) || (n == 0) || (batch_count == 0))
    {
        return;
    }

    auto const nsmall = GETRF_NOPIV_STOPPING_NB(T);
    bool const use_recursive = true;
    if(use_recursive)
    {
        rocsolver_getrf_nopiv_recursive_getMemorySize<BATCHED, STRIDED, T>(
            m, n, batch_count, size_work1, size_work2, size_work3, size_work4, size_iinfo, optim_mem);
    }
    else
    {
        rocsolver_getrf_nopiv_rightlooking_getMemorySize<BATCHED, STRIDED, T>(
            m, n, batch_count, size_work1, size_work2, size_work3, size_work4, size_iinfo, optim_mem);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_getrf_nopiv_rightlooking_template(rocblas_handle handle,
                                                           const I m,
                                                           const I n,
                                                           U A,
                                                           const I shiftA,
                                                           const I lda,
                                                           const rocblas_stride strideA,
                                                           I* info,
                                                           const I batch_count,
                                                           void* work1,
                                                           void* work2,
                                                           void* work3,
                                                           void* work4,
                                                           I* iinfo,
                                                           bool optim_mem)
{
    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I const nb = get_getrf_nopiv_blocksize<T>(n);

    // constants for rocblas functions calls
    T t_one = 1;
    T t_minone = -1;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const blocksReset = std::max(1, (batch_count - 1) / BS1 + 1);
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // -------------------------------------------
    // [ L11      ] * [ U11   U12 ] = [A11   A12]
    // [ L21  L22 ]   [       U22 ]   [A21   A22]
    //
    // (1) L11 * U11 = A11 => factorize diagonal block
    // (2) L21 * U11 = A21 => L21 = A21 / U11
    // (3) L11 * U12 = A12 => U12 = L11 \ A12
    //
    // (4) L21 * U12 + L22 * U22 = A22
    // (4a)  A22 = A22 - L21 * U12,   GEMM
    // (4b)  L22 * U22 = A22,  factorize remaining using tail recursion
    // -------------------------------------------

    I const min_mn = std::min(m, n);
    for(I j = 0; j < min_mn; j += nb)
    {
        // Factor diagonal and subdiagonal blocks
        I const jb = std::min(min_mn - j, nb); // number of columns in the block

        {
            // -----------------------------------------------
            // (1) L11 * U11 = A11 => factorize diagonal block
            // -----------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

            // -------------------------------------------------
            // Perform factorization of  jb by jb diagonal block
            // -------------------------------------------------
            auto const istat = rocsolver_getf2_nopiv_template<T>(
                handle, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iinfo, batch_count);

            // ---------------------------
            // test for singular submatrix
            // ---------------------------
            ROCSOLVER_LAUNCH_KERNEL(chk_singular<I>, gridReset, threads, 0, stream, iinfo, info, j,
                                    batch_count);
            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }
        // --------------------------------------
        // (2) L21 * U11 = A21 => L21 = A21 / U11
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_operation const trans = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;

            I const mm = (m - (j + jb));
            I const nn = jb;

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T, I, U>(
                    handle, side, trans, diag, mm, nn, A, shiftA + idx2D(j, j, lda), lda, strideA,
                    A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count, optim_mem, work1,
                    work2, work3, work4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        // --------------------------------------
        // (3) L11 * U12 = A12 => U12 = L11 \ A12
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_left;
            rocblas_fill const uplo = rocblas_fill_lower;
            rocblas_operation const trans = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_unit;

            I const mm = jb;
            I const nn = (n - (j + jb));

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T, I, U>(
                    handle, side, trans, diag, mm, nn, A, shiftA + idx2D(j, j, lda), lda, strideA,
                    A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count, optim_mem, work1,
                    work2, work3, work4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        // -----------------------------------
        // (4a)  A22 = A22 - L21 * U12,   GEMM
        // -----------------------------------
        {
            rocblas_operation const trans_a = rocblas_operation_none;
            rocblas_operation const trans_b = rocblas_operation_none;

            I const mm = (m - (j + jb));
            I const nn = (n - (j + jb));
            I const kk = jb;

            const T* alpha = &t_minone;
            const T* beta = &t_one;

            T** work = nullptr;

            bool const has_work = (mm >= 1) && (nn >= 1) && (kk >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_gemm<BATCHED, STRIDED, T, U>(
                    handle, trans_a, trans_b, mm, nn, kk, alpha, A, shiftA + idx2D(j + jb, j, lda),
                    lda, strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, beta, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, (T**)nullptr);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

    } // end for j

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_getrf_nopiv_recursive_template(rocblas_handle handle,
                                                        const I m,
                                                        const I n,
                                                        U A,
                                                        const I shiftA,
                                                        const I lda,
                                                        const rocblas_stride strideA,
                                                        I* info,
                                                        const I batch_count,
                                                        void* work1,
                                                        void* work2,
                                                        void* work3,
                                                        void* work4,
                                                        I* iinfo,
                                                        bool optim_mem,
                                                        I row_offset)
{
    // constants for rocblas functions calls
    T t_one = 1;
    T t_minone = -1;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const blocksReset = std::max(1, (batch_count - 1) / BS1 + 1);
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // -------------------------------------------
    // [ L11      ] * [ U11   U12 ] = [A11   A12]
    // [ L21  L22 ]   [       U22 ]   [A21   A22]
    //
    // (1) L11 * U11 = A11 => factorize diagonal block
    // (2) L21 * U11 = A21 => L21 = A21 / U11
    // (3) L11 * U12 = A12 => U12 = L11 \ A12
    //
    // (4) L21 * U12 + L22 * U22 = A22
    // (4a)  A22 = A22 - L21 * U12,   GEMM
    // (4b)  L22 * U22 = A22,  factorize remaining using tail recursion
    // -------------------------------------------

    I const min_mn = std::min(m, n);

    I const n_stopping_nb = GETRF_NOPIV_STOPPING_NB(T);
    bool const is_small = (min_mn <= n_stopping_nb);
    if(is_small)
    {
        // -----------------------------------------------
        // (1) L11 * U11 = A11 => factorize diagonal block
        // -----------------------------------------------
        if(iinfo != nullptr)
        {
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
        }

        // -------------------------------------------------
        // Perform factorization of  jb by jb diagonal block
        // -------------------------------------------------
        I const mm = m;
        I const nn = n;
        I const j = 0;

        rocblas_status const istat
            = rocsolver_getrf_nopiv_rightlooking_template<BATCHED, STRIDED, T, I, U>(
                handle, mm, nn, A, shiftA, lda, strideA, info, batch_count, work1, work2, work3,
                work4, (iinfo != nullptr) ? iinfo : info, optim_mem);
        // ---------------------------
        // test for singular submatrix
        // ---------------------------

        if(iinfo != nullptr)
        {
            ROCSOLVER_LAUNCH_KERNEL(chk_singular<I>, gridReset, threads, 0, stream, iinfo, info,
                                    row_offset + j, batch_count);
        }
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }
    else
    {
        I const jb = partition_mn(m, n);
        I const m1 = jb;
        I const n1 = jb;
        I const m2 = m - m1;
        I const n2 = n - n1;

        auto const A11_offset = idx2D(0, 0, lda);
        auto const A21_offset = idx2D(m1, 0, lda);
        auto const A12_offset = idx2D(0, n1, lda);
        auto const A22_offset = idx2D(m1, n1, lda);

        auto const L11_offset = A11_offset;
        auto const L21_offset = A21_offset;
        auto const L22_offset = A22_offset;

        auto const U11_offset = A11_offset;
        auto const U12_offset = A12_offset;
        auto const U22_offset = A22_offset;

        {
            // -----------------------------------------------
            // (1) L11 * U11 = A11 => factorize diagonal block
            // -----------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

            // -------------------------------------------------
            // Perform factorization of  jb by jb diagonal block
            // -------------------------------------------------

            I const mm = m1;
            I const nn = n1;
            rocblas_status istat = rocblas_status_success;

            bool const use_rightlooking = true;
            if(use_rightlooking)
            {
                istat = rocsolver_getrf_nopiv_rightlooking_template<BATCHED, STRIDED, T, I, U>(
                    handle, mm, nn, A, shiftA + A11_offset, lda, strideA, info, batch_count, work1,
                    work2, work3, work4, iinfo, optim_mem);
            }
            else
            {
                istat = rocsolver_getrf_nopiv_recursive_template<BATCHED, STRIDED, T, I, U>(
                    handle, mm, nn, A, shiftA + A11_offset, lda, strideA, info, batch_count, work1,
                    work2, work3, work4, iinfo, optim_mem, row_offset);
            }

            // ---------------------------
            // test for singular submatrix
            // ---------------------------
            I const j = 0;
            ROCSOLVER_LAUNCH_KERNEL(chk_singular<I>, gridReset, threads, 0, stream, iinfo, info,
                                    row_offset + j, batch_count);

            assert(istat == rocblas_status_success);

            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }

        {
            // --------------------------------------
            // (2) L21 * U11 = A21 => L21 = A21 / U11
            // --------------------------------------
            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_operation const trans = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;

            I const mm = m2;
            I const nn = n1;

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T, I, U>(
                    handle, side, trans, diag, mm, nn, A, shiftA + U11_offset, lda, strideA, A,
                    shiftA + A21_offset, lda, strideA, batch_count, optim_mem, work1, work2, work3,
                    work4);

                assert(istat == rocblas_status_success);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        {
            // --------------------------------------
            // (3) L11 * U12 = A12 => U12 = L11 \ A12
            // --------------------------------------
            rocblas_side const side = rocblas_side_left;
            rocblas_fill const uplo = rocblas_fill_lower;
            rocblas_operation const trans = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_unit;

            I const mm = m1;
            I const nn = n2;

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T, I, U>(
                    handle, side, trans, diag, mm, nn, A, shiftA + L11_offset, lda, strideA, A,
                    shiftA + A12_offset, lda, strideA, batch_count, optim_mem, work1, work2, work3,
                    work4);

                assert(istat == rocblas_status_success);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        {
            // -----------------------------------
            // (4a)  A22 = A22 - L21 * U12,   GEMM
            // -----------------------------------
            rocblas_operation const trans_a = rocblas_operation_none;
            rocblas_operation const trans_b = rocblas_operation_none;

            I const mm = m2;
            I const nn = n2;
            I const kk = n1;

            const T* alpha = &t_minone;
            const T* beta = &t_one;

            T** work = nullptr;

            bool const has_work = (mm >= 1) && (nn >= 1) && (kk >= 1);
            if(has_work)
            {
                auto const istat = rocsolver_gemm<BATCHED, STRIDED, T, U>(
                    handle, trans_a, trans_b, mm, nn, kk, alpha, A, shiftA + L21_offset, lda,
                    strideA, A, shiftA + U12_offset, lda, strideA, beta, A, shiftA + A22_offset,
                    lda, strideA, batch_count, (T**)nullptr);

                assert(istat == rocblas_status_success);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        {
            // -----------------------------------------------
            // (4b) L22 * U22 = A22 => factorize diagonal block
            // -----------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

            // -------------------------------------------------
            // Perform factorization of  m2 by n2 diagonal block
            // -------------------------------------------------

            I const mm = m2;
            I const nn = n2;
            I const j = m1;
            bool const use_rightlooking = true;
            rocblas_status istat = rocblas_status_success;
            if(use_rightlooking)
            {
                istat = rocsolver_getrf_nopiv_rightlooking_template<BATCHED, STRIDED, T, I, U>(
                    handle, mm, nn, A, shiftA + A22_offset, lda, strideA, info, batch_count, work1,
                    work2, work3, work4, iinfo, optim_mem);
            }
            else
            {
                istat = rocsolver_getrf_nopiv_recursive_template<BATCHED, STRIDED, T, I, U>(
                    handle, mm, nn, A, shiftA + A22_offset, lda, strideA, info, batch_count, work1,
                    work2, work3, work4, iinfo, optim_mem, row_offset + j);
            }

            // ---------------------------
            // test for singular submatrix
            // ---------------------------

            ROCSOLVER_LAUNCH_KERNEL(chk_singular<I>, gridReset, threads, 0, stream, iinfo, info,
                                    row_offset + j, batch_count);

            assert(istat == rocblas_status_success);

            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_getrf_nopiv_template(rocblas_handle handle,
                                              const I m,
                                              const I n,
                                              U A,
                                              const I shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              I* info,
                                              const I batch_count,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              void* work4,
                                              I* iinfo,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("getrf_nopiv", "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const blocksReset = std::max(1, (batch_count - 1) / BS1 + 1);
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;
    bool const use_recursive = true;
    if(use_recursive)
    {
        I const row_offset = 0;
        istat = rocsolver_getrf_nopiv_recursive_template<BATCHED, STRIDED, T, I, U>(
            handle, m, n, A, shiftA, lda, strideA, info, batch_count, work1, work2, work3, work4,
            iinfo, optim_mem, row_offset);
    }
    else
    {
        istat = rocsolver_getrf_nopiv_rightlooking_template<BATCHED, STRIDED, T, I, U>(
            handle, m, n, A, shiftA, lda, strideA, info, batch_count, work1, work2, work3, work4,
            iinfo, optim_mem);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return istat;
}
