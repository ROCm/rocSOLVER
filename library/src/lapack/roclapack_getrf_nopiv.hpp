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

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I const nb = get_getrf_nopiv_blocksize<T>(n);

    // constants for rocblas functions calls
    T t_one = 1;
    T t_minone = -1;
    bool const use_rocblas = (batch_count == 1);

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

    I const min_mn = std::min(m, n);
    for(I j = 0; j < min_mn; j += nb)
    {
        // Factor diagonal and subdiagonal blocks
        I const jb = std::min(min_mn - j, nb); // number of columns in the block

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

        // -------------------------------------------------
        // Perform factorization of  jb by jb diagonal block
        // -------------------------------------------------
        rocsolver_getf2_nopiv_template<T>(handle, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                          iinfo, batch_count);

        // test for singular submatrix
        ROCSOLVER_LAUNCH_KERNEL(chk_singular<I>, gridReset, threads, 0, stream, iinfo, info, j,
                                batch_count);
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
                if(use_rocblas)
                {
                    rocblasCall_trsm(handle, side, uplo, trans, diag, mm, nn, alpha, A,
                                     shiftA + idx2D(j, j, lda), lda, strideA, A,
                                     shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                                     optim_mem, work1, work2, work3, work4);
                }
                else
                {
                    rocsolver_trsm_upper<BATCHED, STRIDED, T, I, U>(
                        handle, side, trans, diag, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                        strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                        optim_mem, work1, work2, work3, work4);
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
                if(use_rocblas)
                {
                    rocblasCall_trsm(handle, side, uplo, trans, diag, mm, nn, alpha, A,
                                     shiftA + idx2D(j, j, lda), lda, strideA, A,
                                     shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                                     optim_mem, work1, work2, work3, work4);
                }
                else
                {
                    rocsolver_trsm_lower<BATCHED, STRIDED, T, I, U>(
                        handle, side, trans, diag, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                        strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                        optim_mem, work1, work2, work3, work4);
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
                if(use_rocblas)
                {
                    rocblasCall_gemm(handle, trans_a, trans_b, mm, nn, kk, alpha, A,
                                     shiftA + idx2D(j + jb, j, lda), lda, strideA, A,
                                     shiftA + idx2D(j, j + jb, lda), lda, strideA, beta, A,
                                     shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count,
                                     work);
                }
                else
                {
                    rocsolver_gemm<BATCHED, STRIDED, T, U>(
                        handle, trans_a, trans_b, mm, nn, kk, alpha, A,
                        shiftA + idx2D(j + jb, j, lda), lda, strideA, A,
                        shiftA + idx2D(j, j + jb, lda), lda, strideA, beta, A,
                        shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, (T**)nullptr);
                }
            }
        }

    } // end for j

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
