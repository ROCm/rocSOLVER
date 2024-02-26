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

template <typename U>
ROCSOLVER_KERNEL void
    chk_singular(rocblas_int* iinfo, rocblas_int* info, int j, rocblas_int batch_count)
{
    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count && info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
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
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    rocblas_int const jb = GETRF_NOPIV_BLOCKSIZE(T);
    *size_iinfo = sizeof(rocblas_int) * jb * std::max(1, batch_count);
    *optim_mem = true;

    if(n <= jb)
    {
        return;
    }

    {
        // ------------------------
        // storage for rocblas TRSM
        // ------------------------

        size_t w1a = 0;
        size_t w1b = 0;
        size_t w2a = 0;
        size_t w2b = 0;
        size_t w3a = 0;
        size_t w3b = 0;
        size_t w4a = 0;
        size_t w4b = 0;

        // --------------------------------------
        //  L21 * U11 = A21 => L21 = A21 / U11
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_right;
            rocblas_operation const trans = rocblas_operation_none;

            rocblas_int const j = 0;
            rocblas_int const mm = (m - (j + jb));
            rocblas_int const nn = jb;

            rocblasCall_trsm_mem<BATCHED || STRIDED, T>(side, trans, mm, nn, batch_count, &w1a,
                                                        &w2a, &w3a, &w4a);
        }

        // --------------------------------------
        //  L11 * U12 = A12 => U12 = L11 \ A12
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_left;
            rocblas_operation const trans = rocblas_operation_none;

            rocblas_int const j = 0;
            rocblas_int const mm = jb;
            rocblas_int const nn = (n - (j + jb));

            rocblasCall_trsm_mem<BATCHED || STRIDED, T>(side, trans, mm, nn, batch_count, &w1b,
                                                        &w2b, &w3b, &w4b);
        }

        *size_work1 = max(*size_work1, max(w1a, w1b));
        *size_work2 = max(*size_work2, max(w2a, w2b));
        *size_work3 = max(*size_work3, max(w3a, w3b));
        *size_work4 = max(*size_work4, max(w4a, w4b));
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getrf_nopiv_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              void* work4,
                                              rocblas_int* iinfo,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("getrf_nopiv", "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
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
    rocblas_int const nb = GETRF_NOPIV_BLOCKSIZE(T);

    // constants for rocblas functions calls
    T t_one = 1;
    T t_minone = -1;
    S s_one = 1;
    S s_minone = -1;

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

    rocblas_int jb, j = 0;
    rocblas_int min_mn = std::min(m, n);
    for(rocblas_int j = 0; j < min_mn; j += nb)
    {
        // Factor diagonal and subdiagonal blocks
        jb = min(min_mn - j, nb); // number of columns in the block

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

        // -------------------------------------------------
        // Perform factorization of  jb by jb diagonal block
        // -------------------------------------------------
        rocsolver_getf2_nopiv_template<T>(handle, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                          iinfo, batch_count);

        // test for singular submatrix
        ROCSOLVER_LAUNCH_KERNEL(chk_singular<U>, gridReset, threads, 0, stream, iinfo, info, j,
                                batch_count);
        // --------------------------------------
        // (2) L21 * U11 = A21 => L21 = A21 / U11
        // --------------------------------------
        {
            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_operation const trans = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;

            rocblas_int const mm = (m - (j + jb));
            rocblas_int const nn = jb;

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                rocblasCall_trsm(handle, side, uplo, trans, diag, mm, nn, alpha, A,
                                 shiftA + idx2D(j, j, lda), lda, strideA, A,
                                 shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                                 optim_mem, work1, work2, work3, work4);
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

            rocblas_int const mm = jb;
            rocblas_int const nn = (n - (j + jb));

            const T* alpha = &t_one;

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(has_work)
            {
                rocblasCall_trsm(handle, side, uplo, trans, diag, mm, nn, alpha, A,
                                 shiftA + idx2D(j, j, lda), lda, strideA, A,
                                 shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                                 optim_mem, work1, work2, work3, work4);
            }
        }

        // -----------------------------------
        // (4a)  A22 = A22 - L21 * U12,   GEMM
        // -----------------------------------
        {
            rocblas_operation const trans_a = rocblas_operation_none;
            rocblas_operation const trans_b = rocblas_operation_none;
            rocblas_int const mm = (m - (j + jb));
            rocblas_int const nn = (n - (j + jb));
            rocblas_int const kk = jb;

            const T* alpha = &t_minone;
            const T* beta = &t_one;

            T** work = nullptr;

            bool const has_work = (mm >= 1) && (nn >= 1) && (kk >= 1);
            if(has_work)
            {
                rocblasCall_gemm(
                    handle, trans_a, trans_b, mm, nn, kk, alpha, A, shiftA + idx2D(j + jb, j, lda),
                    lda, strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, beta, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, work);
            }
        }

    } // end for j

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
