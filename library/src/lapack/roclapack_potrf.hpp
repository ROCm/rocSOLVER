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
#include "roclapack_potf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

bool constexpr use_recursive = true;

static rocblas_int get_lds_size()
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

template <typename U>
ROCSOLVER_KERNEL void
    chk_positive(rocblas_int* iinfo, rocblas_int* info, int j, rocblas_int batch_count)
{
    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count && info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_potrf_rightlooking_getMemorySize(const rocblas_int n,
                                                const rocblas_fill uplo,
                                                const rocblas_int batch_count,
                                                size_t* size_scalars,
                                                size_t* size_work1,
                                                size_t* size_work2,
                                                size_t* size_work3,
                                                size_t* size_work4,
                                                size_t* size_pivots,
                                                size_t* size_iinfo,
                                                bool* optim_mem)
{
    *size_scalars = 0;
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_pivots = 0;
    *size_iinfo = 0;
    *optim_mem = true;
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    rocblas_int nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // requirements for calling a single POTF2
        rocsolver_potf2_getMemorySize<T>(n, batch_count, size_scalars, size_work1, size_pivots);
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iinfo = 0;
        *optim_mem = true;
    }
    else
    {
        rocblas_int jb = nb;
        size_t s1, s2;

        // size to store info about positiveness of each subblock
        *size_iinfo = sizeof(rocblas_int) * batch_count;

        // requirements for calling POTF2 for the subblocks
        rocsolver_potf2_getMemorySize<T>(jb, batch_count, size_scalars, &s1, size_pivots);

        // extra requirements for calling TRSM
        if(uplo == rocblas_fill_upper)
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb, batch_count,
                &s2, size_work2, size_work3, size_work4, optim_mem);
        }
        else
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_right, rocblas_operation_conjugate_transpose, n - jb, jb, batch_count,
                &s2, size_work2, size_work3, size_work4, optim_mem);
        }

        *size_work1 = std::max(s1, s2);
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_potrf_recursive_getMemorySize(const rocblas_int n,
                                             const rocblas_fill uplo,
                                             const rocblas_int batch_count,
                                             size_t* size_scalars,
                                             size_t* size_work1,
                                             size_t* size_work2,
                                             size_t* size_work3,
                                             size_t* size_work4,
                                             size_t* size_pivots,
                                             size_t* size_iinfo,
                                             bool* optim_mem)
{
    size_t lsize_scalars = 0;
    size_t lsize_work1 = 0;
    size_t lsize_work2 = 0;
    size_t lsize_work3 = 0;
    size_t lsize_work4 = 0;
    size_t lsize_pivots = 0;
    size_t lsize_iinfo = 0;

    auto const nsmall = POTRF_BLOCKSIZE(T);
    bool const is_nsmall = (n <= nsmall);
    if(is_nsmall)
    {
        rocsolver_potf2_getMemorySize<T>(n, batch_count, &lsize_scalars, &lsize_work1, &lsize_pivots);

        *size_scalars = std::max(*size_scalars, lsize_scalars);
        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_pivots = std::max(*size_pivots, lsize_pivots);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);

        return;
    }

    auto const n2 = n / 2;
    auto const n1 = n - n2;

    if(uplo == rocblas_fill_upper)
    {
        // -----------------------------------------------
        // (2) U11' * U12 = A12,     TRSM triangular solve
        // or  U12 = A12/U11'
        // -----------------------------------------------
        auto const mm = n1;
        auto const nn = n2;
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(
            rocblas_side_left, rocblas_operation_conjugate_transpose, mm, nn, batch_count,
            &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
    }
    else
    {
        // ------------------------------------------
        // (2)  L21 * L11' = A21 or
        //      L21 = A21 / L11'     TRSM triangular solve
        // ------------------------------------------
        auto const mm = n2;
        auto const nn = n1;
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(
            rocblas_side_right, rocblas_operation_conjugate_transpose, mm, nn, batch_count,
            &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, optim_mem);

        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
    }

    {
        auto const nn = n1;
        rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T>(
            nn, uplo, batch_count, &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3,
            &lsize_work4, &lsize_pivots, &lsize_iinfo, optim_mem);

        *size_scalars = std::max(*size_scalars, lsize_scalars);
        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_pivots = std::max(*size_pivots, lsize_pivots);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
    }

    {
        auto const nn = n2;
        rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T>(
            nn, uplo, batch_count, &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3,
            &lsize_work4, &lsize_pivots, &lsize_iinfo, optim_mem);

        *size_scalars = std::max(*size_scalars, lsize_scalars);
        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_pivots = std::max(*size_pivots, lsize_pivots);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_potrf_rightlooking_template(rocblas_handle handle,
                                                     const rocblas_fill uplo,
                                                     const I n,
                                                     U A,
                                                     const I shiftA,
                                                     const I lda,
                                                     const rocblas_stride strideA,
                                                     I* info,
                                                     const I batch_count,
                                                     T* scalars,
                                                     void* work1,
                                                     void* work2,
                                                     void* work3,
                                                     void* work4,
                                                     T* pivots,
                                                     I* iinfo,
                                                     bool optim_mem)
{
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I const nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
        return rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                           batch_count, scalars, (T*)work1, pivots);

    // constants for rocblas functions calls
    T t_one = 1;
    S s_one = 1;
    S s_minone = -1;

    I jb = 0;
    I j = 0;

    // (TODO: When the matrix is detected to be non positive definite, we need to
    //  prevent TRSM and HERK to modify further the input matrix; ideally with no
    //  synchronizations.)

    if(uplo == rocblas_fill_upper)
    {
        // Compute the Cholesky factorization A = U'*U.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            jb = std::min(n - j, nb); // number of columns in the block
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, iinfo, batch_count, scalars, (T*)work1, pivots);

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, j,
                                    batch_count);

            if(j + jb < n)
            {
                {
                    auto const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, jb, (n - j - jb), A, shiftA + idx2D(j, j, lda),
                        lda, strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                        optim_mem, work1, work2, work3, work4);
                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }

                {
                    // update trailing submatrix
                    auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                        handle, uplo, rocblas_operation_conjugate_transpose, n - j - jb, jb,
                        &s_minone, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, &s_one, A,
                        shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);

                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }
            }
            j += nb;
        }
    }
    else
    {
        // Compute the Cholesky factorization A = L*L'.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            jb = std::min(n - j, nb); // number of columns in the block
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, iinfo, batch_count, scalars, (T*)work1, pivots);

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, j,
                                    batch_count);

            if(j + jb < n)
            {
                {
                    auto const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                        handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda),
                        lda, strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                        optim_mem, work1, work2, work3, work4);

                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }

                {
                    // update trailing submatrix
                    auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                        handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
                        shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
                        shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);

                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }
            }
            j += nb;
        }
    }

    // factor last block
    if(j < n)
    {
        rocsolver_potf2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    iinfo, batch_count, scalars, (T*)work1, pivots);
        ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, j,
                                batch_count);
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_potrf_getMemorySize(const rocblas_int n,
                                   const rocblas_fill uplo,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivots,
                                   size_t* size_iinfo,
                                   bool* optim_mem)
{
    *size_scalars = 0;
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_pivots = 0;
    *size_iinfo = batch_count;
    *optim_mem = true;
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    size_t lsize_scalars = 0;
    size_t lsize_work1 = 0;
    size_t lsize_work2 = 0;
    size_t lsize_work3 = 0;
    size_t lsize_work4 = 0;
    size_t lsize_pivots = 0;
    size_t lsize_iinfo = 0;

    {
        // -----------------------
        // right looking algorithm
        // -----------------------
        rocsolver_potrf_rightlooking_getMemorySize<BATCHED, STRIDED, T>(
            n, uplo, batch_count, &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3,
            &lsize_work4, &lsize_pivots, &lsize_iinfo, optim_mem);

        *size_scalars = std::max(*size_scalars, lsize_scalars);
        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_pivots = std::max(*size_pivots, lsize_pivots);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
    }

    {
        // -------------------
        // recursive algorithm
        // -------------------
        rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T>(
            n, uplo, batch_count, &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3,
            &lsize_work4, &lsize_pivots, &lsize_iinfo, optim_mem);

        *size_scalars = std::max(*size_scalars, lsize_scalars);
        *size_work1 = std::max(*size_work1, lsize_work1);
        *size_work2 = std::max(*size_work2, lsize_work2);
        *size_work3 = std::max(*size_work3, lsize_work3);
        *size_work4 = std::max(*size_work4, lsize_work4);
        *size_pivots = std::max(*size_pivots, lsize_pivots);
        *size_iinfo = std::max(*size_iinfo, lsize_iinfo);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_potrf_recursive_template(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const I n,
                                                  U A,
                                                  const I shiftA,
                                                  const I lda,
                                                  const rocblas_stride strideA,
                                                  I* info,
                                                  const I batch_count,
                                                  T* scalars,
                                                  void* work1,
                                                  void* work2,
                                                  void* work3,
                                                  void* work4,
                                                  T* pivots,
                                                  I* iinfo,
                                                  bool optim_mem,
                                                  I joffset)
{
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto const blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // constants for rocblas functions calls
    T t_one = 1;
    S s_one = 1;
    S s_minone = -1;

    auto const nsmall = POTRF_BLOCKSIZE(T);
    bool const is_nsmall = (n <= nsmall);
    if(is_nsmall)
    {
        if(iinfo != nullptr)
        {
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
        }

        auto const istat = rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                       (iinfo != nullptr) ? iinfo : info,
                                                       batch_count, scalars, (T*)work1, pivots);

        // test for non-positive-definiteness.
        if(iinfo != nullptr)
        {
            I const j = 0;
            ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info,
                                    j + joffset, batch_count);
        }

        return (istat);
    }
    else
    {
        auto const n2 = n / 2;
        auto const n1 = n - n2;

        auto const A11_offset = idx2D(0, 0, lda);
        auto const A21_offset = idx2D(n1, 0, lda);
        auto const A12_offset = idx2D(0, n1, lda);
        auto const A22_offset = idx2D(n1, n1, lda);

        if(uplo == rocblas_fill_lower)
        {
            // ------------------------------------------------
            // [A11  A21'] = [L11   0  ] * [L11'  L21']
            // [A21  A22 ]   [L21   L22]   [0     L22']
            //
            // where A11 is n1 by n1,  A22 is n2 by n2,  n == (n1 + n2)
            //
            // (1)  A11 = L11 * L11'     Cholesky factorization
            //
            // (2)  L21 * L11' = A21 or
            //      L21 = A21 / L11'     TRSM triangular solve
            //
            // (3)  L22 * L22' = (A22 - L21 * L21')
            // or
            // (3a)  A22 <-  A22 - L21 * L21',   SYRK
            // (3b)  L22 * L22' = A22    Cholesky factorization
            // ------------------------------------------------

            // --------------------
            // (1)  A11 = L11 * L11'
            // --------------------
            {
                // ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
                auto const nn = n1;

                I const j = 0;
                auto const istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, S, U>(
                    handle, uplo, nn, A, shiftA + A11_offset, lda, strideA, info, batch_count,
                    scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem, joffset + j);

                // test for non-positive-definiteness.
                // ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, j + joffset, batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // -------------------
            // (2) L21 = A21 / L11'
            // -------------------

            {
                // update trailing submatrix
                I const mm = n2;
                I const nn = n1;
                auto const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, mm, nn, A, shiftA + A11_offset, lda, strideA, A,
                    shiftA + A21_offset, lda, strideA, batch_count, optim_mem, work1, work2, work3,
                    work4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // --------------------------------------
            // (3a)  A22 <-  A22 - L21 * L21',   SYRK
            // --------------------------------------
            {
                I const nn = n2;
                I const kk = n1;

                auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, nn, kk, &s_minone, A, shiftA + A21_offset,
                    lda, strideA, &s_one, A, shiftA + A22_offset, lda, strideA, batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // ------------------------------------------------
            // (3b)  L22 * L22' = A22    Cholesky factorization
            // ------------------------------------------------
            {
                I const nn = n2;
                I const j = n1;

                // ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
                auto const istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, S, U>(
                    handle, uplo, nn, A, shiftA + A22_offset, lda, strideA, info, batch_count,
                    scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem, joffset + j);

                // test for non-positive-definiteness.
                // ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, joffset + j, batch_count);

                {
                    return (istat);
                };
            }
        }
        else
        {
            // -------------------------------------------------
            // A = U' * U
            // [A11  A12] = [ U11'  0   ] * [U11  U12]
            // [A12' A22]   [ U12'  U22']   [0    U22]
            //
            // where A11 is n1 by n1,  A22 is n2 by n2,  n == (n1 + n2)
            //
            // (1) A11 = U11' * U11,     Cholesky factorization
            //
            // (2) U11' * U12 = A12,     TRSM triangular solve
            // or  U12 = A12/U11'
            //
            // (3) U22' * U22 = (A22 - U12' * U12]
            // or
            // (3a)  A22 <- A22 - U12' * U12     SYRK
            // (3b)  U22' * U22 = A22    Cholesky factorization
            // -------------------------------------------------

            // -------------------------------------
            // (1) A11 = U11' * U11,     Cholesky factorization
            // -------------------------------------
            {
                // ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);

                auto const nn = n1;
                I const j = 0;
                rocblas_status const istat
                    = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, S, U>(
                        handle, uplo, nn, A, shiftA + A11_offset, lda, strideA, info, batch_count,
                        scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem, joffset + j);

                // test for non-positive-definiteness.
                // ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, j + joffset, batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // -----------------------------------------------
            // (2) U11' * U12 = A12,     TRSM triangular solve
            // or  U12 = A12/U11'
            // -----------------------------------------------
            {
                auto const mm = n1;
                auto const nn = n2;
                auto const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, mm, nn, A, shiftA + A11_offset, lda, strideA, A,
                    shiftA + A12_offset, lda, strideA, batch_count, optim_mem, work1, work2, work3,
                    work4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // ---------------------------------------
            // (3a)  A22 <- A22 - U12' * U12     SYRK
            // ---------------------------------------
            {
                auto const nn = n2;
                auto const kk = n1;
                auto const istat = rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, nn, kk, &s_minone, A,
                    shiftA + A12_offset, lda, strideA, &s_one, A, shiftA + A22_offset, lda, strideA,
                    batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // -------------------------------------------------
            // (3b)  U22' * U22 = A22    Cholesky factorization
            // -------------------------------------------------
            {
                I const nn = n2;
                I const j = n1;
                // ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
                //
                auto const istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, S, U>(
                    handle, uplo, nn, A, shiftA + A22_offset, lda, strideA, info, batch_count,
                    scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem, joffset + j);

                // test for non-positive-definiteness.
                // ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info, joffset + j, batch_count);

                {
                    return (istat);
                }
            }
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_potrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivots,
                                        rocblas_int* iinfo,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("potrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    auto const blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    rocblas_status istat = rocblas_status_success;
    if(use_recursive)
    {
        rocblas_int const joffset = 0;
        istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, rocblas_int, S, U>(
            handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2,
            work3, work4, pivots, iinfo, optim_mem, joffset);
    }
    else
    {
        istat = rocsolver_potrf_rightlooking_template<BATCHED, STRIDED, T, rocblas_int, S, U>(
            handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2,
            work3, work4, pivots, iinfo, optim_mem);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return istat;
}
