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
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots = 0;
        *size_iinfo = 0;
        *optim_mem = true;
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

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_potrf_rightlooking_template(rocblas_handle handle,
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
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    rocblas_int nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
        return rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                           batch_count, scalars, (T*)work1, pivots);

    // constants for rocblas functions calls
    T t_one = 1;
    S s_one = 1;
    S s_minone = -1;

    rocblas_int jb, j = 0;

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
                // update trailing submatrix
                rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, jb, (n - j - jb), A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);

                rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, n - j - jb, jb, &s_minone,
                    A, shiftA + idx2D(j, j + jb, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
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
                // update trailing submatrix
                rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);

                rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
                    shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
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

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_potrf_recursive_template(rocblas_handle handle,
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
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // constants for rocblas functions calls
    T t_one = 1;
    S s_one = 1;
    S s_minone = -1;

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    auto const STOPPING_NB = (sizeof(T) == 16) ? 512 : 1024;
    if(n <= STOPPING_NB)
    {
        return rocsolver_potrf_rightlooking_template<BATCHED, STRIDED, T, S, U>(
            handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2,
            work3, work4, pivots, iinfo, optim_mem);
    }
    else
    {
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

            auto n1 = n / 2;
            auto n2 = n - n1;

            // --------------------
            // (1)  A11 = L11 * L11'
            // --------------------
            {
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, 0);
                rocblas_status istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, S, U>(
                    handle, uplo, n1, A, shiftA, lda, strideA, info, batch_count, scalars, work1,
                    work2, work3, work4, pivots, iinfo, optim_mem);

                // test for non-positive-definiteness.
                auto const j = 0;
                ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info,
                                        j, batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                };
            }

            // -------------------
            // (2) L21 = A21 / L11'
            // -------------------

            {
                // update trailing submatrix
                rocblas_int const mm = n2;
                rocblas_int const nn = n1;
                rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, mm, nn, A, shiftA, lda, strideA, A,
                    shiftA + idx2D(n1, 0, lda), lda, strideA, batch_count, optim_mem, work1, work2,
                    work3, work4);
            }

            // --------------------------------------
            // (3a)  A22 <-  A22 - L21 * L21',   SYRK
            // --------------------------------------
            {
                rocblas_int const mm = n2;
                rocblas_int const nn = n1;

                rocblasCall_syrk_herk<BATCHED, T>(handle, uplo, rocblas_operation_none, mm, nn,
                                                  &s_minone, A, shiftA + idx2D(n1, 0, lda), lda,
                                                  strideA, &s_one, A, shiftA + idx2D(n1, n1, lda),
                                                  lda, strideA, batch_count);
            }

            // ------------------------------------------------
            // (3b)  L22 * L22' = A22    Cholesky factorization
            // ------------------------------------------------
            {
                auto const nn = n2;
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, 0);
                rocblas_status istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, S, U>(
                    handle, uplo, nn, A, shiftA + idx2D(n1, n1, lda), lda, strideA, info,
                    batch_count, scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);

                {
                    // test for non-positive-definiteness.
                    auto const j = n1;
                    ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo,
                                            info, j, batch_count);
                }

                if(istat != rocblas_status_success)
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
            //
            // (3) U22' * U22 = (A22 - U12' * U12]
            // or
            // (3a)  A22 <- A22 - U12' * U12     SYRK
            // (3b)  U22' * U22 = A22    Cholesky factorization
            // -------------------------------------------------

            auto const n1 = n / 2;
            auto const n2 = n - n1;
            // -------------------------------------
            // (1) A11 = U11' * U11,     Cholesky factorization
            // -------------------------------------
            {
                auto const j = 0;
                auto const nn = n1;
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, j);
                rocblas_status istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, S, U>(
                    handle, uplo, nn, A, shiftA, lda, strideA, info, batch_count, scalars, work1,
                    work2, work3, work4, pivots, iinfo, optim_mem);

                // test for non-positive-definiteness.
                ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info,
                                        j, batch_count);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            // -----------------------------------------------
            // (2) U11' * U12 = A12,     TRSM triangular solve
            // -----------------------------------------------
            {
                auto const mm = n1;
                auto const nn = n2;
                rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, mm, nn, A, shiftA, lda, strideA, A,
                    shiftA + idx2D(0, n1, lda), lda, strideA, batch_count, optim_mem, work1, work2,
                    work3, work4);
            }

            // ---------------------------------------
            // (3a)  A22 <- A22 - U12' * U12     SYRK
            // ---------------------------------------
            {
                auto const mm = n2;
                auto const nn = n1;
                rocblasCall_syrk_herk<BATCHED, T>(handle, uplo, rocblas_operation_conjugate_transpose,
                                                  mm, nn, &s_minone, A, shiftA, lda, strideA,
                                                  &s_one, A, shiftA + idx2D(n1, n1, lda), lda,
                                                  strideA, batch_count);
            }

            // -------------------------------------------------
            // (3b)  U22' * U22 = A22    Cholesky factorization
            // -------------------------------------------------
            {
                auto const j = n1;
                auto const nn = n2;
                ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo,
                                        batch_count, 0);
                rocblas_status istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, S, U>(
                    handle, uplo, nn, A, shiftA + idx2D(n1, n1, lda), lda, strideA, info,
                    batch_count, scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);

                // test for non-positive-definiteness.
                ROCSOLVER_LAUNCH_KERNEL(chk_positive<U>, gridReset, threads, 0, stream, iinfo, info,
                                        j, batch_count);

                if(istat != rocblas_status_success)
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

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    bool const use_recursive = false;
    rocblas_status const istat = (use_recursive)
        ? rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, S, U>(
            handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2,
            work3, work4, pivots, iinfo, optim_mem)
        : rocsolver_potrf_rightlooking_template<BATCHED, STRIDED, T, S, U>(
            handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2,
            work3, work4, pivots, iinfo, optim_mem);

    rocblas_set_pointer_mode(handle, old_mode);
    return istat;
}
