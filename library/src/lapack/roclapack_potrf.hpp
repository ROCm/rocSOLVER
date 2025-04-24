/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

#include "rocblas.hpp"
#include "roclapack_potf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
static __device__ __host__ bool get_use_recursive(I n)
{
    return (n > POTRF_RECURSIVE_SWITCHSIZE(T));
};

static bool constexpr use_non_recursive_potrf_in_recursion = false;

template <typename I>
static __device__ __host__ I split_n(I const n)
{
    assert(n >= 2);
    auto const n_over_2 = n / 2;
    auto const n1 = (rocsolver_is_po2(n_over_2)) ? n_over_2 : rocsolver_previous_po2(n_over_2);
    auto const n2 = n - n1;
    bool const is_valid = (n1 >= 1) && (n2 >= 1);

    return ((is_valid) ? n1 : 1);
};

template <typename I>
static I get_lds_size()
{
    I const default_lds_size = 64 * 1024;

    I lds_size = 0;
    I deviceId = 0;
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

template <typename I>
static __device__ __host__ I ceil(I const n, I const nb)
{
    return (1 + (n - 1) / nb);
}

template <typename I>
static __device__ __host__ void adjust_for_alignment(I* isize)
{
    // --------------------------------------------
    // align to cache line size
    // --------------------------------------------
    I const isize_in = *isize;
    if(isize_in == 0)
    {
        return;
    }

    I constexpr ialign = 128;
    *isize = ceil(isize_in, ialign) * ialign;
}

template <typename I, typename INFO>
ROCSOLVER_KERNEL void chk_positive(INFO* iinfo, INFO* info, I j, I batch_count)
{
    I id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count && info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO>
void rocsolver_potrf_non_recursive_getMemorySize(const I n,
                                                 const rocblas_fill uplo,
                                                 const I batch_count,
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

    I nb = POTRF_BLOCKSIZE(T);
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
        I jb = nb;
        size_t s1 = 0;
        size_t s2 = 0;

        // size to store info about positiveness of each subblock
        *size_iinfo = sizeof(INFO) * batch_count;

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

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_non_recursive_template(rocblas_handle handle,
                                                      const rocblas_fill uplo,
                                                      const I n,
                                                      U A,
                                                      const rocblas_stride shiftA,
                                                      const I lda,
                                                      const rocblas_stride strideA,
                                                      INFO* info,
                                                      const I batch_count,
                                                      T* scalars,
                                                      void* work1,
                                                      void* work2,
                                                      void* work3,
                                                      void* work4,
                                                      T* pivots,
                                                      INFO* iinfo,
                                                      bool optim_mem,
                                                      const I row_offset = 0)
{
    ROCSOLVER_ENTER("potrf_non_recursive", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I const nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        return rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                           batch_count, scalars, (T*)work1, pivots);
    }

    // constants for rocblas functions calls
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

            ROCBLAS_CHECK(rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda),
                                                      lda, strideA, iinfo, batch_count, scalars,
                                                      (T*)work1, pivots));

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, 0, stream,

                                    iinfo, info, j + row_offset, batch_count);

            if(j + jb < n)
            {
                // update trailing submatrix
                ROCBLAS_CHECK(rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, jb, (n - j - jb), A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4));

                ROCBLAS_CHECK(rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, n - j - jb, jb, &s_minone,
                    A, shiftA + idx2D(j, j + jb, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count));
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
            ROCBLAS_CHECK(rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda),
                                                      lda, strideA, iinfo, batch_count, scalars,
                                                      (T*)work1, pivots));

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, 0, stream, iinfo, info,
                                    j + row_offset, batch_count);

            if(j + jb < n)
            {
                // update trailing submatrix
                ROCBLAS_CHECK(rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4));

                ROCBLAS_CHECK(rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
                    shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count));
            }
            j += nb;
        }
    }

    // factor last block
    if(j < n)
    {
        ROCBLAS_CHECK(rocsolver_potf2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda),
                                                  lda, strideA, iinfo, batch_count, scalars,
                                                  (T*)work1, pivots));
        ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, 0, stream, iinfo, info,
                                j + row_offset, batch_count);
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO>
void rocsolver_potrf_recursive_getMemorySize(const I n,
                                             const rocblas_fill uplo,
                                             const I batch_count,
                                             size_t* size_work)
{
    *size_work = 0;
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    I const nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // requirements for calling a single POTF2
        size_t size_iinfo = sizeof(INFO) * batch_count;
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_pivots = 0;
        rocsolver_potf2_getMemorySize<T>(n, batch_count, &size_scalars, &size_work1, &size_pivots);

        adjust_for_alignment(&size_iinfo);
        adjust_for_alignment(&size_scalars);
        adjust_for_alignment(&size_work1);
        adjust_for_alignment(&size_pivots);

        size_t size_potf2 = size_iinfo + size_scalars + size_work1 + size_pivots;
        *size_work = std::max(*size_work, size_potf2);
    }
    else if(use_non_recursive_potrf_in_recursion && (n <= POTRF_RECURSIVE_SWITCHSIZE(T)))
    {
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivots = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_potrf_non_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
            n, uplo, batch_count,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivots,
            &size_iinfo, &optim_mem);

        adjust_for_alignment(&size_scalars);

        adjust_for_alignment(&size_work1);
        adjust_for_alignment(&size_work2);
        adjust_for_alignment(&size_work3);
        adjust_for_alignment(&size_work4);

        adjust_for_alignment(&size_pivots);
        adjust_for_alignment(&size_iinfo);

        size_t size_potrf = size_scalars + size_work1 + size_work2 + size_work3 + size_work4
            + size_pivots + size_iinfo;
        *size_work = std::max(*size_work, size_potrf);

#if(0)
        I const jb = nb;

        size_t size_potf2 = 0;
        {
            size_t size_scalars = 0;
            size_t s1 = 0;
            size_t size_pivots = 0;

            // size to store info about positiveness of each subblock
            size_t size_iinfo = sizeof(INFO) * batch_count;

            // requirements for calling POTF2 for the subblocks
            rocsolver_potf2_getMemorySize<T>(jb, batch_count, &size_scalars, &s1, &size_pivots);

            adjust_for_alignment(&size_scalars);
            adjust_for_alignment(&s1);
            adjust_for_alignment(&size_pivots);

            adjust_for_alignment(&size_iinfo);

            size_potf2 = size_scalars + s1 + size_pivots + size_iinfo;
        }

        size_t size_trsm = 0;
        {
            bool optim_mem = true;
            size_t s2 = 0;
            size_t size_work2 = 0;
            size_t size_work3 = 0;
            size_t size_work4 = 0;

            // extra requirements for calling TRSM
            if(uplo == rocblas_fill_upper)
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T, I>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb,
                    batch_count, &s2, &size_work2, &size_work3, &size_work4, &optim_mem);
            }
            else
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T, I>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n - jb, jb,
                    batch_count, &s2, &size_work2, &size_work3, &size_work4, &optim_mem);
            }

            adjust_for_alignment(&s2);
            adjust_for_alignment(&size_work2);
            adjust_for_alignment(&size_work3);
            adjust_for_alignment(&size_work4);

            size_trsm = s2 + size_work2 + size_work3 + size_work4;
        }

        *size_work = std::max(*size_work, std::max(size_trsm, size_potf2));
#endif
    }
    else
    {
        // requirements for recursive POTRF
        auto const n1 = split_n(n);
        auto const n2 = n - n1;

        // size to store info about positiveness of each subblock

        // requirements for calling POTRF recursively on submatrices

        size_t size_potrf_n1 = 0;
        {
            rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                n1, uplo, batch_count, &size_potrf_n1);

            adjust_for_alignment(&size_potrf_n1);
        }

        size_t size_potrf_n2 = 0;
        {
            rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                n2, uplo, batch_count, &size_potrf_n2);

            adjust_for_alignment(&size_potrf_n2);
        }

        // extra requirements for calling TRSM
        size_t size_trsm = 0;
        {
            size_t w13 = 0;
            size_t w23 = 0;
            size_t w33 = 0;
            size_t w43 = 0;
            bool opt3 = true;

            if(uplo == rocblas_fill_upper)
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left,
                                                        rocblas_operation_conjugate_transpose, n1, n2,
                                                        batch_count, &w13, &w23, &w33, &w43, &opt3);
            }
            else
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right,
                                                        rocblas_operation_conjugate_transpose, n2, n1,
                                                        batch_count, &w13, &w23, &w33, &w43, &opt3);
            }

            adjust_for_alignment(&w13);
            adjust_for_alignment(&w23);
            adjust_for_alignment(&w33);
            adjust_for_alignment(&w43);

            size_trsm = w13 + w23 + w33 + w43;
        }

        *size_work
            = std::max(*size_work, std::max(size_trsm, std::max(size_potrf_n1, size_potrf_n2)));
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_potrf_getMemorySize(const I n,
                                   const rocblas_fill uplo,
                                   const I batch_count,
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

    // ---------------------------------------
    // TODO: assume type INFO is the same as  type I
    //
    // The potrf_template has type INFO in template argument
    // but potrf_getMemSize does not have type INFO
    // as template argument
    // ---------------------------------------
    using INFO = decltype(I{});

    {
        // ------------------------------------------------------------------------------
        // Note: call potrf_non_recursive_getMemorySzie even when using recursive option
        // for backward compatibility, just in case other code try to reuse
        // scratch space intended for potrf
        // ------------------------------------------------------------------------------
        rocsolver_potrf_non_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
            n, uplo, batch_count,

            size_scalars, size_work1, size_work2, size_work3, size_work4, size_pivots, size_iinfo,
            optim_mem);
    }

    bool const use_recursive = get_use_recursive<T>(n);
    if(use_recursive)
    {
        size_t size_work = 0;
        rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(n, uplo, batch_count,
                                                                              &size_work);

        // --------------------------------------------------------------
        // all workspace for recursive routine allocated in array work1[]
        // --------------------------------------------------------------
        *size_work1 = std::max(*size_work1, size_work);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_recursive_template(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const I n,
                                                  U A,
                                                  const rocblas_stride shiftA,
                                                  const I lda,
                                                  const rocblas_stride strideA,
                                                  INFO* info,
                                                  const I batch_count,
                                                  void* work,
                                                  size_t size_work,
                                                  const I row_offset = 0)
{
    ROCSOLVER_ENTER("potrf_recursive", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count, "row_offset:", row_offset);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

#define CHECK_MEM()                                           \
    {                                                         \
        bool const isok_mem = (pfree <= (pwork + size_work)); \
        assert(isok_mem);                                     \
        if(!isok_mem)                                         \
        {                                                     \
            return (rocblas_status_memory_error);             \
        }                                                     \
    }

    // -------------------------------------------------
    // UNBLOCKED ALGORITHM FOR SMALL MATRICES
    // -------------------------------------------------
    I const nb = POTRF_BLOCKSIZE(T);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // constants for rocblas functions calls
    S s_one = 1;
    S s_minone = -1;

    // (TODO: When the matrix is detected to be non positive definite, we need to
    //  prevent TRSM and HERK to modify further the input matrix; ideally with no
    //  synchronizations.)

    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // -----------------------
        // small matrix, use POTF2
        // -----------------------

        auto const pfree_saved = pfree;

        size_t size_iinfo = sizeof(INFO) * batch_count;
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_pivots = 0;

        rocsolver_potf2_getMemorySize<T>(n, batch_count, &size_scalars, &size_work1, &size_pivots);

        adjust_for_alignment(&size_iinfo);
        adjust_for_alignment(&size_scalars);
        adjust_for_alignment(&size_work1);
        adjust_for_alignment(&size_pivots);

        INFO* const iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        T* const scalars = (T*)pfree;
        pfree += size_scalars;
        T* const work1 = (T*)pfree;
        pfree += size_work1;
        T* const pivots = (T*)pfree;
        pfree += size_pivots;

        CHECK_MEM();

        // Factor diagonal and subdiagonal blocks
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
        I const j = 0;
        auto const istat
            = rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA + idx2D(j, j, lda), lda,
                                          strideA, iinfo, batch_count, scalars, work1, pivots);

        // test for non-positive-definiteness.
        ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, 0, stream,

                                iinfo, info, j + row_offset, batch_count);

        pfree = pfree_saved;
        return istat;
    }
    else if(use_non_recursive_potrf_in_recursion && (n <= POTRF_RECURSIVE_SWITCHSIZE(T)))
    {
        // -------------------------------------------------
        // RIGHT-LOOKING ALGORITHM FOR MEDIUM MATRICES
        // -------------------------------------------------
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivots = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_potrf_non_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
            n, uplo, batch_count,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivots,
            &size_iinfo, &optim_mem);

        adjust_for_alignment(&size_scalars);

        adjust_for_alignment(&size_work1);
        adjust_for_alignment(&size_work2);
        adjust_for_alignment(&size_work3);
        adjust_for_alignment(&size_work4);

        adjust_for_alignment(&size_pivots);
        adjust_for_alignment(&size_iinfo);

        T* const scalars = (T*)pfree;
        pfree += size_scalars;

        void* const work1 = (T*)pfree;
        pfree += size_work1;
        void* const work2 = (T*)pfree;
        pfree += size_work2;
        void* const work3 = (T*)pfree;
        pfree += size_work3;
        void* const work4 = (T*)pfree;
        pfree += size_work4;

        T* const pivots = (T*)pfree;
        pfree += size_pivots;
        INFO* const iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        CHECK_MEM();

        auto const istat = rocsolver_potrf_non_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem, row_offset);

        /*
        ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, 0, stream, iinfo, info,
                                row_offset, batch_count);
				*/
        return (istat);
    }
    else
    {
        // -------------------------------------------------
        // RECURSIVE ALGORITHM FOR LARGE MATRICES
        // -------------------------------------------------
        auto const n1 = split_n(n);
        auto const n2 = n - n1;

        if(uplo == rocblas_fill_upper)
        {
            // -------------------------------------------------
            // A = U' * U
            // [A11  A12] = [ U11'  0   ] * [U11  U12]
            // [A12' A22]   [ U12'  U22']   [0    U22]
            //
            // where A11 is n1 by n1,  A22 is n2 by n2,  n == (n1 + n2)
            // -------------------------------------------------

            // ----------------------------------
            // compute U11 given A11 = U11' * U11
            // ----------------------------------
            {
                auto const pfree_saved = pfree;

                size_t size_work1 = 0;
                rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n1, uplo, batch_count, &size_work1);

                adjust_for_alignment(&size_work1);
                void* const work1 = (void*)pfree;

                CHECK_MEM();

                ROCBLAS_CHECK(rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
                    handle, uplo, n1, A, shiftA, lda, strideA, info, batch_count, work1, size_work,
                    row_offset));
            }

            auto const A12_offset = idx2D(0, n1, lda);
            auto const A22_offset = idx2D(n1, n1, lda);

            {
                auto const pfree_saved = pfree;

                size_t size_work1 = 0;
                size_t size_work2 = 0;
                size_t size_work3 = 0;
                size_t size_work4 = 0;
                bool optim_mem = true;

                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, n1, n2, batch_count,
                    &size_work1, &size_work2, &size_work3, &size_work4, &optim_mem);

                adjust_for_alignment(&size_work1);
                adjust_for_alignment(&size_work2);
                adjust_for_alignment(&size_work3);
                adjust_for_alignment(&size_work4);

                T* const work1 = (T*)pfree;
                pfree += size_work1;
                T* const work2 = (T*)pfree;
                pfree += size_work2;
                T* const work3 = (T*)pfree;
                pfree += size_work3;
                T* const work4 = (T*)pfree;
                pfree += size_work4;

                // ----------------------------------
                // compute U12 given A12 = U11' * U12
                // ----------------------------------
                ROCBLAS_CHECK(rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, n1, n2, A, shiftA, lda, strideA, A, shiftA + A12_offset,
                    lda, strideA, batch_count, optim_mem, work1, work2, work3, work4));

                pfree = pfree_saved;
            }

            // ------------------------------
            // update A22 as A22 - U12' * U12
            // ------------------------------
            ROCBLAS_CHECK(rocblasCall_syrk_herk<BATCHED, T>(
                handle, uplo, rocblas_operation_conjugate_transpose, n2, n1, &s_minone, A,
                shiftA + A12_offset, lda, strideA, &s_one, A, shiftA + A22_offset, lda, strideA,
                batch_count));

            {
                size_t size_work1 = 0;
                rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n2, uplo, batch_count, &size_work1);

                adjust_for_alignment(&size_work1);

                void* const work1 = (void*)pfree;

                CHECK_MEM();

                // ----------------------------------
                // compute U22 given A22 = U22' * U22
                // ----------------------------------
                ROCBLAS_CHECK(rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
                    handle, uplo, n2, A, shiftA + A22_offset, lda, strideA, info, batch_count,
                    work1, size_work, row_offset + n1));
            }
        }
        else
        {
            // ------------------------------------------------
            // A = L * L'
            // [A11  A21'] = [L11   0  ] * [L11'  L21']
            // [A21  A22 ]   [L21   L22]   [0     L22']
            //
            // where A11 is n1 by n1,  A22 is n2 by n2,  n == (n1 + n2)
            // ------------------------------------------------

            // ---------------------------------
            // compute L11 given A11 = L11 * L11'
            // ---------------------------------
            {
                size_t size_work1 = 0;
                rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n1, uplo, batch_count, &size_work1);

                adjust_for_alignment(&size_work1);

                void* const work1 = (void*)pfree;

                CHECK_MEM();

                ROCBLAS_CHECK(rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
                    handle, uplo, n1, A, shiftA, lda, strideA, info, batch_count, work1, size_work,
                    row_offset));
            }

            // ---------------------------------
            // compute L21 given A21 = L21 * L11'
            // ---------------------------------
            auto const A21_offset = idx2D(n1, 0, lda);
            auto const A22_offset = idx2D(n1, n1, lda);
            {
                auto const pfree_saved = pfree;

                size_t size_work1 = 0;
                size_t size_work2 = 0;
                size_t size_work3 = 0;
                size_t size_work4 = 0;

                bool optim_mem = true;

                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n2, n1, batch_count,
                    &size_work1, &size_work2, &size_work3, &size_work4, &optim_mem);

                adjust_for_alignment(&size_work1);
                adjust_for_alignment(&size_work2);
                adjust_for_alignment(&size_work3);
                adjust_for_alignment(&size_work4);

                T* const work1 = (T*)pfree;
                pfree += size_work1;
                T* const work2 = (T*)pfree;
                pfree += size_work2;
                T* const work3 = (T*)pfree;
                pfree += size_work3;
                T* const work4 = (T*)pfree;
                pfree += size_work4;

                CHECK_MEM();

                ROCBLAS_CHECK(rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, n2, n1, A, shiftA, lda, strideA, A, shiftA + A21_offset,
                    lda, strideA, batch_count, optim_mem, work1, work2, work3, work4));

                pfree = pfree_saved;
            }
            // ------------------------------
            // update A22 as A22 - L21 * L21'
            // ------------------------------
            ROCBLAS_CHECK(rocblasCall_syrk_herk<BATCHED, T>(
                handle, uplo, rocblas_operation_none, n2, n1, &s_minone, A, shiftA + A21_offset,
                lda, strideA, &s_one, A, shiftA + A22_offset, lda, strideA, batch_count));

            // ----------------------------------
            // compute L22 given A22 = L22 * L22'
            // ----------------------------------
            {
                size_t size_work1 = 0;

                rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(
                    n2, uplo, batch_count, &size_work1);

                void* const work1 = (void*)pfree;
                pfree += size_work1;

                CHECK_MEM();

                ROCBLAS_CHECK(rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
                    handle, uplo, n2, A, shiftA + A22_offset, lda, strideA, info, batch_count,
                    work1, size_work1, row_offset + n1));
            }
        }

        return rocblas_status_success;
    }
    return rocblas_status_success;

#undef CHECK_MEM
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivots,
                                        INFO* iinfo,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("potrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);
    {
        // quick return

        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return rocblas_status_success;
        }
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;

    bool const use_recursive = get_use_recursive<T>(n);
    if(use_recursive)
    {
        size_t size_work1 = 0;
        rocsolver_potrf_recursive_getMemorySize<BATCHED, STRIDED, T, I, INFO>(n, uplo, batch_count,
                                                                              &size_work1);

        I const row_offset = 0;
        istat = rocsolver_potrf_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            work1, size_work1,

            row_offset);
    }
    else
    {
        istat = rocsolver_potrf_non_recursive_template<BATCHED, STRIDED, T, I, INFO, S, U>(
            handle, uplo, n,

            A, shiftA, lda, strideA,

            info, batch_count,

            scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return istat;
}

ROCSOLVER_END_NAMESPACE
