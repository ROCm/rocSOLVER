/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/*
 * ===========================================================================
 *    common location for internal functions that reproduce LAPACK
 *    and BLAS functionality.
 * ===========================================================================
 */

/** Constants for block size of trsm **/
// clang-format off
#define TRSM_NUMROWS_REAL 12
#define TRSM_NUMCOLS_REAL 16
#define TRSM_INTERVALSROW_REAL                                          \
    40, 56, 80, 112, 144, 176, 208, 240, 288, 352, 480
#define TRSM_INTERVALSCOL_REAL                                          \
    448, 768, 960, 1152, 1408, 1920, 2304, 2816, 3840, 4096, 4736,      \
    4992, 5888, 7680, 9728
#define TRSM_BLKSIZES_REAL                                              \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 24, 24, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1, 32, 32, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1, 48, 48, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1, 80, 80, 80, 56, 56, 40, 40, 40, 32, 32, 32},    \
    {1,  1,  1,  1, 80, 80, 80, 80, 80, 48, 48, 48, 40, 32,  0,  0},    \
    {1,  1,  1, 80, 80, 80, 80, 80, 56, 56, 32, 32, 32, 32,  0,  0},    \
    {1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}

#define TRSM_NUMROWS_COMPLEX 10
#define TRSM_NUMCOLS_COMPLEX 12
#define TRSM_INTERVALSROW_COMPLEX                                       \
    40, 56, 80, 112, 144, 208, 240, 288, 480
#define TRSM_INTERVALSCOL_COMPLEX                                       \
    704, 960, 1344, 1920, 2304, 2816, 3200, 3840, 4864, 5888, 7680
#define TRSM_BLKSIZES_COMPLEX                                           \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                               \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 24, 24},                            \
    {1, 1, 1, 1, 1, 1, 1, 1, 32, 32, 32, 32},                           \
    {1, 1, 1, 1, 1, 72, 72, 56, 48, 32, 32, 32},                        \
    {1, 1, 1, 1, 64, 64, 64, 64, 48, 32, 32, 32},                       \
    {1, 1, 1, 80, 80, 80, 64, 64, 48, 32, 32, 32},                      \
    {1, 1, 80, 80, 80, 80, 64, 64, 40, 40, 32, 32},                     \
    {1, 1, 72, 72, 64, 64, 64, 64, 32, 32, 32, 0},                      \
    {1, 80, 80, 80, 80, 80, 64, 64, 48, 40, 32, 0},                     \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define TRSM_BATCH_NUMROWS_REAL 11
#define TRSM_BATCH_NUMCOLS_REAL 17
#define TRSM_BATCH_INTERVALSROW_REAL                                        \
    20, 28, 40, 80, 112, 176, 208, 288, 352, 480
#define TRSM_BATCH_INTERVALSCOL_REAL                                        \
    6, 10, 12, 22, 28, 30, 36, 42, 46, 50, 60, 96, 432, 928, 960, 1472
#define TRSM_BATCH_BLKSIZES_REAL                                            \
    { 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1, 24, 24, 24, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},   \
    {48, 48, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 16, 16, 16, 16, 16},   \
    {64, 64, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 24, 24, 24, 24, 24},   \
    {64, 64, 32, 32, 24, 24, 24, 24, 32, 32, 32, 32, 32, 24, 24, 24, 24},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 40, 40, 40, 32, 32, 24, 24, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32,  0},   \
    {64, 64, 64, 32, 32, 32, 48, 48, 48, 48, 40, 32, 32, 32,  0,  0,  0}

#define TRSM_BATCH_NUMROWS_COMPLEX 10
#define TRSM_BATCH_NUMCOLS_COMPLEX 16
#define TRSM_BATCH_INTERVALSROW_COMPLEX                                     \
    20, 28, 40, 56, 80, 112, 144, 176, 480
#define TRSM_BATCH_INTERVALSCOL_COMPLEX                                     \
    4, 12, 16, 28, 32, 40, 48, 50, 60, 72, 88, 176, 232, 400, 464
#define TRSM_BATCH_BLKSIZES_COMPLEX                                         \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  8,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1, 16, 16, 16, 16,  1,  1,  1, 16, 16, 16, 16, 16},        \
    {1,  1,  1, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},        \
    {1,  1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 48, 48, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 80, 80, 56, 56, 32, 32},        \
    {1, 64, 32, 32, 32, 64, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32},        \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 64, 64, 48, 48, 48}
// clang-format on

/** This function returns the block size for the internal
    (blocked) trsm implementation **/
template <bool ISBATCHED, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_int rocsolver_trsm_blksize(const rocblas_int m, const rocblas_int n)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int M = TRSM_BATCH_NUMROWS_REAL - 1;
        rocblas_int N = TRSM_BATCH_NUMCOLS_REAL - 1;
        rocblas_int intervalsM[] = {TRSM_BATCH_INTERVALSROW_REAL};
        rocblas_int intervalsN[] = {TRSM_BATCH_INTERVALSCOL_REAL};
        rocblas_int size[][TRSM_BATCH_NUMCOLS_REAL] = {TRSM_BATCH_BLKSIZES_REAL};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }
    else
    {
        rocblas_int M = TRSM_NUMROWS_REAL - 1;
        rocblas_int N = TRSM_NUMCOLS_REAL - 1;
        rocblas_int intervalsM[] = {TRSM_INTERVALSROW_REAL};
        rocblas_int intervalsN[] = {TRSM_INTERVALSCOL_REAL};
        rocblas_int size[][TRSM_NUMCOLS_REAL] = {TRSM_BLKSIZES_REAL};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }

    if(blk == 1)
        blk = m;

    return blk;
}

/** complex type version **/
template <bool ISBATCHED, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_int rocsolver_trsm_blksize(const rocblas_int m, const rocblas_int n)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int M = TRSM_BATCH_NUMROWS_COMPLEX - 1;
        rocblas_int N = TRSM_BATCH_NUMCOLS_COMPLEX - 1;
        rocblas_int intervalsM[] = {TRSM_BATCH_INTERVALSROW_COMPLEX};
        rocblas_int intervalsN[] = {TRSM_BATCH_INTERVALSCOL_COMPLEX};
        rocblas_int size[][TRSM_BATCH_NUMCOLS_COMPLEX] = {TRSM_BATCH_BLKSIZES_COMPLEX};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }
    else
    {
        rocblas_int M = TRSM_NUMROWS_COMPLEX - 1;
        rocblas_int N = TRSM_NUMCOLS_COMPLEX - 1;
        rocblas_int intervalsM[] = {TRSM_INTERVALSROW_COMPLEX};
        rocblas_int intervalsN[] = {TRSM_INTERVALSCOL_COMPLEX};
        rocblas_int size[][TRSM_NUMCOLS_COMPLEX] = {TRSM_BLKSIZES_COMPLEX};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }

    if(blk == 1)
        blk = m;

    return blk;
}

/** This function determine workspace size for the internal trsm **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_trsm_mem(const rocblas_side side,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int batch_count,
                        size_t* size_work1,
                        size_t* size_work2,
                        size_t* size_work3,
                        size_t* size_work4,
                        bool* optim_mem,
                        bool inblocked = false)
{
    // always allocate all required memory for TRSM optimal performance
    *optim_mem = true;

    rocblas_int mm = m;

    if(!inblocked)
    {
        static constexpr bool ISBATCHED = BATCHED || STRIDED;

        // determine block size
        rocblas_int blk = rocsolver_trsm_blksize<ISBATCHED, T>(m, n);

        if(blk > 0)
        {
            *size_work1 = 0;
            *size_work2 = 0;
            *size_work3 = 0;
            *size_work4 = 0;
            return;
        }
        else
            mm = m;
    }
    else
    {
        // inblocked = true when called from inside blocked algorithms like GETRF.

        // (Note: rocblas TRSM workspace size is less than expected when the number of rows is multiple of 128.
        //  For this reason, when trying to set up a workspace that fits all the TRSM calls for m <= blk,
        //  blk cannot be multiple of 128.)
        //        rocblas_int mm = (blk % 128 != 0) ? blk : blk + 1;
        mm = (m % 128 != 0) ? m : m + 1;
    }

    rocblasCall_trsm_mem<BATCHED, T>(side, rocblas_operation_none, mm, n, batch_count, size_work1,
                                     size_work2, size_work3, size_work4);
}

/** Internal TRSM (lower case):
    Optimized function that solves a simple triangular system B <- Ax=B
    with A unit lower triangular matrix. A and B are sub blocks of the same matrix MM with
    leading dimension ldim and stride. A and B are
    located in MM by their respective shifts.

    This is blocked implementation that calls the  internal trsm2_kernel
    to solve the diagonal blocks, and uses gemm to update the right-hand-sides **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_lower(rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4)
{
    ROCSOLVER_ENTER("trsm_lower", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, blocksy, jb;
    rocblas_int nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine block size
    rocblas_int blk = rocsolver_trsm_blksize<ISBATCHED, T>(m, n);

    if(blk == 0)
    {
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                     rocblas_operation_none, rocblas_diagonal_unit, m, n, &one, A,
                                     shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                     optim_mem, work1, work2, work3, work4);
        return;
    }

    // main loop
    for(rocblas_int j = 0; j < m; j += blk)
    {
        jb = min(m - j, blk);
        nextpiv = j + jb;

        // solve for current diagonal block
        dimx = jb;
        dimy = 1024 / dimx;
        blocks = (n - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimy * sizeof(T);
        ROCSOLVER_LAUNCH_KERNEL(trsm2_lower_kernel<T>, grid, threads, lmemsize, stream, jb, n, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, B, shiftB + idx2D(j, 0, ldb), ldb, strideB);

        // update right hand sides
        if(nextpiv < m)
        {
            rocblasCall_gemm<BATCHED, STRIDED, T>(
                handle, rocblas_operation_none, rocblas_operation_none, m - nextpiv, n, jb, &minone,
                A, shiftA + idx2D(nextpiv, j, lda), lda, strideA, B, shiftB + idx2D(j, 0, ldb),
                ldb, strideB, &one, B, shiftB + idx2D(nextpiv, 0, ldb), ldb, strideB, batch_count,
                nullptr);
        }
    }
}

/** Internal TRSM (upper case):
    Optimized function that solves a simple triangular system B <- xA=B
    with A non-unit upper triangular matrix. A and B are sub blocks of the same matrix MM with
    leading dimension ldim and stride. A and B are
    located in MM by their respective shifts.

    This is blocked implementation that calls the  internal trsm2_kernel
    to solve the diagonal blocks, and uses gemm to update the right-hand-sides **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_upper(rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4)
{
    ROCSOLVER_ENTER("trsm_upper", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, blocksy, jb;
    rocblas_int nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine block size
    rocblas_int blk = rocsolver_trsm_blksize<ISBATCHED, T>(n, m);

    if(blk == 0)
    {
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_right, rocblas_fill_upper,
                                     rocblas_operation_none, rocblas_diagonal_non_unit, m, n, &one,
                                     A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                     batch_count, optim_mem, work1, work2, work3, work4);
        return;
    }

    // main loop
    for(rocblas_int j = 0; j < n; j += blk)
    {
        jb = min(n - j, blk);
        nextpiv = j + jb;

        // solve for current diagonal block
        dimy = jb;
        dimx = 1024 / dimy;
        blocks = (m - 1) / dimx + 1;
        grid = dim3(blocks, 1, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimx * sizeof(T);
        ROCSOLVER_LAUNCH_KERNEL(trsm2_upper_kernel<T>, grid, threads, lmemsize, stream, m, jb, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, B, shiftB + idx2D(0, j, ldb), ldb, strideB);

        // update right hand sides
        if(nextpiv < n)
        {
            rocblasCall_gemm<BATCHED, STRIDED, T>(
                handle, rocblas_operation_none, rocblas_operation_none, m, n - nextpiv, jb, &minone,
                B, shiftB + idx2D(0, j, ldb), ldb, strideB, A, shiftA + idx2D(j, nextpiv, lda),
                lda, strideA, &one, B, shiftB + idx2D(0, nextpiv, ldb), ldb, strideB, batch_count,
                nullptr);
        }
    }
}
