/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

/*
 * ===========================================================================
 *    common location for internal functions that reproduce LAPACK
 *    and BLAS functionality.
 * ===========================================================================
 */

/** This function returns the block size for the internal
    (blocked) trsm implementation **/
inline rocblas_int rocsolver_trsm_blksize(const rocblas_int m, const rocblas_int n)
{
    rocblas_int M = 6;
    rocblas_int N = 9;
    rocblas_int intervalsM[6] = {96, 160, 224, 352, 416, 480};
    rocblas_int intervalsN[9] = {256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096};
    //clang-format off
    rocblas_int size[7][10]
        = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},       {1, 1, 1, 1, 64, 64, 64, 48, 48, 0},
           {1, 1, 1, 64, 64, 64, 64, 48, 64, 0}, {1, 1, 64, 64, 64, 64, 64, 0, 0, 0},
           {1, 1, 64, 64, 64, 64, 0, 0, 0, 0},   {1, 64, 64, 64, 64, 64, 0, 0, 0, 0},
           {1, 64, 64, 64, 64, 0, 0, 0, 0, 0}};
    //clang-format on
    return size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
}

/** This function determine workspace size fir the internal trsm **/
template <bool BATCHED, typename T>
void rocsolver_trsm_mem(const rocblas_side side,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int batch_count,
                        size_t* size_work1,
                        size_t* size_work2,
                        size_t* size_work3,
                        size_t* size_work4,
                        bool* optim_mem)
{
    // determine block size
    rocblas_int blk = rocsolver_trsm_blksize(m, n);

    if(blk == 1)
        blk = m;

    if(blk == 0)
    {
        // (Note: rocblas TRSM workspace size is less than expected when the number of rows is multiple of 128.
        //  For this reason, when trying to set up a workspace that fits all the TRSM calls for m <= blk,
        //  blk cannot be multiple of 128.)
        //        rocblas_int mm = (blk % 128 != 0) ? blk : blk + 1;
        rocblas_int mm = (m % 128 != 0) ? m : m + 1;
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, mm, n, batch_count, size_work1,
                                         size_work2, size_work3, size_work4);
    }
    else
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
    }

    // always allocate all required memory for TRSM optimal performance
    *optim_mem = true;
}

/** Internal TRSM:
    Optimized function that solves a simple triangular system B <- Ax=B
    with A unit matrix. A and B are sub blocks of the same matrix MM with
    leading dimension ldim and stride. A and B are
    located in MM by their respective shifts.

    This is blocked implementation that calls the  internal trsm2_kernel
    to solve the diagonal blocks, and uses gemm to update the right-hand-sides **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm(rocblas_handle handle,
                    const rocblas_int m,
                    const rocblas_int n,
                    U MM,
                    const rocblas_int shiftA,
                    const rocblas_int shiftB,
                    const rocblas_int ldim,
                    const rocblas_stride stride,
                    const rocblas_int batch_count,
                    const bool optim_mem,
                    void* work1,
                    void* work2,
                    void* work3,
                    void* work4)
{
    ROCSOLVER_ENTER("trsm", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", ldim, "shiftB:", shiftB,
                    "ldb:", ldim, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, blocksy, jb;
    rocblas_int nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine block size
    rocblas_int blk = rocsolver_trsm_blksize(m, n);

    if(blk == 1)
        blk = m;

    if(blk == 0)
    {
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                     rocblas_operation_none, rocblas_diagonal_unit, m, n, &one, MM,
                                     shiftA, ldim, stride, MM, shiftB, ldim, stride, batch_count,
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
        ROCSOLVER_LAUNCH_KERNEL(trsm2_kernel<T>, grid, threads, lmemsize, stream, jb, n, MM,
                                shiftA + idx2D(j, j, ldim), shiftB + idx2D(j, 0, ldim), ldim, stride);

        // update right hand sides
        if(nextpiv < m)
        {
            rocblasCall_gemm<BATCHED, STRIDED, T>(
                handle, rocblas_operation_none, rocblas_operation_none, m - nextpiv, n, jb, &minone,
                MM, shiftA + idx2D(nextpiv, j, ldim), ldim, stride, MM, shiftB + idx2D(j, 0, ldim),
                ldim, stride, &one, MM, shiftB + idx2D(nextpiv, 0, ldim), ldim, stride, batch_count,
                nullptr);
        }
    }
}
