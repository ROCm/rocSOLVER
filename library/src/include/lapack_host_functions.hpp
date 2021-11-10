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

/** Optimized function that solves a simple triangular system B <- Ax=B
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
                    const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, blocksy, jb;
    rocblas_int nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine block size
    rocblas_int blk;
    if(m <= 44)
        blk = m;
    else if(m <= 68)
    {
        if(n <= 6144)
            blk = m;
        else
            blk = 32;
    }
    else if(m <= 88)
    {
        if(n <= 2048)
            blk = m;
        else
            blk = 32;
    }
    else if(m <= 120)
    {
        if(n <= 1536)
            blk = m;
        else
            blk = 32;
    }
    else if(m <= 192)
    {
        if(n <= 1024)
            blk = m;
        else
            blk = 32;
    }
    else
        blk = 64;

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
        hipLaunchKernelGGL(trsm2_kernel<T>, grid, threads, lmemsize, stream, jb, n, MM,
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
