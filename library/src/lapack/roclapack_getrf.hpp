/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver.h"

/** This function returns the outter block size based on defined variables
    tunable by the user (defined in ideal_sizes.hpp) **/
template <bool ISBATCHED, bool PIVOT>
rocblas_int getrf_get_blksize(rocblas_int dim)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(PIVOT)
        {
            rocblas_int size[] = {GETRF_BATCH_BLKSIZES};
            rocblas_int intervals[] = {GETRF_BATCH_INTERVALS};
            rocblas_int max = GETRF_BATCH_NUM_INTERVALS;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BATCH_BLKSIZES};
            rocblas_int intervals[] = {GETRF_NPVT_BATCH_INTERVALS};
            rocblas_int max = GETRF_NPVT_BATCH_NUM_INTERVALS;
            blk = size[get_index(intervals, max, dim)];
        }
    }
    else
    {
        if(PIVOT)
        {
            rocblas_int size[] = {GETRF_BLKSIZES};
            rocblas_int intervals[] = {GETRF_INTERVALS};
            rocblas_int max = GETRF_NUM_INTERVALS;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BLKSIZES};
            rocblas_int intervals[] = {GETRF_NPVT_INTERVALS};
            rocblas_int max = GETRF_NPVT_NUM_INTERVALS;
            blk = size[get_index(intervals, max, dim)];
        }
    }

    return blk;
}

/** This function returns the inner block size. This has been tuned based on
    experiments with panel matrices; it is not expected to change a lot.
    (not tunable by user for now) **/
/*inline rocblas_int get_1st_innerBlkSize(rocblas_int m, rocblas_int n)
{
    rocblas_int blk;

    if(n <= 80) // n = 16,32,48,64
    {
        blk = n;
    }
    else if(n <= 144) // n = 96,128
    {
        if(m > 4352)
            blk = 32;
        else
            blk = n;
    }
    else if(n <= 224) // n = 160,192
    {
        if(m <= 1664)
            blk = n;
        else if(m <= 3712)
            blk = 96;
        else if(m <= 7936)
            blk = 128;
        else
            blk = 32;
    }
    else if(n <= 288) // n = 256
    {
        if(m <= 3456)
            blk = n;
        else if(m <= 3968)
            blk = 96;
        else
            blk = 128;
    }
    else if(n <= 352) // n = 320
    {
        if(m <= 2176)
            blk = n;
        else if(m <= 4352)
            blk = 96;
        else
            blk = 128;
    }
    else if(n <= 416) // n = 384
    {
        if(m <= 320)
            blk = n;
        else if(m <= 5376)
            blk = 128;
        else
            blk = 256;
    }
    else if(n <= 480) // n = 448
    {
        if(m <= 448)
            blk = 288;
        else if(m <= 4864)
            blk = 96;
        else
            blk = 256;
    }
    else if(n <= 544) // n = 512
    {
        if(m <= 2432)
            blk = n;
        else if(m <= 7424)
            blk = 256;
        else
            blk = 128;
    }
    else if(n <= 608) // n = 576
    {
        if(m <= 1408)
            blk = n;
        else if(m <= 9472)
            blk = 256;
        else
            blk = 128;
    }
    else if(n <= 672) //n = 640
    {
        if(m <= 2944)
            blk = n;
        else if(m <= 10496)
            blk = 256;
        else
            blk = 128;
    }
    else if(n <= 736) // n = 704
    {
        if(m <= 1664)
            blk = 128;
        else
            blk = 256;
    }
    else if(n <= 800) // n = 768
    {
        if(m <= 1664)
            blk = 128;
        else
            blk = 256;
    }
    else if(n <= 864) // n = 832
    {
        if(m <= 1664)
            blk = 128;
        else
            blk = 256;
    }
    else if(n <= 928) // n = 896
    {
        if(m <= 1408)
            blk = 128;
        else if(m <= 2432)
            blk = 384;
        else
            blk = 256;
    }
    else if(n <= 992) // n = 960
    {
        if(m <= 1216)
            blk = 128;
        else if(m <= 1664)
            blk = 384;
        else
            blk = 256;
    }
    else if(n <= 1088) // n = 1024
    {
        if(m <= 1920)
            blk = 128;
        else
            blk = 256;
    }
    else if(n <= 1216) // n = 1152
    {
        if(m <= 1920)
            blk = 128;
        else if(m <= 3200)
            blk = 384;
        else
            blk = 256;
    }
    else if(n <= 1344) // n = 1280
    {
        if(m <= 1920)
            blk = 128;
        else if(m <= 5376)
            blk = 320;
        else
            blk = 256;
    }
    else if(n <= 1472) // n = 1408
    {
        if(m <= 1920)
            blk = 128;
        else if(m <= 6400)
            blk = 320;
        else
            blk = 256;
    }
    else // n = 1536
    {
        if(m <= 1920)
            blk = 128;
        else if(m <= 6400)
            blk = 448;
        else
            blk = 256;
    }

    return blk;
}*/

/** This function returns the inner-inner block size. This has been tuned based on
    experiments with panel matrices; it is not expected to change a lot.
    (not tunable by user for now) **/
template <bool ISBATCHED>
inline rocblas_int get_2nd_innerBlkSize(rocblas_int m, rocblas_int n)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(n <= 72) // n = 16,32,48,64
        {
            if(m <= 64)
                blk = n;
            else
                blk = 16;
        }
        else if(n <= 88) // n = 80
        {
            if(m <= 55)
                blk = 64;
            else
                blk = 16;
        }
        else // n = 96,112,128,144,160,176,192,208,224,240,256,...
        {
            if(m <= 64)
                blk = 64;
            else
                blk = 16;
        }
    }

    else
    {
        if(n <= 72) // n = 16,32,48,64
        {
            blk = n;
        }
        else if(n <= 88) // n = 80
        {
            if(m <= 32)
                blk = 48;
            else if(m <= 64)
                blk = n;
            else if(m <= 352)
                blk = 16;
            else if(m <= 8960)
                blk = n;
            else
                blk = 16;
        }
        else if(n <= 104) // n = 96
        {
            if(m <= 32)
                blk = 48;
            else if(m <= 64)
                blk = n;
            else if(m <= 352)
                blk = 32;
            else if(m <= 4736)
                blk = n;
            else
                blk = 32;
        }
        else // n = 112,128,144,160,176,192,208,224,240,256,...
        {
            if(m < 64)
                blk = 64;
            else
                blk = 32;
        }
    }

    return blk;
}

template <bool PIVOT, typename T, typename U>
ROCSOLVER_KERNEL void getrf_check_singularity(const rocblas_int n,
                                              const rocblas_int j,
                                              const rocblas_int jb,
                                              U AA,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              rocblas_int* ipivA,
                                              const rocblas_int shiftP,
                                              const rocblas_stride strideP,
                                              rocblas_int* iipivA,
                                              const rocblas_int* iinfo,
                                              rocblas_int* info)
{
    int id = hipBlockIdx_y;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // update info (check singularity)
        if(tid == j && info[id] == 0 && iinfo[id] > 0)
            info[id] = iinfo[id] + j;

        if(PIVOT)
        {
            T orig;
            rocblas_int* iipiv = iipivA + id * jb;
            T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);

            // update ipiv (pivots in columns j : j+jb-1)
            if(tid >= j && tid < j + jb)
            {
                rocblas_int* ipiv = ipivA + id * strideP + shiftP;
                ipiv[tid] = iipiv[tid - j] + j;
            }

            // swap rows in columns 0 : j-1 and j+jb : n-1
            else
            {
                for(rocblas_int i = j; i < j + jb; ++i)
                {
                    rocblas_int exch = iipiv[i - j] + j - 1;

                    // will exchange rows i and exch if they are not the same
                    if(exch != i)
                    {
                        orig = A[i + tid * lda];
                        A[i + tid * lda] = A[exch + tid * lda];
                        A[exch + tid * lda] = orig;
                    }
                }
            }
        }
    }
}

template <bool BATCHED, bool STRIDED, bool PIVOT, typename T>
void rocsolver_getrf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx,
                                   size_t* size_iinfo,
                                   bool* optim_mem)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    rocblas_int dim = min(m, n);
    //    rocblas_int blk = getrf_get_blksize<ISBATCHED, PIVOT>(dim);
    rocblas_int blk = 18; //atoi(getenv("BLK"));

    if(blk == 1)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, PIVOT, T>(m, n, batch_count, size_scalars,
                                                           size_pivotval, size_pivotidx);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iinfo = 0;
        *optim_mem = true;
    }
    else
    {
        //////////////   check dimensions in getf2 and trsm ////////////////////////
        // requirements for calling GETF2 for the sub blocks
        rocsolver_getf2_getMemorySize<ISBATCHED, PIVOT, T>(m, n, batch_count, size_scalars,
                                                           size_pivotval, size_pivotidx);

        // to store info about singularity of sub blocks
        *size_iinfo = sizeof(rocblas_int) * batch_count;

        // extra workspace (for calling TRSM)
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, m, n, batch_count, size_work1,
                                         size_work2, size_work3, size_work4);

        // always allocate all required memory for TRSM optimal performance
        *optim_mem = true;
    }
}

template <bool BATCHED, bool STRIDED, bool PIVOT, typename T, typename U>
rocblas_status rocsolver_getrf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_int shiftP,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivotval,
                                        rocblas_int* pivotidx,
                                        rocblas_int* iinfo,
                                        bool optim_mem)
{
    rocblas_int* iipiv;
    hipMalloc(&iipiv, n * sizeof(rocblas_int) * batch_count);

    ROCSOLVER_ENTER("getrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, grid, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dim = min(m, n); // total number of pivots
    rocblas_int jb, jb1, jb2, blk, blk1, blk2;
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    blocks = (n - 1) / BLOCKSIZE + 1;
    grid = dim3(blocks, batch_count, 1);

    // size of outter blocks
    blk = getrf_get_blksize<ISBATCHED, PIVOT>(dim);
    //blk = atoi(getenv("BLK"));

    if(blk == 1)
        return rocsolver_getf2_template<ISBATCHED, PIVOT, T>(handle, m, n, A, shiftA, lda, strideA,
                                                             ipiv, shiftP, strideP, info, batch_count,
                                                             scalars, pivotval, pivotidx);

    //print_device_matrix("original",m,n,A,lda);
    //print_device_matrix(std::cout,"original",1,1,A,lda);

    // MAIN LOOP =====>
    for(rocblas_int j = 0; j < dim; j += blk)
    {
        jb = min(dim - j, blk); // number of columns/pivots in the block
        //        blk1 = get_1st_innerBlkSize(m - j, jb); // size of 1st-level inner blocks
        blk1 = jb; //atoi(getenv("BLK"));

        // LOOP FACTORIZING 1st-LEVEL INNER BLOCKS =====>
        for(rocblas_int k = 0; k < jb; k += blk1)
        {
            jb1 = min(jb - k, blk1); // number of columns/pivots in the inner block
            blk2 = get_2nd_innerBlkSize<ISBATCHED>(m - j - k, jb1); // size of 2nd-level inner block
            //blk2 = atoi(getenv("BLK"));

            // LOOP FACTORIZING 2nd-LEVEL INNER BLOCKS =====>
            for(rocblas_int kk = 0; kk < jb1; kk += blk2)
            {
                jb2 = min(jb1 - kk, blk2); // number of columns/pivots in the inner-inner block

                // factorize panel
                rocsolver_getf2_template<ISBATCHED, PIVOT, T>(
                    handle, m - j - k - kk, jb2, A, shiftA + idx2D(j + k + kk, j + k + kk, lda),
                    lda, strideA, iipiv, 0, jb2, iinfo, batch_count, scalars, pivotval, pivotidx);

                //print_device_matrix(std::cout,"after panel",1,1,A,lda);
                // adjust pivots, swap rows and check singularity
                hipLaunchKernelGGL((getrf_check_singularity<PIVOT, T>), grid, threads, 0, stream, n,
                                   j + k + kk, jb2, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                   iipiv, iinfo, info);

                //print_device_matrix(std::cout,"after pivoting",1,1,A,lda);
                // update trailing 2nd-level sub-block
                if(kk + jb2 < jb1)
                {
                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                        rocblas_diagonal_unit, jb2, jb1 - kk - jb2, &one, A,
                        shiftA + idx2D(j + k + kk, j + k + kk, lda), lda, strideA, A,
                        shiftA + idx2D(j + k + kk, j + k + kk + jb2, lda), lda, strideA,
                        batch_count, optim_mem, work1, work2, work3, work4);
                    //print_device_matrix(std::cout,"trsm2",1,1,A,lda);

                    if(kk + jb2 < m - j - k)
                    {
                        rocblasCall_gemm<BATCHED, STRIDED, T>(
                            handle, rocblas_operation_none, rocblas_operation_none,
                            m - j - k - kk - jb2, jb1 - kk - jb2, jb2, &minone, A,
                            shiftA + idx2D(j + k + kk + jb2, j + k + kk, lda), lda, strideA, A,
                            shiftA + idx2D(j + k + kk, j + k + kk + jb2, lda), lda, strideA, &one,
                            A, shiftA + idx2D(j + k + kk + jb2, j + k + kk + jb2, lda), lda,
                            strideA, batch_count, nullptr);
                        //print_device_matrix(std::cout,"gemm2",1,1,A,lda);
                    }
                }
            }
            // <===== (LOOP FACTORIZING 2nd-LEVEL INNER BLOCKS)

            // update trailing 1st-level sub-block
            if(k + jb1 < jb)
            {
                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                             rocblas_operation_none, rocblas_diagonal_unit, jb1,
                                             jb - k - jb1, &one, A,
                                             shiftA + idx2D(j + k, j + k, lda), lda, strideA, A,
                                             shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA,
                                             batch_count, optim_mem, work1, work2, work3, work4);
                //print_device_matrix(std::cout,"trsm1",1,1,A,lda);
                if(k + jb1 < m - j)
                {
                    rocblasCall_gemm<BATCHED, STRIDED, T>(
                        handle, rocblas_operation_none, rocblas_operation_none, m - j - k - jb1,
                        jb - k - jb1, jb1, &minone, A, shiftA + idx2D(j + k + jb1, j + k, lda), lda,
                        strideA, A, shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA, &one, A,
                        shiftA + idx2D(j + k + jb1, j + k + jb1, lda), lda, strideA, batch_count,
                        nullptr);
                    //print_device_matrix(std::cout,"gemm1",1,1,A,lda);
                }
            }
        }
        // <===== (LOOP FACTORIZING 1st-LEVEL INNER BLOCKS)

        // update trailing matrix
        if(j + jb < n)
        {
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_unit, jb,
                                         n - j - jb, &one, A, shiftA + idx2D(j, j, lda), lda,
                                         strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                                         batch_count, optim_mem, work1, work2, work3, work4);
            //print_device_matrix(std::cout,"trsm",1,1,A,lda);

            if(j + jb < m)
            {
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                    jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, A,
                    shiftA + idx2D(j, j + jb, lda), lda, strideA, &one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, nullptr);
                //print_device_matrix(std::cout,"gemm",1,1,A,lda);
            }
        }
    }
    // <===== (MAIN LOOP)

    //print_device_matrix(std::cout,"final",1,1,A,lda);
    //print_device_matrix("final",m,n,A,lda);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
