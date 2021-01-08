/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver.h"

template <typename U>
__global__ void getrf_check_singularity(const rocblas_int n,
                                        const rocblas_int j,
                                        rocblas_int* ipivA,
                                        const rocblas_int shiftP,
                                        const rocblas_stride strideP,
                                        const rocblas_int* iinfo,
                                        rocblas_int* info,
                                        const int pivot)
{
    int id = hipBlockIdx_y;
    rocblas_int* ipiv;

    if(info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;

    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n && pivot)
    {
        ipiv = ipivA + id * strideP + shiftP;
        ipiv[tid] += j;
    }
}


template <bool ISBATCHED, std::enable_if_t<!ISBATCHED, int> = 0> 
rocblas_int get_blksize(rocblas_int dim)
{
    char *b;
    b = std::getenv("BLK");
    int i = atoi(b);
    return i;
/*    rocblas_int i;
    std::vector<rocblas_int> size{GETRF_BLKSIZES_NORMAL};
    std::vector<rocblas_int> intervals{GETRF_INTERVALS_NORMAL};

    for(i=0;i<GETRF_NUM_INTERVALS_NORMAL;++i) 
    {
        if(dim < intervals[i]) 
            break;
    }

    return size[i];*/
}

template <bool ISBATCHED, std::enable_if_t<ISBATCHED, int> = 0> 
rocblas_int get_blksize(rocblas_int dim)
{
    char *b;
    b = std::getenv("BLK");
    int i = atoi(b);
    return i;
/*    rocblas_int i;
    std::vector<rocblas_int> size{GETRF_BLKSIZES_BATCH};
    std::vector<rocblas_int> intervals{GETRF_INTERVALS_BATCH};

    for(i=0;i<GETRF_NUM_INTERVALS_BATCH;++i) 
    {
        if(dim < intervals[i]) 
            break;
    }

    return size[i];*/
}

template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_getrf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx,
                                   size_t* size_iinfo)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iinfo = 0;
        return;
    }

    rocblas_int dim = min(m,n);
    rocblas_int blk = get_blksize<ISBATCHED>(dim);
    
    if(blk == 1)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, T, S>(m, n, batch_count, size_scalars, size_work,
                                                       size_pivotval, size_pivotidx);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iinfo = 0;
    }
    else
    {
        // requirements for calling GETF2 for the sub blocks
        rocsolver_getf2_getMemorySize<ISBATCHED, T, S>(m, blk, batch_count, size_scalars, size_work,
                                                       size_pivotval, size_pivotidx);

        // to store info about singularity of sub blocks
        *size_iinfo = sizeof(rocblas_int) * batch_count;

        // extra workspace (for calling TRSM)
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, blk, n - blk, batch_count, size_work1,
                                         size_work2, size_work3, size_work4);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
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
                                        const rocblas_int pivot,
                                        T* scalars,
                                        rocblas_index_value_t<S>* work,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivotval,
                                        rocblas_int* pivotidx,
                                        rocblas_int* iinfo,
                                        bool optim_mem)
{
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksPivot;
    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridPivot;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if the matrix is small, use the unblocked (level-2-blas) variant of the
    // algorithm
//    if(m < GETRF_GETF2_SWITCHSIZE || n < GETRF_GETF2_SWITCHSIZE)
//        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
//                                                      shiftP, strideP, info, batch_count, pivot,
//                                                      scalars, work, pivotval, pivotidx);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dim = min(m, n); // total number of pivots
    rocblas_int jb, sizePivot;



//    rocblas_int mik = rocblas_int(round(0.7320508*dim));
//   rocblas_int mik = 64;
    rocblas_int blk = get_blksize<ISBATCHED>(dim);

//    std::vector<int> vals{3,   10,   16,   25,   34,   45,   58,   72,   88,   105,   124,   144,   166,   189,   213,   240,   267,   296, 327,   359,   393,   428,   465,   503,   543,   584,   626,   671,   716,   763,   812,   862,   914,   967,   1022, 1078,   1136,   1195,   1255,   1318,   1381,   1446,   1513,   1581,   1651,   1722,   1795,   1869,   1945,   2022, 2100,   2181,   2262,   2345,   2430,   2516,   2604,   2693,   2784,   2876,   2970,   3065,   3161,   3260,   3359, 3460,   3563,   3667,   3773,   3880,   3989,   4099,   4211,   4324,   4438,   4555,   4672,   4791,   4912,   5034, 5158,   5283,   5410,   5538,   5668,   5799,   5931,   6066,   6201,   6338,   6477,   6617,   6759,   6902,   7047, 7193,   7341,   7490,   7640,   7793,   7946,   8101,   8258,   8416,   8576,   8737,   8900,   9064,   9230,   9397, 9565,   9736,   9907,   10080,   10255,   10431,   10609,   10788,   10969,   11151,   11335,   11520,   11706, 11895,   12084,   12275};

//printf("blk: %d\n",blk);

    if(blk == 1)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      shiftP, strideP, info, batch_count, pivot,
                                                      scalars, work, pivotval, pivotidx);

//    for(rocblas_int j = 0; j < dim; j += GETRF_GETF2_SWITCHSIZE)
    for(rocblas_int j = 0; j < dim; j += blk)//dim
    {
        // Factor diagonal and subdiagonal blocks
//        jb = min(dim - j, GETRF_GETF2_SWITCHSIZE); // number of columns in the block
        jb = min(dim - j, blk); // number of columns in the block
        hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
        rocsolver_getf2_template<ISBATCHED, T>(handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                               strideA, ipiv, shiftP + j, strideP, iinfo,
                                               batch_count, pivot, scalars, work, pivotval, pivotidx);

        // adjust pivot indices and check singularity
        sizePivot = min(m - j, jb); // number of pivots in the block
        blocksPivot = (sizePivot - 1) / BLOCKSIZE + 1;
        gridPivot = dim3(blocksPivot, batch_count, 1);
        hipLaunchKernelGGL(getrf_check_singularity<U>, gridPivot, threads, 0, stream, sizePivot, j,
                           ipiv, shiftP + j, strideP, iinfo, info, pivot);

        // apply interchanges to columns 1 : j-1
        if(pivot)
            rocsolver_laswp_template<T>(handle, j, A, shiftA, lda, strideA, j + 1, j + jb, ipiv,
                                        shiftP, strideP, 1, batch_count);

        if(j + jb < n)
        {
            if(pivot)
            {
                // apply interchanges to columns j+jb : n
                rocsolver_laswp_template<T>(handle, (n - j - jb), A, shiftA + idx2D(0, j + jb, lda),
                                            lda, strideA, j + 1, j + jb, ipiv, shiftP, strideP, 1,
                                            batch_count);
            }

            // compute block row of U
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_unit, jb,
                                         (n - j - jb), &one, A, shiftA + idx2D(j, j, lda), lda,
                                         strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                                         batch_count, optim_mem, work1, work2, work3, work4);

            // update trailing submatrix
            if(j + jb < m)
            {
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                    jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, A,
                    shiftA + idx2D(j, j + jb, lda), lda, strideA, &one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, nullptr);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
