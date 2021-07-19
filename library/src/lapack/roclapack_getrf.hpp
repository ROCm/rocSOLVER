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

/*template <typename U>
ROCSOLVER_KERNEL void getrf_check_singularity(const rocblas_int n,
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
}*/
template <typename T, typename U>
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
                                              rocblas_int* info,
                                              const int pivot)
{
    int id = hipBlockIdx_y;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // update info (check singularity)
        if(tid == j && info[id] == 0 && iinfo[id] > 0)
                info[id] = iinfo[id] + j;

        if(pivot)
        {
            T orig;
            rocblas_int *iipiv = iipivA + id * jb;
            T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);

            // update ipiv (pivots in columns j : j+jb-1)
            if(tid >= j && tid < j + jb)
            {
                rocblas_int *ipiv = ipivA + id * strideP + shiftP;
                ipiv[tid] = iipiv[tid-j] + j;
            }
            
            // swap rows in columns 0 : j-1 and j+jb : n-1
            else
            {
                for(rocblas_int i = j; i < j+jb; ++i)
                {
                    rocblas_int exch = iipiv[i-j] + j - 1;

                    // will exchange rows i and exch if they are not the same
                    if(exch != i)
                    {
                        orig = A[i+tid*lda];
                        A[i+tid*lda] = A[exch+tid*lda];
                        A[exch+tid*lda] = orig;
                    }
                }
            }
        }
    }
}

template <bool BATCHED, bool STRIDED, bool PIVOT, typename T, typename S>
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
                                   size_t* size_iinfo,
                                   bool* optim_mem)
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
        *optim_mem = true;
        return;
    }

    rocblas_int dim = min(m, n);
    rocblas_int blk = getrf_get_blksize<ISBATCHED, PIVOT>(dim);
//rocblas_int blk = atoi(getenv("BLK"));

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
        *optim_mem = true;
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

        // always allocate all required memory for TRSM optimal performance
        *optim_mem = true;
    }
}

template <bool BATCHED, bool STRIDED, bool PIVOT, typename T, typename S, typename U>
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

rocblas_int *iipiv;
hipMalloc(&iipiv,n*sizeof(rocblas_int)*batch_count);

    ROCSOLVER_ENTER("getrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

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

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dim = min(m, n); // total number of pivots
    rocblas_int jb, sizePivot;

    rocblas_int blk = getrf_get_blksize<ISBATCHED, PIVOT>(dim);
//rocblas_int blk = atoi(getenv("BLK"));

    if(blk == 1)
        return rocsolver_getf2_template<ISBATCHED, PIVOT, T>(handle, m, n, A, shiftA, lda, strideA,
                                                             ipiv, shiftP, strideP, info, batch_count,
                                                             scalars, work, pivotval, pivotidx);

    for(rocblas_int j = 0; j < dim; j += blk) //dim
    {
        // Factor diagonal and subdiagonal blocks
        jb = min(dim - j, blk); // number of columns in the block
        hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
//        rocsolver_getf2_template<ISBATCHED, PIVOT, T>(
//            handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, ipiv, shiftP + j,
//            strideP, iinfo, batch_count, scalars, work, pivotval, pivotidx);
        rocsolver_getf2_template<ISBATCHED, PIVOT, T>(
            handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, iipiv, 0,
            jb, iinfo, batch_count, scalars, work, pivotval, pivotidx);

        // adjust pivot indices and check singularity
//        sizePivot = min(m - j, jb); // number of pivots in the block
//        blocksPivot = (sizePivot - 1) / BLOCKSIZE + 1;
        blocksPivot = (n - 1) / BLOCKSIZE + 1;
        gridPivot = dim3(blocksPivot, batch_count, 1);
//        hipLaunchKernelGGL(getrf_check_singularity<U>, gridPivot, threads, 0, stream, sizePivot, j,
//                           ipiv, shiftP + j, strideP, iinfo, info, PIVOT);
        hipLaunchKernelGGL(getrf_check_singularity<T>, gridPivot, threads, 0, stream, n, j, jb,
                           A, shiftA, lda, strideA, ipiv, shiftP, strideP, iipiv, iinfo, info, PIVOT);

//        // apply interchanges to columns 1 : j-1
//        if(PIVOT)
//            rocsolver_laswp_template<T>(handle, j, A, shiftA, lda, strideA, j + 1, j + jb, ipiv,
//                                        shiftP, strideP, 1, batch_count);

        if(j + jb < n)
        {
//            if(PIVOT)
//            {
//                // apply interchanges to columns j+jb : n
//                rocsolver_laswp_template<T>(handle, (n - j - jb), A, shiftA + idx2D(0, j + jb, lda),
//                                            lda, strideA, j + 1, j + jb, ipiv, shiftP, strideP, 1,
//                                            batch_count);
//            }

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
