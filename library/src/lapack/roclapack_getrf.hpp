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

template <typename T, typename U>
ROCSOLVER_KERNEL void getrf_panelgemm(const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,
                                      U AA,
                                      const rocblas_int shift,
                                      const rocblas_int shiftA,
                                      const rocblas_int shiftB,
                                      const rocblas_int shiftC,
                                      const rocblas_int lda,
                                      const rocblas_stride stride)
{
    // indices
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int ty = hipThreadIdx_y;
    rocblas_int bdimx = hipBlockDim_x;
    rocblas_int bdimy = hipBlockDim_y;
    rocblas_int i = hipBlockIdx_x * bdimx + tx;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shift + shiftA, stride);
    T* B = load_ptr_batch(AA, bid, shift + shiftB, stride);
    T* C = load_ptr_batch(AA, bid, shift + shiftC, stride);

    // shared mem and local arrays
    extern __shared__ double lmem[];
    T* a = (T*)lmem;
    T* b = (T*)lmem + bdimx * k;
    T c;

    // read block A into shared mem (row major order)
    for(int j = ty; j < k; j += bdimy)
        a[tx * k + j] = A[i + j * lda];

    // for each column of B and C
    for(int j = ty; j < n; j += bdimy)
    {
        // read b and c
        for(int ii = tx; ii < k; ii += bdimx)
            b[ii + j * k] = B[ii + j * lda];
        c = C[i + j * lda];
        __syncthreads();

        // update c
        for(int kk = 0; kk < k; ++kk)
            c -= a[tx * k + kk] * b[kk + j * k];

        // write back to global memory
        C[i + j * lda] = c;
    }
}

/** This function returns the outer block size based on defined variables
    tunable by the user (defined in ideal_sizes.hpp) **/
template <bool ISBATCHED>
rocblas_int getrf_get_blksize(rocblas_int dim, const bool pivot)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(pivot)
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
        if(pivot)
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
    (not tunable by the user for now) **/
template <bool ISBATCHED>
inline rocblas_int getrf_get_innerBlkSize(rocblas_int m, rocblas_int n, const bool pivot)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(pivot)
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
            if(n <= 40) // n = 16,32
            {
                if(m <= 256)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 72) // n = 48,64
            {
                if(m <= 45)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 120) // n = 80,96,112
            {
                if(m <= 51)
                    blk = 48;
                else
                    blk = 16;
            }
            else if(n <= 152) // n = 128,144
            {
                if(m <= 256)
                    blk = 32;
                else
                    blk = 16;
            }
            else // n = 160,176,192,208,224,240,256,...
            {
                if(m <= 304)
                    blk = 32;
                else
                    blk = 16;
            }
        }
    }

    else
    {
        if(pivot)
        {
            if(n <= 40)
            {
                if(m <= 10240)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 56)
            {
                if(m <= 64)
                    blk = 16;
                else if(m <= 5760)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 72)
            {
                if(m <= 96)
                    blk = 16;
                else if(m <= 4224)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 88)
            {
                if(m <= 96)
                    blk = 16;
                else if(m <= 3776)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 104)
            {
                if(m <= 96)
                    blk = 16;
                else if(m <= 3200)
                    blk = n;
                else if(m <= 6784)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 120)
            {
                if(m <= 128)
                    blk = 16;
                else if(m <= 2880)
                    blk = n;
                else if(m <= 6272)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 136)
            {
                if(m <= 128)
                    blk = 16;
                else if(m <= 2880)
                    blk = n;
                else if(m <= 6784)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 152)
            {
                if(m < 128)
                    blk = 16;
                else if(m < 176)
                    blk = 32;
                else if(m <= 2112)
                    blk = n;
                else if(m <= 8064)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 168)
            {
                if(m < 128)
                    blk = 16;
                else if(m < 176)
                    blk = 32;
                else if(m <= 1568)
                    blk = n;
                else if(m <= 10496)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 184)
            {
                if(m < 128)
                    blk = 16;
                else if(m < 208)
                    blk = 32;
                else if(m <= 1312)
                    blk = n;
                else if(m <= 2016)
                    blk = 96;
                else if(m <= 8960)
                    blk = 32;
                else
                    blk = 16;
            }
            else if(n <= 200)
            {
                if(m < 128)
                    blk = 16;
                else if(m < 256)
                    blk = 32;
                else if(m <= 1024)
                    blk = n;
                else if(m <= 1888)
                    blk = 96;
                else
                    blk = 32;
            }
            else if(n <= 216)
            {
                if(m < 128)
                    blk = 16;
                else if(m < 256)
                    blk = 32;
                else if(m <= 1376)
                    blk = n;
                else
                    blk = 32;
            }
            else if(n <= 232)
            {
                if(m < 128)
                    blk = 16;
                else if(m <= 256)
                    blk = 32;
                else if(m <= 928)
                    blk = n;
                else if(m <= 1312)
                    blk = 128;
                else
                    blk = 32;
            }
            else if(n <= 248)
            {
                if(m < 128)
                    blk = 16;
                else if(m <= 256)
                    blk = 32;
                else if(m <= 800)
                    blk = n;
                else if(m <= 1568)
                    blk = 128;
                else
                    blk = 32;
            }
            else if(n <= 264)
            {
                if(m < 128)
                    blk = 16;
                else if(m <= 320)
                    blk = 32;
                else if(m <= 1440)
                    blk = 128;
                else
                    blk = 32;
            }
            else
            {
                if(m < 128)
                    blk = 16;
                else
                    blk = 32;
            }
        }
        else
        {
            if(n <= 56) // n = 16,32,48
            {
                blk = n;
            }
            else if(n <= 72) // n = 64
            {
                if(m <= 8192)
                    blk = n;
                else
                    blk = 32;
            }
            else if(n <= 88) // n = 80
            {
                if(m <= 64)
                    blk = 64;
                else if(m <= 2048)
                    blk = 16;
                else if(m <= 4992)
                    blk = n;
                else
                    blk = 16;
            }
            else if(n <= 104) // n = 96
            {
                if(m <= 64)
                    blk = 64;
                else
                    blk = 32;
            }
            else if(n <= 152) // n = 112,128,144
            {
                if(m < 64)
                    blk = 80;
                else
                    blk = 32;
            }
            else // n = 160,176,192,208,224,240,256,...
            {
                if(m < 64)
                    blk = 96;
                else
                    blk = 32;
            }
        }
    }

    return blk;
}

/** This kernel updates the chosen pivots, checks singularity and
    interchanges rows all at once (pivoting + laswp)**/
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
                                              rocblas_int* info)
{
    int id = hipBlockIdx_y;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // update info (check singularity)
        //        if(tid == j && info[id] == 0 && iinfo[id] > 0)
        //            info[id] = iinfo[id] + j;

        //        T orig;
        //        rocblas_int* iipiv = iipivA + id * jb;
        T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
        rocblas_int* ipiv = ipivA + id * strideP + shiftP;

        // update ipiv (pivots in columns j : j+jb-1)
        if(tid >= j && tid < j + jb)
        {
            //            rocblas_int* ipiv = ipivA + id * strideP + shiftP;
            //            ipiv[tid] = iipiv[tid - j] + j;
        }

        // swap rows in columns 0 : j-1 and j+jb : n-1
        else
        {
            for(rocblas_int i = j; i < j + jb; ++i)
            {
                //                rocblas_int exch = iipiv[i - j] + j - 1;
                rocblas_int exch = ipiv[i] - 1;

                // will exchange rows i and exch if they are not the same
                if(exch != i)
                    swap(A[i + tid * lda], A[exch + tid * lda]);
            }
        }
    }
}

/** non-pivoting version **/
template <typename T>
ROCSOLVER_KERNEL void
    getrf_npvt_check_singularity(const rocblas_int j, const rocblas_int* iinfo, rocblas_int* info)
{
    int id = hipBlockIdx_y;

    // update info (check singularity)
    if(info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
}

template <typename T, typename U>
ROCSOLVER_KERNEL void getrf_row_permutate(const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int offset,
                                          const rocblas_int blk,
                                          U AA,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride stride,
                                          rocblas_int* pividx)
{
    int id = hipBlockIdx_z;
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int bdx = hipBlockDim_x;
    int j = hipBlockIdx_y * hipBlockDim_y + ty;
    if(j >= offset)
        j += blk;

    if(j < n)
    {
        // batch instance
        T* A = load_ptr_batch(AA, id, shiftA, stride);
        rocblas_int* piv = pividx + id * m;

        // shared mem for temporary values
        extern __shared__ double lmem[];
        T* temp = (T*)lmem;

        // do permutations in parallel (each tx perform a row swap)
        rocblas_int idx1 = piv[tx];
        rocblas_int idx2 = piv[idx1];
        temp[tx + ty * bdx] = A[idx1 + j * lda];
        A[idx1 + j * lda] = A[idx2 + j * lda];
        __syncthreads();

        // copy temp results back to A
        A[tx + j * lda] = temp[tx + ty * bdx];
    }
}

/** Return the sizes of the different workspace arrays **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getrf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const bool pivot,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx,
                                   size_t* size_iipiv,
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
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    rocblas_int dim = min(m, n);
    rocblas_int blk = getrf_get_blksize<ISBATCHED>(dim, pivot);

    if(blk == 1)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, n, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }
    else
    {
        // requirements for calling GETF2 for the sub blocks
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, blk, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx);

        // to store info about singularity and pivots of sub blocks
        *size_iinfo = sizeof(rocblas_int) * batch_count;
        *size_iipiv = pivot ? m * sizeof(rocblas_int) * batch_count : 0;

        // extra workspace (for calling TRSM)
        // (Note: TRSM workspace size is less than expected when the number of rows is multiple of 128.
        //  For this reason, when trying to set up a workspace that fits all the TRSM calls for m <= blk,
        //  blk cannot be multiple of 128.)
        rocblas_int mm = (blk % 128 != 0) ? blk : blk + 1;
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, mm, n, batch_count, size_work1,
                                         size_work2, size_work3, size_work4);

        // always allocate all required memory for TRSM optimal performance
        *optim_mem = true;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status getrf_panelLU(rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             U A,
                             const rocblas_int shiftA,
                             const rocblas_int lda,
                             const rocblas_stride strideA,
                             rocblas_int* ipiv,
                             rocblas_int* info,
                             const rocblas_int batch_count,
                             T* scalars,
                             void* work1,
                             void* work2,
                             void* work3,
                             void* work4,
                             T* pivotval,
                             rocblas_int* pivotidx,
                             const bool optim_mem,
                             const bool pivot)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int blocks = pivot ? (n - 1) / 64 + 1 : 1;
    dim3 threads((pivot ? 64 : 1), 1, 1);
    dim3 grid(blocks, batch_count, 1);

    if(n <= 128)
    {
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      0, n, info, batch_count, scalars, pivotval,
                                                      pivotidx, pivot);
    }

    else
    {
        rocblas_int nin = n / 2;
        rocblas_int nout = n - nin;

        getrf_panelLU<BATCHED, STRIDED, T>(handle, m, nin, A, shiftA, lda, strideA, ipiv, info,
                                           batch_count, scalars, work1, work2, work3, work4,
                                           pivotval, pivotidx, optim_mem, pivot);

        if(pivot)
            // adjust pivots, swap rows and check singularity
            hipLaunchKernelGGL(getrf_check_singularity<T>, grid, threads, 0, stream, n, 0, nin, A,
                               shiftA, lda, strideA, ipiv, 0, n, ipiv, info, info);
        //        else
        //            // check singularity
        //            hipLaunchKernelGGL(getrf_npvt_check_singularity<T>, grid, threads, 0, stream, 0,
        //                               info, info);

        // update trailing sub-block
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                     rocblas_operation_none, rocblas_diagonal_unit, nin, nout, &one,
                                     A, shiftA, lda, strideA, A, shiftA + idx2D(0, nin, lda), lda,
                                     strideA, batch_count, optim_mem, work1, work2, work3, work4);

        rocblasCall_gemm<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, rocblas_operation_none, m - nin, nout, nin, &minone, A,
            shiftA + idx2D(nin, 0, lda), lda, strideA, A, shiftA + idx2D(0, nin, lda), lda, strideA,
            &one, A, shiftA + idx2D(nin, nin, lda), lda, strideA, batch_count, nullptr);
        /*                    dimx = 128;
                    dimy = 8;
                    blocks = (m-j-k-jb1-1)/dimx + 1;
                    gridGemm = dim3(blocks,1,batch_count);
                    thdsGemm = dim3(dimx,dimy,1);
                    size_t lmemsize = jb1*(dimx+dimy)*sizeof(T);
                    hipLaunchKernelGGL((getrf_panelgemm<T>), gridGemm, thdsGemm, lmemsize, stream,
                                        m-j-k-jb1, jb-k-jb1, jb1, A, shiftA, idx2D(j + k + jb1, j + k, lda),
                                        idx2D(j + k, j + k + jb1, lda), idx2D(j + k + jb1, j + k + jb1, lda),
                                        lda, strideA);*/

        getrf_panelLU<BATCHED, STRIDED, T>(
            handle, m - nin, nout, A, shiftA + idx2D(nin, nin, lda), lda, strideA, ipiv + nin, info,
            batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, optim_mem, pivot);

        if(pivot)
            // adjust pivots, swap rows and check singularity
            hipLaunchKernelGGL(getrf_check_singularity<T>, grid, threads, 0, stream, n, nin, nin, A,
                               shiftA, lda, strideA, ipiv, 0, n, ipiv + nin, info, info);
        else
            // check singularity
            hipLaunchKernelGGL(getrf_npvt_check_singularity<T>, grid, threads, 0, stream, nin, info,
                               info);
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
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
                                        rocblas_int* iipiv,
                                        rocblas_int* iinfo,
                                        const bool optim_mem,
                                        const bool pivot)
{
    ROCSOLVER_ENTER("getrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    rocblas_int dim = min(m, n);
    rocblas_int blocks;
    dim3 grid, threads;

    // quick return if no dimensions
    if(m == 0 || n == 0)
    {
        blocks = (batch_count - 1) / BLOCKSIZE + 1;
        grid = dim3(blocks, 1, 1);
        threads = dim3(BLOCKSIZE, 1, 1);
        hipLaunchKernelGGL(reset_info, grid, threads, 0, stream, info, batch_count, 0);
        return rocblas_status_success;
    }

    // size of outer blocks
    rocblas_int blk = getrf_get_blksize<ISBATCHED>(dim, pivot);

    if(blk == 1)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      shiftP, strideP, info, batch_count, scalars,
                                                      pivotval, pivotidx, pivot);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    // number of threads for the check singularity kernel
    //    rocblas_int singular_thds;
    //    if(n < 4096)
    //        singular_thds = 64;
    //    else
    //        singular_thds = 1024;

    //    if(pivot)
    //    {
    //        // initialize permutation array
    //        blocks = (m - 1) / 256 + 1;
    //        threads = dim3(256,1,1);
    //        grid = dim3(blocks,batch_count,1);
    //        hipLaunchKernelGGL(permut_init<T>, grid, threads, 0, stream, m, iipiv);
    //    }

    // prepare kernels
    rocblas_int jb, jb1, blk1, dimx, dimy;
    size_t lmemsize;
    //    blocks = pivot ? (n - 1) / singular_thds + 1 : 1;
    //    threads = dim3((pivot ? singular_thds : 1), 1, 1);
    //    grid = dim3(blocks, batch_count, 1);
    //dim3 gridGemm, thdsGemm;
    //rocblas_int dimx, dimy;

    //print_device_matrix(std::cout,"A orig",m,n,A,lda,lda*n,1);
    //print_device_matrix(std::cout,"iipiv orig",1,m,iipiv,1);

    // MAIN LOOP =====>
    for(rocblas_int j = 0; j < dim; j += blk)
    {
        jb = min(dim - j, blk); // number of columns/pivots in the block
        blk1 = getrf_get_innerBlkSize<ISBATCHED>(m - j, jb, pivot); // size of inner blocks

        // LOOP FACTORIZING INNER BLOCKS =====>
        for(rocblas_int k = 0; k < jb; k += blk1)
        {
            jb1 = min(jb - k, blk1); // number of columns/pivots in the inner block

            // factorize inner block panel
            rocsolver_getf2_template<ISBATCHED, T>(
                handle, m - j - k, jb1, A, shiftA + idx2D(j + k, j + k, lda), lda, strideA, ipiv,
                shiftP + j + k, strideP, info, batch_count, scalars, pivotval, pivotidx, pivot,
                j + k, iipiv, m);
            //print_device_matrix(std::cout,"A after tf2",m,n,A,lda,lda*n,1);
            //print_device_matrix(std::cout,"ipiv after tf2",1,m,ipiv,1,m,1);
            //print_device_matrix(std::cout,"permut after tf2",1,m,iipiv,1,m,1);

            //            rocsolver_getf2_template<ISBATCHED, T>(
            //                handle, m - j - k, jb1, A, shiftA + idx2D(j + k, j + k, lda), lda, strideA, iipiv,
            //                0, jb1, iinfo, batch_count, scalars, pivotval, pivotidx, pivot);
            //            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j - k, jb1, A, shiftA + idx2D(j + k, j + k, lda), lda, strideA, iipiv,
            //                                        iinfo, batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, optim_mem, pivot);

            if(pivot)
            {
                dimx = jb1;
                dimy = 1024 / dimx;
                blocks = (n - 1 - jb1) / dimy + 1;
                grid = dim3(1, blocks, batch_count);
                threads = dim3(dimx, dimy, 1);
                lmemsize = dimx * dimy * sizeof(T);

                // swap rows
                hipLaunchKernelGGL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, m, n,
                                   j + k, jb1, A, shiftA + idx2D(j + k, 0, lda), lda, strideA, iipiv);

                //print_device_matrix(std::cout,"A after permutate",m,n,A,lda,lda*n,1);
            }
            //                hipLaunchKernelGGL(getrf_check_singularity<T>, grid, threads, 0, stream, n, j + k,
            //                                   jb1, A, shiftA, lda, strideA, ipiv, shiftP, strideP, iipiv,
            //                                   iinfo, info);
            //            else
            //                // check singularity
            //                hipLaunchKernelGGL(getrf_npvt_check_singularity<T>, grid, threads, 0, stream, j + k,
            //                                   iinfo, info);

            // update trailing sub-block
            if(k + jb1 < jb)
            {
                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                             rocblas_operation_none, rocblas_diagonal_unit, jb1,
                                             jb - k - jb1, &one, A,
                                             shiftA + idx2D(j + k, j + k, lda), lda, strideA, A,
                                             shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA,
                                             batch_count, optim_mem, work1, work2, work3, work4);

                if(k + jb1 < m - j)
                {
                    rocblasCall_gemm<BATCHED, STRIDED, T>(
                        handle, rocblas_operation_none, rocblas_operation_none, m - j - k - jb1,
                        jb - k - jb1, jb1, &minone, A, shiftA + idx2D(j + k + jb1, j + k, lda), lda,
                        strideA, A, shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA, &one, A,
                        shiftA + idx2D(j + k + jb1, j + k + jb1, lda), lda, strideA, batch_count,
                        nullptr);
                    /*                    dimx = 128;
                    dimy = 8;
                    blocks = (m-j-k-jb1-1)/dimx + 1;
                    gridGemm = dim3(blocks,1,batch_count);
                    thdsGemm = dim3(dimx,dimy,1);
                    size_t lmemsize = jb1*(dimx+dimy)*sizeof(T);
                    hipLaunchKernelGGL((getrf_panelgemm<T>), gridGemm, thdsGemm, lmemsize, stream,
                                        m-j-k-jb1, jb-k-jb1, jb1, A, shiftA, idx2D(j + k + jb1, j + k, lda),
                                        idx2D(j + k, j + k + jb1, lda), idx2D(j + k + jb1, j + k + jb1, lda),
                                        lda, strideA);*/
                }
            }
        }
        // <===== (LOOP FACTORIZING INNER BLOCKS)

        // update trailing matrix
        if(j + jb < n)
        {
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_unit, jb,
                                         n - j - jb, &one, A, shiftA + idx2D(j, j, lda), lda,
                                         strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                                         batch_count, optim_mem, work1, work2, work3, work4);

            if(j + jb < m)
            {
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                    jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, A,
                    shiftA + idx2D(j, j + jb, lda), lda, strideA, &one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count, nullptr);
                /*                dimx = 8;
                dimy = 128;
                blocks = (m-j-jb-1)/dimx + 1;
                gridGemm = dim3(blocks,1,batch_count);
                thdsGemm = dim3(dimx,dimy,1);
                size_t lmemsize = jb*(dimx+dimy)*sizeof(T);
                hipLaunchKernelGGL((getrf_panelgemm<T>), gridGemm, thdsGemm, 0, stream,
                                    m-j-jb, n-j-jb, jb, A, shiftA, idx2D(j + jb, j, lda), idx2D(j, j + jb, lda),
                                    idx2D(j + jb, j + jb, lda), lda, strideA);*/
            }
        }
    }
    // <===== (MAIN LOOP)

    //print_device_matrix(std::cout,"final A",m,n,A,lda);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
