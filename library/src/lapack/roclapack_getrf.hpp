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
ROCSOLVER_KERNEL void getrf_gemm(const rocblas_int m,
                                 const rocblas_int n,
                                 const rocblas_int k,
                                 U MM,
                                 const rocblas_int shiftA,
                                 const rocblas_int shiftB,
                                 const rocblas_int shiftC,
                                 const rocblas_int ldim,
                                 const rocblas_stride stride)
{
    // indices
    int id = hipBlockIdx_z;
    int tx = hipThreadIdx_x;
    int j = hipThreadIdx_y;
    int bdx = hipBlockDim_x;

    // batch instance
    T* A = load_ptr_batch(MM, id, shiftA, stride);
    T* B = load_ptr_batch(MM, id, shiftB, stride);
    T* C = load_ptr_batch(MM, id, shiftC, stride);

    // shared mem setup
    extern __shared__ double lmem[];
    T* a = (T*)lmem;
    T* b = a + k * bdx;
    T c;

    // local row and column of the shared arrays
    a += tx * k;
    b += j * k;

    // read B into shared mem
    for(int i = tx; i < k; i += bdx)
        b[i] = B[i + j * ldim];

    // for each row of A and C (main loop)
    for(int i = tx; i < m; i += bdx)
    {
        // read a and c
        a[j] = A[i + j * ldim];
        c = C[i + j * ldim];
        __syncthreads();

        // update c
        for(int kk = 0; kk < k; ++kk)
            c -= a[kk] * b[kk];

        // write back to global memory
        C[i + j * ldim] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void getrf_trsm(const rocblas_int m,
                                 const rocblas_int n,
                                 U MM,
                                 const rocblas_int shiftA,
                                 const rocblas_int shiftB,
                                 const rocblas_int ldim,
                                 const rocblas_stride stride)
{
    int id = hipBlockIdx_z;
    int i = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int bdy = hipBlockDim_y;

    // batch instance
    T* A = load_ptr_batch(MM, id, shiftA, stride);
    T* B = load_ptr_batch(MM, id, shiftB, stride);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = (T*)lmem;
    T* a = b + m * bdy;
    int c;

    // local column of the shared array b
    b += ty * m;

    // copy triangular matrix to shared linear array
    // (only the elements below the diagonal)
    for(int j = ty; j < m; j += bdy)
    {
        if(i > j)
        {
            c = i - 1 + ((2 * m - 3 - j) * j) / 2;
            a[c] = A[i + j * ldim];
        }
    }

    // solve system (main loop)
    for(int j = ty; j < n; j += bdy)
    {
        // copy current right-hand-sides to shared array
        b[i] = B[i + j * ldim];
        __syncthreads();

        // solve for right-hand sides in shared memory
        for(int k = 0; k < m - 1; ++k)
        {
            c = i - 1 + ((2 * m - 3 - k) * k) / 2;
            if(i > k)
                b[i] -= a[c] * b[k];
            __syncthreads();
        }

        // move results back to global
        B[i + j * ldim] = b[i];
    }
}

/** Execute all permutations dictated by the panel factorization
    in parallel (concurrency by rows and columns) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void getrf_row_permutate(const rocblas_int n,
                                          const rocblas_int offset,
                                          const rocblas_int blk,
                                          U AA,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          rocblas_int* pividx,
                                          const rocblas_stride stridePI)
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
        T* A = load_ptr_batch(AA, id, shiftA, strideA);
        rocblas_int* piv = pividx + id * stridePI;

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

    /** TODO: We need to do especific tuning for batched and non-pivoting cases.
        Constants could go to ideal_sizes.hpp. Leaving here for now) **/
    rocblas_int N = 9;
    rocblas_int M = 10;
    rocblas_int intervalsN[9] = {24, 40, 56, 72, 104, 184, 200, 232, 248};
    rocblas_int intervalsM[10] = {128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 10240};
    rocblas_int size[11][10]
        = {{0, 0, 0, 0, 16, 16, 16, 16, 16, 16},    {0, 0, 16, 16, 16, 16, 16, 16, 16, 16},
           {0, 16, 8, 8, 8, 16, 16, 16, 16, 16},    {0, 8, 8, 8, 8, 8, 8, 8, 8, 8},
           {0, 0, 0, 16, 16, 16, 16, 16, 16, 16},   {0, 0, 0, 16, 16, 16, 16, 16, 16, 24},
           {0, 0, 16, 16, 16, 16, 24, 16, 16, 24},  {0, 0, 16, 16, 16, 16, 16, 16, 24, 32},
           {0, 16, 16, 16, 16, 16, 16, 24, 24, 32}, {0, 16, 16, 16, 16, 16, 16, 16, 24, 24},
           {0, 16, 16, 16, 16, 16, 16, 16, 16, 24}};

    //if(ISBATCHED)
    //{
    //   if(pivot)
    //    {
    //    }
    //    else
    //    {
    //    }
    //}
    //else
    //{
    //    if(pivot)
    //    {
    blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    //    }
    //    else
    //    {
    //    }
    //}

    if(blk == 0 || m < n)
        blk = n;

    return blk;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status getrf_panelLU(rocblas_handle handle,
                             const rocblas_int mm,
                             const rocblas_int nn,
                             const rocblas_int n,
                             U A,
                             const rocblas_int r_shiftA,
                             const rocblas_int lda,
                             const rocblas_stride strideA,
                             rocblas_int* ipiv,
                             const rocblas_int shiftP,
                             const rocblas_stride strideP,
                             rocblas_int* info,
                             const rocblas_int batch_count,
                             const bool pivot,
                             T* scalars,
                             void* work1,
                             void* work2,
                             void* work3,
                             void* work4,
                             const bool optim_mem,
                             T* pivotval,
                             rocblas_int* pivotidx,
                             const rocblas_int offset,
                             rocblas_int* permut_idx,
                             const rocblas_stride stridePI)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    // r_shiftA is the row where the panel-block starts,
    // the actual position of the panel-block in the matrix is:
    rocblas_int shiftA = r_shiftA + idx2D(0, offset, lda);

    //rocblas_int blk = atoi(getenv("BLK"));
    rocblas_int blk = getrf_get_innerBlkSize<ISBATCHED>(mm, nn, pivot);
    //rocblas_int trsm = atoi(getenv("TRSM"));

    rocblas_int jb;
    rocblas_int dimx, dimy, blocks;
    dim3 grid, threads;
    size_t lmemsize;

    //print_device_matrix(std::cout,"Orig",18,8,A,lda);
    for(rocblas_int k = 0; k < nn; k += blk)
    {
        jb = min(nn - k, blk); // number of columns/pivots in the inner block

        // factorize block
        rocsolver_getf2_template<ISBATCHED, T>(handle, mm - k, jb, A, shiftA + idx2D(k, k, lda),
                                               lda, strideA, ipiv, shiftP + k, strideP, info,
                                               batch_count, scalars, pivotval, pivotidx, pivot,
                                               offset + k, permut_idx, stridePI);

        //print_device_matrix(std::cout,"After tf2",18,8,A,lda);
        if(pivot)
        {
            dimx = jb;
            dimy = 1024 / dimx;
            blocks = (n - 1 - jb) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimx * dimy * sizeof(T);

            // swap rows
            hipLaunchKernelGGL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n,
                               offset + k, jb, A, r_shiftA + k, lda, strideA, permut_idx, stridePI);
        }
        //print_device_matrix(std::cout,"After permut",18,8,A,lda);

        // update trailing sub-block
        if(k + jb < nn)
        {
            //            if(trsm == 0)
            //            {
            //                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
            //                                             rocblas_operation_none, rocblas_diagonal_unit, jb,
            //                                             nn - k - jb, &one, A, shiftA + idx2D(k, k, lda), lda,
            //                                             strideA, A, shiftA + idx2D(k, k + jb, lda), lda, strideA,
            //                                             batch_count, optim_mem, work1, work2, work3, work4);
            //            }
            //            else
            //            {
            dimx = jb;
            dimy = min(nn - k - jb, 1024 / dimx);
            grid = dim3(1, 1, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = (jb * dimy + ((jb - 1) * jb) / 2) * sizeof(T);
            hipLaunchKernelGGL(getrf_trsm<T>, grid, threads, lmemsize, stream, jb, nn - k - jb, A,
                               shiftA + idx2D(k, k, lda), shiftA + idx2D(k, k + jb, lda), lda,
                               strideA);
            //            }
            //print_device_matrix(std::cout,"After trsm",18,8,A,lda);

            if(k + jb < mm)
            {
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, mm - k - jb,
                    nn - k - jb, jb, &minone, A, shiftA + idx2D(k + jb, k, lda), lda, strideA, A,
                    shiftA + idx2D(k, k + jb, lda), lda, strideA, &one, A,
                    shiftA + idx2D(k + jb, k + jb, lda), lda, strideA, batch_count, nullptr);
            }
            //print_device_matrix(std::cout,"After gemm",18,8,A,lda);
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status getrf_panelLU_recursive(rocblas_handle handle,
                                       const rocblas_int mm,
                                       const rocblas_int nn,
                                       const rocblas_int n,
                                       U A,
                                       const rocblas_int r_shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       rocblas_int* ipiv,
                                       const rocblas_int shiftP,
                                       const rocblas_stride strideP,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       const bool pivot,
                                       T* scalars,
                                       void* work1,
                                       void* work2,
                                       void* work3,
                                       void* work4,
                                       const bool optim_mem,
                                       T* pivotval,
                                       rocblas_int* pivotidx,
                                       const rocblas_int offset,
                                       rocblas_int* permut_idx,
                                       const rocblas_stride stridePI,
                                       const rocblas_int minblk)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    // r_shiftA is the row where the panel-block starts,
    // the actual position of the panel-block in the matrix is:
    rocblas_int shiftA = r_shiftA + idx2D(0, offset, lda);

    //    rocblas_int minblk = atoi(getenv("BLK"));
    //    rocblas_int minblk = getrf_get_blksize<ISBATCHED>(min(mm,nn), pivot);
    rocblas_int trsm = atoi(getenv("TRSM"));
    rocblas_int dimx, dimy, blocks;
    dim3 grid, threads;
    size_t lmemsize;

    /** BASE CASE **/
    if(nn <= minblk)
    {
        getrf_panelLU<BATCHED, STRIDED, T>(handle, mm, nn, n, A, r_shiftA, lda, strideA, ipiv,
                                           shiftP, strideP, info, batch_count, pivot, scalars,
                                           work1, work2, work3, work4, optim_mem, pivotval,
                                           pivotidx, offset, permut_idx, stridePI);
        //        rocsolver_getf2_template<ISBATCHED, T>(handle, mm, nn, A, shiftA, lda, strideA, ipiv,
        //                                               shiftP, strideP, info, batch_count, scalars, pivotval,
        //                                               pivotidx, pivot, offset, permut_idx, stridePI);
        //print_device_matrix(std::cout,"after tf2",8,8,A,lda);

        if(pivot)
        {
            dimx = nn;
            dimy = 1024 / dimx;
            blocks = (n - nn - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimx * dimy * sizeof(T);

            hipLaunchKernelGGL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n, offset,
                               nn, A, r_shiftA, lda, strideA, permut_idx, stridePI);
        }
        //print_device_matrix(std::cout,"after permute",8,8,A,lda);

        //print_device_matrix(std::cout,"after trsm",8,8,A,lda);
    }

    /** RECURSION **/
    else
    {
        // halve the dimension
        rocblas_int nin = nn / 2;
        rocblas_int nout = nn - nin;

        // factorize first half
        getrf_panelLU_recursive<BATCHED, STRIDED, T>(
            handle, mm, nin, n, A, r_shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count,
            pivot, scalars, work1, work2, work3, work4, optim_mem, pivotval, pivotidx, offset,
            permut_idx, stridePI, minblk);

        // update second half
        //        if(trsm == 0)
        //        {
        //            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
        //                                         rocblas_operation_none, rocblas_diagonal_unit, nin, nout,
        //                                         &one, A, shiftA, lda, strideA, A,
        //                                         shiftA + idx2D(0, nin, lda), lda, strideA, batch_count,
        //                                         optim_mem, work1, work2, work3, work4);
        //        }
        //        else
        //        {
        dimx = nin;
        dimy = min(nout, 1024 / dimx);
        grid = dim3(1, 1, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = (nin * dimy + ((nin - 1) * nin) / 2) * sizeof(T);
        hipLaunchKernelGGL(getrf_trsm<T>, grid, threads, lmemsize, stream, nin, nout, A, shiftA,
                           shiftA + idx2D(0, nin, lda), lda, strideA);
        //        }

        rocblasCall_gemm<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, rocblas_operation_none, mm - nin, nout, nin, &minone, A,
            shiftA + idx2D(nin, 0, lda), lda, strideA, A, shiftA + idx2D(0, nin, lda), lda, strideA,
            &one, A, shiftA + idx2D(nin, nin, lda), lda, strideA, batch_count, nullptr);
        /*        dimy = nout;
        dimx = min(mm-nin, 1024 / dimy);
        grid = dim3(1, 1, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = (nin * dimx + nin * nout) * sizeof(T);
        hipLaunchKernelGGL(getrf_gemm<T>, grid, threads, lmemsize, stream, mm - nin, nout, nin, A,
                           shiftA + idx2D(nin, 0, lda), shiftA + idx2D(0, nin, lda),
                           shiftA + idx2D(nin, nin, lda), lda, strideA);*/
        //print_device_matrix(std::cout,"after gemm",8,8,A,lda);

        // factorize second half
        getrf_panelLU_recursive<BATCHED, STRIDED, T>(
            handle, mm - nin, nout, n, A, r_shiftA + nin, lda, strideA, ipiv, shiftP + nin, strideP,
            info, batch_count, pivot, scalars, work1, work2, work3, work4, optim_mem, pivotval,
            pivotidx, offset + nin, permut_idx, stridePI, minblk);
    }

    return rocblas_status_success;
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
    //rocblas_int blk = dim; //atoi(getenv("BLK"));

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
        //        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, blk, pivot, batch_count, size_scalars,
        //                                                    size_pivotval, size_pivotidx);
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, n, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx);

        // to store info about singularity and pivots of sub blocks
        *size_iinfo = sizeof(rocblas_int) * batch_count;
        *size_iipiv = pivot ? m * sizeof(rocblas_int) * batch_count : 0;

        // extra workspace (for calling TRSM)
        // (Note: TRSM workspace size is less than expected when the number of rows is multiple of 128.
        //  For this reason, when trying to set up a workspace that fits all the TRSM calls for m <= blk,
        //  blk cannot be multiple of 128.)
        //        rocblas_int mm = (blk % 128 != 0) ? blk : blk + 1;
        rocblas_int mm = (m % 128 != 0) ? m : m + 1;
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, mm, n, batch_count, size_work1,
                                         size_work2, size_work3, size_work4);

        // always allocate all required memory for TRSM optimal performance
        *optim_mem = true;
    }
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
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, info, batch_count, 0);
        return rocblas_status_success;
    }

    // size of outer blocks
    rocblas_int blk = getrf_get_blksize<ISBATCHED>(dim, pivot);
    //    rocblas_int blk = dim; //atoi(getenv("BLK"));
    rocblas_int recur = atoi(getenv("RECUR"));

    if(blk == 1)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      shiftP, strideP, info, batch_count, scalars,
                                                      pivotval, pivotidx, pivot);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    if(recur == 1)
    {
        getrf_panelLU_recursive<BATCHED, STRIDED, T>(handle, m, n, n, A, shiftA, lda, strideA, ipiv,
                                                     shiftP, strideP, info, batch_count, pivot,
                                                     scalars, work1, work2, work3, work4, optim_mem,
                                                     pivotval, pivotidx, 0, iipiv, m, blk);
    }
    else
    {
        // constants to use when calling rocablas functions
        T one = 1; // constant 1 in host
        T minone = -1; // constant -1 in host

        rocblas_int jb, jb1, blk1, dimx, dimy;
        rocblas_int nextpiv, mm, nn;
        size_t lmemsize;

        // MAIN LOOP
        for(rocblas_int j = 0; j < dim; j += blk)
        {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            jb1 = min(jb - k, blk1); //number of columns/pivots in the current inner block

            // factorize inner block panel
            /*            rocsolver_getf2_template<ISBATCHED, T>(
                handle, m - j - k, jb1, A, shiftA + idx2D(j + k, j + k, lda), lda, strideA, ipiv,
                shiftP + j + k, strideP, info, batch_count, scalars, pivotval, pivotidx, pivot,
                j + k, iipiv, m);*/
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j - k, jb1, n, A, shiftA + j + k, lda,
                                               strideA, ipiv, shiftP + j + k, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j + k, iipiv, m);
            //print_device_matrix(std::cout,"after tf2",m,n,A,lda);

            if(pivot)
            {
                dimx = jb1;
                dimy = 1024 / dimx;
                blocks = (n - 1 - jb1) / dimy + 1;
                grid = dim3(1, blocks, batch_count);
                threads = dim3(dimx, dimy, 1);
                lmemsize = dimx * dimy * sizeof(T);

                // swap rows
                ROCSOLVER_LAUNCH_KERNEL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n, j + k,
                                   jb1, A, shiftA + idx2D(j + k, 0, lda), lda, strideA, iipiv, m);
            }

            // update trailing sub-block
            if(k + jb1 < jb)
            {
                /*                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                             rocblas_operation_none, rocblas_diagonal_unit, jb1,
                                             jb - k - jb1, &one, A,
                                             shiftA + idx2D(j + k, j + k, lda), lda, strideA, A,
                                             shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA,
                                             batch_count, optim_mem, work1, work2, work3, work4);*/
                dimx = jb1;
                dimy = min(jb - k - jb1, 1024 / jb1);
                grid = dim3(1, 1, batch_count);
                threads = dim3(dimx, dimy, 1);
                lmemsize = (jb1 * dimy + ((jb1 - 1) * jb1) / 2) * sizeof(T);
                hipLaunchKernelGGL(getrf_trsm<T>, grid, threads, lmemsize, stream, jb1,
                                   jb - k - jb1, A, shiftA + idx2D(j + k, j + k, lda),
                                   shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA);
                //print_device_matrix(std::cout,"after itrsm",m,n,A,lda);

                if(k + jb1 < m - j)
                {
                    rocblasCall_gemm<BATCHED, STRIDED, T>(
                        handle, rocblas_operation_none, rocblas_operation_none, m - j - k - jb1,
                        jb - k - jb1, jb1, &minone, A, shiftA + idx2D(j + k + jb1, j + k, lda), lda,
                        strideA, A, shiftA + idx2D(j + k, j + k + jb1, lda), lda, strideA, &one, A,
                        shiftA + idx2D(j + k + jb1, j + k + jb1, lda), lda, strideA, batch_count,
                        nullptr);
                    //print_device_matrix(std::cout,"after gemm",m,n,A,lda);
                }
            }
=======
=======
=======
>>>>>>> tuning
            jb = min(dim - j, blk); //number of columns/pivots in the current block
            nextpiv = j + jb; //posicion for the matrix update
            mm = m - nextpiv; //size for the matrix update
            nn = n - nextpiv; //size for the matrix update

            // factorize inner block panel
<<<<<<< HEAD
            //        if(recur == 0)
            //        {
>>>>>>> tuning
=======
>>>>>>> fix local trsm bug
=======
>>>>>>> tuning
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j, lda, strideA,
                                               ipiv, shiftP + j, strideP, info, batch_count, pivot,
                                               scalars, work1, work2, work3, work4, optim_mem,
                                               pivotval, pivotidx, j, iipiv, m);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        }
        else
        {
=======
        }
        else
        {
            //print_device_matrix(std::cout,"orig",m,n,A,lda);
>>>>>>> fix local trsm bug
            getrf_panelLU_recursive<BATCHED, STRIDED, T>(
                handle, m - j, jb, n, A, shiftA + j, lda, strideA, ipiv, shiftP + j, strideP, info,
                batch_count, pivot, scalars, work1, work2, work3, work4, optim_mem, pivotval,
                pivotidx, j, iipiv, m);
<<<<<<< HEAD
>>>>>>> add option for recursive panel fact
=======
>>>>>>> fix local trsm bug
        }

        // update trailing matrix
        if(nextpiv < n)
        {
<<<<<<< HEAD
            if(nn <= 128)
            {
                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                             rocblas_operation_none, rocblas_diagonal_unit, jb, nn,
                                             &one, A, shiftA + idx2D(j, j, lda), lda, strideA, A,
                                             shiftA + idx2D(j, nextpiv, lda), lda, strideA,
                                             batch_count, optim_mem, work1, work2, work3, work4);
            }
            else
=======
            //        }
            //        else
            //        {
            //            getrf_panelLU_recursive<BATCHED, STRIDED, T>(
            //                handle, m - j, jb, n, A, shiftA + j, lda, strideA, ipiv, shiftP + j, strideP, info,
            //                batch_count, pivot, scalars, work1, work2, work3, work4, optim_mem, pivotval,
            //                pivotidx, j, iipiv, m);
            //        }

            // update trailing matrix
            if(nextpiv < n)
>>>>>>> tuning
=======
            //                if(trsm == 0)
            //                {
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_unit, jb, nn,
                                         &one, A, shiftA + idx2D(j, j, lda), lda, strideA, A,
                                         shiftA + idx2D(j, nextpiv, lda), lda, strideA, batch_count,
                                         optim_mem, work1, work2, work3, work4);
            //                }
            //                else
            //                {
            //                    dimx = jb;
            //                    dimy = min(nn, 1024 / dimx);
            //                    grid = dim3(1, 1, batch_count);
            //                    threads = dim3(dimx, dimy, 1);
            //                    lmemsize = (jb * dimy + ((jb - 1) * jb) / 2) * sizeof(T);
            //                    hipLaunchKernelGGL(getrf_trsm<T>, grid, threads, lmemsize, stream, jb, nn, A,
            //                                       shiftA + idx2D(j, j, lda), shiftA + idx2D(j, nextpiv, lda),
            //                                       lda, strideA);
            //                }

            if(nextpiv < m)
>>>>>>> fix local trsm bug
=======

            // update trailing matrix
            if(nextpiv < n)
>>>>>>> tuning
            {
                if(nn > 256)
                {
                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                        rocblas_diagonal_unit, jb, nn, &one, A, shiftA + idx2D(j, j, lda), lda,
                        strideA, A, shiftA + idx2D(j, nextpiv, lda), lda, strideA, batch_count,
                        optim_mem, work1, work2, work3, work4);
                }
                else
                {
                    dimx = jb;
                    dimy = min(nn, 1024 / dimx);
                    grid = dim3(1, 1, batch_count);
                    threads = dim3(dimx, dimy, 1);
                    lmemsize = (jb * dimy + ((jb - 1) * jb) / 2) * sizeof(T);
                    hipLaunchKernelGGL(getrf_trsm<T>, grid, threads, lmemsize, stream, jb, nn, A,
                                       shiftA + idx2D(j, j, lda), shiftA + idx2D(j, nextpiv, lda),
                                       lda, strideA);
                }

                if(nextpiv < m)
                {
                    rocblasCall_gemm<BATCHED, STRIDED, T>(
                        handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb, &minone,
                        A, shiftA + idx2D(nextpiv, j, lda), lda, strideA, A,
                        shiftA + idx2D(j, nextpiv, lda), lda, strideA, &one, A,
                        shiftA + idx2D(nextpiv, nextpiv, lda), lda, strideA, batch_count, nullptr);
                }
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
