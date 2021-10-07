/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 *
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"
#include "rocsolver_small_kernels.hpp"

// number of threads for the iamax reduction kernel
#define IAMAX_THDS 1024

template <rocblas_int DIM, typename T, typename U>
ROCSOLVER_KERNEL void getf2_panel_kernel(const rocblas_int m,
                                         const rocblas_int n,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         rocblas_int* ipivA,
                                         const rocblas_int shiftP,
                                         const rocblas_stride strideP,
                                         rocblas_int* infoA,
                                         const rocblas_int batch_count,
                                         const bool pivot,
                                         const rocblas_int offset,
                                         rocblas_int* permut_idx,
                                         const rocblas_stride stride)
{
    using S = decltype(std::real(T{}));

    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int id = hipBlockIdx_z;
    const int bdx = hipBlockDim_x;
    const int bdy = hipBlockDim_y;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    rocblas_int *ipiv, *permut;
    if(pivot)
    {
        ipiv = load_ptr_batch<rocblas_int>(ipivA, id, shiftP, strideP);
        if(permut_idx)
            permut = permut_idx + id * stride;
    }
    rocblas_int* info = infoA + id;

    // shared memory (for communication between threads in group)
    extern __shared__ double lmem[];
    T* x = (T*)lmem;
    T* sval = x + bdx;
    T* y = sval + bdx;
    rocblas_int* sidx = (rocblas_int*)(y + n);
    __shared__ T val;

    // local variables
    T val1, val2, pivot_val;
    rocblas_int idx1, idx2, pivot_idx;
    int myinfo = 0; // to build info

    // init step: read column zero from A
    if(ty == 0)
    {
        val1 = (tx < m) ? A[tx] : 0;
        idx1 = tx;
        x[tx] = val1;
        sval[tx] = val1;
        sidx[tx] = idx1;
    }

    // main loop (for each pivot)
    for(rocblas_int k = 0; k < n; ++k)
    {
        // find pivot (maximum in column)
        __syncthreads();
        for(int i = bdx / 2; i > 0; i /= 2)
        {
            if(tx < i && ty == 0)
            {
                val2 = sval[tx + i];
                idx2 = sidx[tx + i];
                if(aabs<S>(val1) < aabs<S>(val2))
                {
                    sval[tx] = val1 = val2;
                    sidx[tx] = idx1 = idx2;
                }
            }
            __syncthreads();
        }
        pivot_idx = sidx[0]; //after reduction this is the index of max value
        pivot_val = sval[0];

        // check singularity and scale value for current column
        if(pivot_val == T(0))
        {
            pivot_idx = k;
            if(myinfo == 0)
                myinfo = k + 1;
        }
        else
            pivot_val = S(1) / pivot_val;

        // update ipiv
        if(tx == 0 && ty == 0)
            ipiv[k] = pivot_idx + 1 + offset;

        // update column k
        if(tx != pivot_idx)
        {
            pivot_val *= x[tx];
            if(ty == 0 && tx >= k && tx < m)
                A[tx + k * lda] = pivot_val;
        }

        // put pivot row in shared mem
        if(tx < n && ty == 0)
        {
            y[tx] = A[pivot_idx + tx * lda];
            if(tx == k)
                val = pivot_val;
        }
        __syncthreads();

        // swap pivot row with updated row k
        if(tx < n && ty == 0 && pivot_idx != k)
        {
            val1 = (tx == k) ? val : A[k + tx * lda];
            val1 -= (tx > k) ? val * y[tx] : 0;
            A[pivot_idx + tx * lda] = val1;
            A[k + tx * lda] = y[tx];
            if(tx == k + 1)
            {
                x[pivot_idx] = val1;
                sval[pivot_idx] = val1;
            }
            if(pivot && permut_idx && tx == k)
                swap(permut[k], permut[pivot_idx]);
        }

        // complete the rank update
        if(tx > k && tx < m && tx != pivot_idx)
        {
            for(int j = ty + k + 2; j < n; j += bdy)
            {
                val1 = A[tx + j * lda];
                val1 -= pivot_val * y[j];
                A[tx + j * lda] = val1;
            }

            if(ty == 0 && k < n - 1)
            {
                val1 = A[tx + (k + 1) * lda];
                val1 -= pivot_val * y[k + 1];
                A[tx + (k + 1) * lda] = val1;
                x[tx] = val1;
                sval[tx] = val1;
            }
        }

        // update ipiv and prepare for next step
        if(tx <= k && ty == 0)
        {
            val1 = 0;
            x[tx] = 0;
            sval[tx] = 0;
        }
        idx1 = tx;
        if(ty == 0)
            sidx[tx] = idx1;
    }

    // update info
    if(tx == 0 && *info == 0 && myinfo > 0 && ty == 0)
        *info = myinfo + offset;
}

template <typename T, typename U>
rocblas_status getf2_run_panel(rocblas_handle handle,
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
                               const bool pivot,
                               const rocblas_int offset,
                               rocblas_int* permut_idx,
                               const rocblas_stride stride)
{
    // determine sizes
    rocblas_int dimy, dimx;
    if(m <= 8)
        dimx = 8;
    else if(m <= 16)
        dimx = 16;
    else if(m <= 32)
        dimx = 32;
    else if(m <= 64)
        dimx = 64;
    else if(m <= 128)
        dimx = 128;
    else if(m <= 256)
        dimx = 256;
    else if(m <= 512)
        dimx = 512;
    else
        dimx = 1024;
    dimy = 1024 / dimx;

    // prepare kernel launch
    size_t lmemsize = (2 * dimx + n) * sizeof(T) + dimx * sizeof(rocblas_int);
    dim3 grid(1, 1, batch_count);
    dim3 block(dimx, dimy, 1);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    hipLaunchKernelGGL((getf2_panel_kernel<0, T>), grid, block, lmemsize, stream, m, n, A, shiftA,
                       lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot, offset,
                       permut_idx, stride);

    return rocblas_status_success;
}

/** This kernel executes an optimized scaled rank-update (scal + ger)
    for panel matrices (matrices with less than 128 columns).
    Useful to speedup the factorization of block-columns in getrf **/
template <typename T, typename U>
//template <rocblas_int N, typename T, typename U>
ROCSOLVER_KERNEL void getf2_scale_update(const rocblas_int m,
                                         const rocblas_int n,
                                         T* pivotval,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA)
{
    // indices
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int ty = hipThreadIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + tx;

    // shared data arrays
    T pivot, val;
    extern __shared__ double lmem[];
    T* x = (T*)lmem;
    T* y = x + hipBlockDim_x;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA + 1 + lda, strideA);
    T* X = load_ptr_batch(AA, bid, shiftA + 1, strideA);
    T* Y = load_ptr_batch(AA, bid, shiftA + lda, strideA);
    pivot = pivotval[bid];

    // read data from global to shared memory
    int j = tx * hipBlockDim_y + ty;
    if(j < n)
        y[j] = Y[j * lda];

    // scale
    if(ty == 0 && i < m)
    {
        x[tx] = X[i];
        x[tx] *= pivot;
        X[i] = x[tx];
    }
    __syncthreads();

    // rank update; put computed values back to global memory
    if(i < m)
    {
#pragma unroll
        //        for(int c = 0; c < cc; ++c)
        for(int j = ty; j < n; j += hipBlockDim_y)
        {
            //            j = c * hipBlockDim_y + ty;
            val = A[i + j * lda];
            val -= x[tx] * y[j];
            A[i + j * lda] = val;
        }

        //        j = N * hipBlockDim_y + ty;
        //        if(j < n)
        //        {
        //            val = A[i + j * lda];
        //            val -= x[tx] * y[j];
        //            A[i + j * lda] = val;
        //        }
    }
}

/** This kernel updates the chosen pivot, checks singularity and
    interchanges rows all at once (pivoting + laswp)**/
template <typename T, typename U>
ROCSOLVER_KERNEL void getf2_check_singularity(const rocblas_int n,
                                              const rocblas_int j,
                                              U AA,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              rocblas_int* ipivA,
                                              const rocblas_int shiftP,
                                              const rocblas_stride strideP,
                                              T* pivot_val,
                                              rocblas_int* pivot_idxA,
                                              rocblas_int* info,
                                              const rocblas_int offset,
                                              rocblas_int* permut_idx,
                                              const rocblas_stride stride)
{
    using S = decltype(std::real(T{}));

    const int id = hipBlockIdx_y;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // batch instance
        T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
        rocblas_int pivot_idx = pivot_idxA[id] + j;

        // swap rows
        rocblas_int exch = pivot_idx - 1;
        if(exch != j)
            swap(A[j + tid * lda], A[exch + tid * lda]);

        if(tid == j)
        {
            // update pivot index
            rocblas_int* ipiv = ipivA + id * strideP + shiftP;
            ipiv[j] = pivot_idx + offset;

            // update row order of final permutated matrix
            if(permut_idx)
            {
                rocblas_int* permut = permut_idx + id * stride;
                if(exch != j)
                    swap(permut[j], permut[exch]);
            }

            // update info (check singularity)
            if(A[j + j * lda] == 0)
            {
                pivot_val[id] = 1;
                if(info[id] == 0)
                    info[id] = j + 1 + offset; // use Fortran 1-based indexing
            }
            else
                pivot_val[id] = S(1) / A[j + j * lda];
        }
    }
}

/** Non-pivoting version **/
template <typename T, typename U>
ROCSOLVER_KERNEL void getf2_npvt_check_singularity(const rocblas_int j,
                                                   U AA,
                                                   const rocblas_int shiftA,
                                                   const rocblas_int lda,
                                                   const rocblas_stride strideA,
                                                   T* pivot_val,
                                                   rocblas_int* info,
                                                   const rocblas_int offset)
{
    using S = decltype(std::real(T{}));

    const int id = hipBlockIdx_y;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);

    // update info (check singularity)
    if(A[j + j * lda] == 0)
    {
        pivot_val[id] = 1;
        if(info[id] == 0)
            info[id] = j + 1 + offset; // use Fortran 1-based indexing
    }
    else
        pivot_val[id] = S(1) / A[j + j * lda];
}

/** This kernel executes an optimized reduction to find the index of the
    maximum element of a given vector (iamax) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(IAMAX_THDS) getf2_iamax(const rocblas_int m,
                                                                U xx,
                                                                const rocblas_int shiftx,
                                                                const rocblas_stride stridex,
                                                                rocblas_int* pivotidx)
{
    // batch instance
    const int bid = hipBlockIdx_y;
    const int tid = hipThreadIdx_x;
    T* x = load_ptr_batch<T>(xx, bid, shiftx, stridex);

    // shared memory setup
    __shared__ T sval[IAMAX_THDS];
    __shared__ rocblas_int sidx[IAMAX_THDS];

    iamax<IAMAX_THDS>(tid, m, x, 1, sval, sidx);

    // write results back to global memory
    // (after the reduction, the maximum of the elements is in sval[0] and sidx[0])
    if(tid == 0)
        pivotidx[bid] = sidx[0];
}

inline rocblas_int getf2_get_checksingularity_blksize(const rocblas_int n)
{
    rocblas_int singular_thds;

    if(n < 1024)
        singular_thds = 64;
    else if(n < 2048)
        singular_thds = 128;
    else if(n < 4096)
        singular_thds = 256;
    else if(n < 8192)
        singular_thds = 512;
    else
        singular_thds = 1024;

    return singular_thds;
}

template <typename T>
ROCSOLVER_KERNEL void
    getf2_permut_init(const rocblas_int m, rocblas_int* permutA, const rocblas_stride stride)
{
    int id = hipBlockIdx_y;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // batch instance
    rocblas_int* permut = permutA + id * stride;

    // initialize
    if(i < m)
        permut[i] = i;
}

inline void getf2_get_ger_blksize(const rocblas_int m,
                                  const rocblas_int n,
                                  rocblas_int* dimx,
                                  rocblas_int* dimy)
{
    rocblas_int dim;

    if(n == 0 || n > 256 || m == 0)
    {
        dim = 1024;
    }
    else if(n <= 24)
    {
        if(m < 1536)
            dim = n < 16 ? n : 16;
        else if(m < 2688)
            dim = n < 8 ? n : 8;
        else if(m < 9216)
            dim = n < 4 ? n : 4;
        else
            dim = n < 8 ? n : 8;
    }
    else if(n <= 40)
    {
        if(m < 1024)
            dim = 16;
        else
            dim = 8;
    }
    else if(n <= 56)
    {
        if(m < 10240)
            dim = 16;
        else
            dim = 8;
    }
    else if(n <= 88)
    {
        if(m < 5632)
            dim = 16;
        else if(m < 7936)
            dim = 8;
        else
            dim = 4;
    }
    else
    {
        if(m < 4096)
            dim = 16;
        else if(m < 8192)
            dim = 8;
        else
            dim = 4;
    }
    *dimy = dim;
    *dimx = 1024 / dim;
}

/** Return the sizes of the different workspace arrays **/
template <bool ISBATCHED, typename T>
void rocsolver_getf2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const bool pivot,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        return;
    }

#ifdef OPTIMAL
    // if using optimized algorithm for small sizes, no workspace needed
    if(n <= GETF2_MAX_COLS && m <= GETF2_MAX_THDS)
    {
        *size_scalars = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        return;
    }
#endif

    // for scalars
    *size_scalars = sizeof(T) * 3;

    // for pivot values
    *size_pivotval = sizeof(T) * batch_count;

    // for pivot indices
    *size_pivotidx = pivot ? sizeof(rocblas_int) * batch_count : 0;
}

/** argument checking **/
template <typename T>
rocblas_status rocsolver_getf2_getrf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              rocblas_int* ipiv,
                                              rocblas_int* info,
                                              const bool pivot,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m * n && !A) || (m * n && pivot && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool ISBATCHED, typename T, typename U>
rocblas_status rocsolver_getf2_template(rocblas_handle handle,
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
                                        T* pivotval,
                                        rocblas_int* pivotidx,
                                        const bool pivot,
                                        const rocblas_int offset = 0,
                                        rocblas_int* permut_idx = nullptr,
                                        const rocblas_stride stride = 0)
{
    ROCSOLVER_ENTER("getf2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(256, 1, 1);
    rocblas_int dim = min(m, n); // total number of pivots

    // info=0 (starting with a nonsingular matrix)
    if(offset == 0)
        hipLaunchKernelGGL(reset_info, grid, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    // initialize permutation array if needed
    if(permut_idx)
    {
        blocks = (m - 1) / 256 + 1;
        threads = dim3(256, 1, 1);
        grid = dim3(blocks, batch_count, 1);
        hipLaunchKernelGGL(getf2_permut_init<T>, grid, threads, 0, stream, m, permut_idx, stride);
    }

#ifdef OPTIMAL
    // Use specialized kernels for small sizes
    if(n <= GETF2_MAX_COLS && m <= GETF2_MAX_THDS && m < n)
        return getf2_run_small<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                  info, batch_count, pivot, offset, permut_idx, stride);
#endif

    // use kernel for small panel matrices
    if(m >= n)
    {
        if((n <= 32 && m <= 1024) || (n <= 88 && m <= 512) || (n <= 128 && m <= 256))
            return getf2_run_panel<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                      info, batch_count, pivot, offset, permut_idx, stride);
    }

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // prepare kernels
    rocblas_int singular_thds = getf2_get_checksingularity_blksize(n);
    dim3 gridMax(1, batch_count, 1);
    dim3 threadsMax(IAMAX_THDS, 1, 1);
    blocks = pivot ? (n - 1) / singular_thds + 1 : 1;
    dim3 threadsPivot((pivot ? singular_thds : 1), 1, 1);
    dim3 gridPivot(blocks, batch_count, 1);
    rocblas_int c, mm, nn;
    rocblas_int sger_thds_x, sger_thds_y;
    size_t lmemsize;

    for(rocblas_int j = 0; j < dim; ++j)
    {
        if(pivot)
        {
            // find pivot. Use Fortran 1-based indexing (to follow LAPACK)
            hipLaunchKernelGGL((getf2_iamax<T>), gridMax, threadsMax, 0, stream, m - j, A,
                               shiftA + idx2D(j, j, lda), strideA, pivotidx);

            // adjust pivot indices, apply row interchanges and check singularity
            hipLaunchKernelGGL(getf2_check_singularity<T>, gridPivot, threadsPivot, 0, stream, n, j,
                               A, shiftA, lda, strideA, ipiv, shiftP, strideP, pivotval, pivotidx,
                               info, offset, permut_idx, stride);
        }
        else
            // check singularity
            hipLaunchKernelGGL(getf2_npvt_check_singularity<T>, gridPivot, threadsPivot, 0, stream,
                               j, A, shiftA, lda, strideA, pivotval, info, offset);

        // get thread block size for matrix update
        mm = m - j - 1;
        nn = n - j - 1;
        getf2_get_ger_blksize(mm, nn, &sger_thds_x, &sger_thds_y);

        if(sger_thds_x == 1) //if working with a general matrix:
        {
            // Scale J'th column
            rocblasCall_scal<T>(handle, mm, pivotval, 1, A, shiftA + idx2D(j + 1, j, lda), 1,
                                strideA, batch_count);

            // update trailing submatrix
            if(j < dim - 1)
            {
                rocblasCall_ger<false, T>(
                    handle, mm, nn, scalars, 0, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, A,
                    shiftA + idx2D(j, j + 1, lda), lda, strideA, A,
                    shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, batch_count, nullptr);
            }
        }
        else //if working with few columns in matrix
        {
            lmemsize = sizeof(T) * (sger_thds_x + nn);
            //            c = (nn - 1) / sger_thds_y;
            blocks = (mm - 1) / sger_thds_x + 1;
            threads = dim3(sger_thds_x, sger_thds_y, 1);
            grid = dim3(blocks, 1, batch_count);

            // scale and update trailing matrix with local function
            //#define RUN_UPDATE(N)
            hipLaunchKernelGGL((getf2_scale_update<T>), grid, threads, lmemsize, stream, mm, nn,
                               pivotval, A, shiftA + idx2D(j, j, lda), lda, strideA);
            /*
            switch(c)
            {
            case 0: RUN_UPDATE(0); break;
            case 1: RUN_UPDATE(1); break;
            case 2: RUN_UPDATE(2); break;
            case 3: RUN_UPDATE(3); break;
            case 4: RUN_UPDATE(4); break;
            case 5: RUN_UPDATE(5); break;
            case 6: RUN_UPDATE(6); break;
            case 7: RUN_UPDATE(7); break;
            case 8: RUN_UPDATE(8); break;
            case 9: RUN_UPDATE(9); break;
            case 10: RUN_UPDATE(10); break;
            case 11: RUN_UPDATE(11); break;
            case 12: RUN_UPDATE(12); break;
            case 13: RUN_UPDATE(13); break;
            case 14: RUN_UPDATE(14); break;
            case 15: RUN_UPDATE(15); break;
            case 16: RUN_UPDATE(16); break;
            case 17: RUN_UPDATE(17); break;
            case 18: RUN_UPDATE(18); break;
            case 19: RUN_UPDATE(19); break;
            case 20: RUN_UPDATE(20); break;
            case 21: RUN_UPDATE(21); break;
            case 22: RUN_UPDATE(22); break;
            case 23: RUN_UPDATE(23); break;
            case 24: RUN_UPDATE(24); break;
            case 25: RUN_UPDATE(25); break;
            case 26: RUN_UPDATE(26); break;
            case 27: RUN_UPDATE(27); break;
            case 28: RUN_UPDATE(28); break;
            case 29: RUN_UPDATE(29); break;
            case 30: RUN_UPDATE(30); break;
            case 31: RUN_UPDATE(31); break;
            case 32: RUN_UPDATE(32); break;
            case 33: RUN_UPDATE(33); break;
            case 34: RUN_UPDATE(34); break;
            case 35: RUN_UPDATE(35); break;
            case 36: RUN_UPDATE(36); break;
            case 37: RUN_UPDATE(37); break;
            case 38: RUN_UPDATE(38); break;
            case 39: RUN_UPDATE(39); break;
            case 40: RUN_UPDATE(40); break;
            case 41: RUN_UPDATE(41); break;
            case 42: RUN_UPDATE(42); break;
            case 43: RUN_UPDATE(43); break;
            case 44: RUN_UPDATE(44); break;
            case 45: RUN_UPDATE(45); break;
            case 46: RUN_UPDATE(46); break;
            case 47: RUN_UPDATE(47); break;
            case 48: RUN_UPDATE(48); break;
            case 49: RUN_UPDATE(49); break;
            case 50: RUN_UPDATE(50); break;
            case 51: RUN_UPDATE(51); break;
            case 52: RUN_UPDATE(52); break;
            case 53: RUN_UPDATE(53); break;
            case 54: RUN_UPDATE(54); break;
            case 55: RUN_UPDATE(55); break;
            case 56: RUN_UPDATE(56); break;
            case 57: RUN_UPDATE(57); break;
            case 58: RUN_UPDATE(58); break;
            case 59: RUN_UPDATE(59); break;
            case 60: RUN_UPDATE(60); break;
            case 61: RUN_UPDATE(61); break;
            case 62: RUN_UPDATE(62); break;
            case 63: RUN_UPDATE(63); break;
            default: ROCSOLVER_UNREACHABLE();
            }*/
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
