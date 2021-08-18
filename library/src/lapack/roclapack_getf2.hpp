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
#include "rocblas.hpp"
#include "rocsolver.h"
#include "rocsolver_small_kernels.hpp"

// number of threads for the iamax reduction kernel
#define IAMAX_THDS 1024
// number of columns at which we switch from panel to general matrix
#define GENERAL_PANEL_SWITCHSIZE 128
// number of threads for the scal+ger kernel
#define SGER_DIMX 128
#define SGER_DIMY 8

/** This kernel executes an optimized scaled rank-update (scal + ger)
    for panel matrices (matrices with less than 128 columns).
    Useful to speedup the factorization of block-columns in getrf **/
template <typename T, typename U>
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
    T pivot;
    __shared__ T x[SGER_DIMX];
    __shared__ T y[GENERAL_PANEL_SWITCHSIZE];

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA + 1 + lda, strideA);
    T* X = load_ptr_batch(AA, bid, shiftA + 1, strideA);
    T* Y = load_ptr_batch(AA, bid, shiftA + lda, strideA);
    pivot = pivotval[bid];

    rocblas_int tyj;

    // read data from global to shared memory
    if(tx == 0)
    {
        for(int j = ty; j < n; j += hipBlockDim_y)
            y[j] = Y[j * lda];
    }
    if(ty == 0 && i < m)
    {
        // scale
        x[tx] = X[i] * pivot;
        X[i] = x[tx];
    }
    __syncthreads();

    // rank update; put computed values back to global memory
    if(i < m)
    {
        for(int j = ty; j < n; j += hipBlockDim_y)
            A[i + j * lda] -= x[tx] * y[j];
    }
}

/** This kernel updates the choosen pivot, checks singularity and
    interchanges rows all at once (pivoting + laswp)**/
template <bool PIVOT, typename T, typename U, std::enable_if_t<PIVOT, int> = 0>
ROCSOLVER_KERNEL void getf2_check_singularity(const rocblas_int n,
                                              U AA,
                                              const rocblas_int shiftA,
                                              const rocblas_stride strideA,
                                              rocblas_int* ipivA,
                                              const rocblas_int shiftP,
                                              const rocblas_stride strideP,
                                              const rocblas_int j,
                                              const rocblas_int lda,
                                              T* pivot_val,
                                              rocblas_int* pivot_idxA,
                                              rocblas_int* info)
{
    using S = decltype(std::real(T{}));

    const int id = hipBlockIdx_y;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // batch instance
        T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);

        if(PIVOT)
        {
            rocblas_int pivot_idx = pivot_idxA[id];
            if(tid == j)
            {
                rocblas_int* ipiv = ipivA + id * strideP + shiftP;
                ipiv[j] = pivot_idx + j; // update pivot index
            }
            if(pivot_idx > 1)
                swap(A[j + tid * lda], A[pivot_idx + j - 1 + tid * lda]); // swap rows
        }

        // update info (check singularity)
        if(tid == j)
        {
            if(A[j + j * lda] == 0)
            {
                pivot_val[id] = 1;
                if(info[id] == 0)
                    info[id] = j + 1; // use Fortran 1-based indexing
            }
            else
                pivot_val[id] = S(1) / A[j + j * lda];
        }
    }
}

/** Non-pivoting version **/
template <bool PIVOT, typename T, typename U, std::enable_if_t<!PIVOT, int> = 0>
ROCSOLVER_KERNEL void getf2_check_singularity(const rocblas_int n,
                                              U AA,
                                              const rocblas_int shiftA,
                                              const rocblas_stride strideA,
                                              rocblas_int* ipivA,
                                              const rocblas_int shiftP,
                                              const rocblas_stride strideP,
                                              const rocblas_int j,
                                              const rocblas_int lda,
                                              T* pivot_val,
                                              rocblas_int* pivot_idxA,
                                              rocblas_int* info)
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
            info[id] = j + 1; // use Fortran 1-based indexing
    }
    else
        pivot_val[id] = S(1) / A[j + j * lda];
}

/** This kernel executes an optimized reduction to find the index of the
    maximum element of a given vector (iamax) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void getf2_iamax(const rocblas_int m,
                                  U xx,
                                  const rocblas_int shiftx,
                                  const rocblas_stride stridex,
                                  rocblas_int* pivotidx)
{
    using S = decltype(std::real(T{}));

    // batch instance
    const int bid = hipBlockIdx_y;
    T* x = load_ptr_batch<T>(xx, bid, shiftx, stridex);

    // shared & local memory setup
    __shared__ T sval[IAMAX_THDS];
    __shared__ rocblas_int sidx[IAMAX_THDS];
    T val1, val2;
    rocblas_int pidx;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bdim = hipBlockDim_x;
    val1 = 0;
    pidx = 1;
    for(unsigned int i = tid; i < m; i += bdim)
    {
        val2 = x[i];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            val1 = val2;
            pidx = i + 1; //add one to make it 1-based index
        }
    }
    sval[tid] = val1;
    sidx[tid] = pidx;
    __syncthreads();

    /** <========= Next do the reduction on the shared memory array =========>
        (We need to execute the for loop
            for(j = IAMAX_THDS; j > 0; j>>=1)
        to have half of the active threads at each step
        reducing two elements in the shared array.
        As IAMAX_THDS is fixed to 1024, we can unroll the loop manualy) **/

    if(tid < 512)
    {
        val1 = sval[tid];
        val2 = sval[tid + 512];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 512];
        }
    }
    __syncthreads();

    if(tid < 256)
    {
        val1 = sval[tid];
        val2 = sval[tid + 256];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 256];
        }
    }
    __syncthreads();

    if(tid < 128)
    {
        val1 = sval[tid];
        val2 = sval[tid + 128];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 128];
        }
    }
    __syncthreads();

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < 64)
    {
        val1 = sval[tid];
        val2 = sval[tid + 64];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 64];
        }
        val1 = sval[tid];
        val2 = sval[tid + 32];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 32];
        }
        val1 = sval[tid];
        val2 = sval[tid + 16];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 16];
        }
        val1 = sval[tid];
        val2 = sval[tid + 8];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 8];
        }
        val1 = sval[tid];
        val2 = sval[tid + 4];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 4];
        }
        val1 = sval[tid];
        val2 = sval[tid + 2];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 2];
        }
        val1 = sval[tid];
        val2 = sval[tid + 1];
        if(aabs<S>(val1) < aabs<S>(val2))
        {
            sval[tid] = val2;
            sidx[tid] = sidx[tid + 1];
        }
    }

    // write results back to global memory
    // (after the reduction, the maximum of the elements is in sval[0] and sidx[0])
    if(tid == 0)
        pivotidx[bid] = sidx[0];
}

/** Return the sizes of the different workspace arrays **/
template <bool ISBATCHED, bool PIVOT, typename T>
void rocsolver_getf2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
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
    *size_pivotidx = PIVOT ? sizeof(rocblas_int) * batch_count : 0;
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
                                              const rocblas_int pivot,
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

template <bool ISBATCHED, bool PIVOT, typename T, typename U>
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
                                        rocblas_int* pivotidx)
{
    ROCSOLVER_ENTER("getf2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threadsReset(BLOCKSIZE, 1, 1);
    rocblas_int dim = min(m, n); // total number of pivots

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

#ifdef OPTIMAL
    // Use optimized LU factorization for the right sizes
    if(n <= GETF2_MAX_COLS && m <= GETF2_MAX_THDS)
        return getf2_run_small<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                  info, batch_count, PIVOT);
#endif

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // prepare kernels
    rocblas_int blocksx;
    dim3 grid, threads;
    dim3 gridMax(1, batch_count, 1);
    dim3 threadsMax(IAMAX_THDS, 1, 1);
    rocblas_int blocksPivot = PIVOT ? (n - 1) / LASWP_BLOCKSIZE + 1 : 1;
    dim3 gridPivot(blocksPivot, batch_count, 1);
    dim3 threadsPivot((PIVOT ? LASWP_BLOCKSIZE : 1), 1, 1);

    for(rocblas_int j = 0; j < dim; ++j)
    {
        if(PIVOT)
            // find pivot. Use Fortran 1-based indexing (to follow LAPACK)
            hipLaunchKernelGGL((getf2_iamax<T>), gridMax, threadsMax, 0, stream, m - j, A,
                               shiftA + idx2D(j, j, lda), strideA, pivotidx);

        // adjust pivot indices, apply row interchanges and check singularity
        hipLaunchKernelGGL((getf2_check_singularity<PIVOT, T>), gridPivot, threadsPivot, 0, stream,
                           n, A, shiftA, strideA, ipiv, shiftP, strideP, j, lda, pivotval, pivotidx,
                           info);

        if(n - j - 1 > GENERAL_PANEL_SWITCHSIZE) //if working with a general matrix:
        {
            // Scale J'th column
            rocblasCall_scal<T>(handle, m - j - 1, pivotval, 1, A, shiftA + idx2D(j + 1, j, lda), 1,
                                strideA, batch_count);

            // update trailing submatrix
            if(j < dim - 1)
            {
                rocblasCall_ger<false, T>(
                    handle, m - j - 1, n - j - 1, scalars, 0, A, shiftA + idx2D(j + 1, j, lda), 1,
                    strideA, A, shiftA + idx2D(j, j + 1, lda), lda, strideA, A,
                    shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, batch_count, nullptr);
            }
        }
        else //if working with a panel matrix
        {
            blocksx = (m - j - 2) / SGER_DIMX + 1;
            grid = dim3(blocksx, 1, batch_count);
            threads = dim3(SGER_DIMX, SGER_DIMY, 1);

            // scale and update panel trailing matrix with local function
            hipLaunchKernelGGL((getf2_scale_update<T>), grid, threads, 0, stream, m - j - 1,
                               n - j - 1, pivotval, A, shiftA + idx2D(j, j, lda), lda, strideA);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
