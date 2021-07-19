/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Small sizes algorithm derived from MAGMA project
 * http://icl.cs.utk.edu/magma/.
 * https://doi.org/10.1016/j.procs.2017.05.250
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"
#include "rocsolver_small_kernels.hpp"

template <bool PIVOT, typename T, typename U>
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
        T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
        rocblas_int pivot_idx = pivot_idxA[id];

        if(PIVOT)
        {
            if(tid == j)
            {
                rocblas_int* ipiv = ipivA + id * strideP + shiftP;
                ipiv[j] = pivot_idx + j; // update pivot index
            }
            if(pivot_idx > 1)
            {
                // swap rows
                T orig = A[j+tid*lda];
                A[j+tid*lda] = A[pivot_idx + j - 1 + tid*lda];
                A[pivot_idx + j - 1 + tid*lda] = orig;
            }
        }

        if(tid == j)
        {
            if(A[j + j*lda] == 0)
            {
                pivot_val[id] = 1;
                if(info[id] == 0)
                    info[id] = j + 1; // use Fortran 1-based indexing
            }
            else
                pivot_val[id] = S(1) / A[j + j*lda];
        }
    }
}

template <bool ISBATCHED, typename T, typename S>
void rocsolver_getf2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        return;
    }

#ifdef OPTIMAL
    // if using optimized algorithm for small sizes, no workspace needed
    if(n <= WAVESIZE)
    {
        if(m <= GETF2_MAX_THDS || (m <= GETF2_OPTIM_MAX_SIZE && !ISBATCHED)
           || (m <= GETF2_BATCH_OPTIM_MAX_SIZE && ISBATCHED))
        {
            *size_scalars = 0;
            *size_work = 0;
            *size_pivotval = 0;
            *size_pivotidx = 0;
            return;
        }
    }
#endif

    // for scalars
    *size_scalars = sizeof(T) * 3;

    // for pivot values
    *size_pivotval = sizeof(T) * batch_count;

    // for pivot indices
    *size_pivotidx = sizeof(rocblas_int) * batch_count;

    // for workspace
    *size_work = sizeof(rocblas_index_value_t<S>) * ((m - 1) / ROCBLAS_IAMAX_NB + 2) * batch_count;
}

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

template <bool ISBATCHED, bool PIVOT, typename T, typename S, typename U>
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
                                        rocblas_index_value_t<S>* work,
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
    if(n <= WAVESIZE)
    {
        if((m <= GETF2_OPTIM_MAX_SIZE && !ISBATCHED)
           || (m <= GETF2_BATCH_OPTIM_MAX_SIZE && ISBATCHED))
            return getf2_run_small<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                      info, batch_count, PIVOT);
    }
#endif

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    rocblas_int blocksPivot = (n - 1) / LASWP_BLOCKSIZE + 1;
    dim3 gridPivot(blocksPivot, batch_count, 1);
    dim3 threadsPivot(LASWP_BLOCKSIZE, 1, 1);

    for(rocblas_int j = 0; j < dim; ++j)
    {
        if(PIVOT)
            // find pivot. Use Fortran 1-based indexing for the ipiv array as iamax
            // does that as well!
            rocblasCall_iamax<ISBATCHED, T, S>(handle, m - j, A, shiftA + idx2D(j, j, lda), 1,
                                               strideA, batch_count, pivotidx, work);

        // adjust pivot indices, apply row interchanges and check singularity
        hipLaunchKernelGGL((getf2_check_singularity<PIVOT,T>), gridPivot, threadsPivot, 0, stream, n, A,
                           shiftA, strideA, ipiv, shiftP, strideP, j, lda, pivotval, pivotidx, info);

        // Compute elements J+1:M of J'th column
        rocblasCall_scal<T>(handle, m - j - 1, pivotval, 1, A, shiftA + idx2D(j + 1, j, lda), 1,
                            strideA, batch_count);

        // update trailing submatrix
        if(j < min(m, n) - 1)
        {
            rocblasCall_ger<false, T>(
                handle, m - j - 1, n - j - 1, scalars, 0, A, shiftA + idx2D(j + 1, j, lda), 1,
                strideA, A, shiftA + idx2D(j, j + 1, lda), lda, strideA, A,
                shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, batch_count, nullptr);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
