/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 *
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

// number of threads for the iamax reduction kernel
#define IAMAX_THDS 1024

/** this kernel initializes the permutation array
    which is instrumental for parallel row permutations in GETRF **/
template <typename T>
ROCSOLVER_KERNEL void
    getf2_permut_init(const rocblas_int m, rocblas_int* permutA, const rocblas_stride stridePI)
{
    int id = hipBlockIdx_y;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // batch instance
    rocblas_int* permut = permutA + id * stridePI;

    // initialize
    if(i < m)
        permut[i] = i;
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
                                              const rocblas_stride stridePI)
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
                rocblas_int* permut = permut_idx + id * stridePI;
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
    using S = decltype(std::real(T{}));

    // batch instance
    const int bid = hipBlockIdx_y;
    const int tid = hipThreadIdx_x;
    T* x = load_ptr_batch<T>(xx, bid, shiftx, stridex);

    // shared memory setup
    __shared__ S sval[IAMAX_THDS];
    __shared__ rocblas_int sidx[IAMAX_THDS];

    iamax<IAMAX_THDS>(tid, m, x, 1, sval, sidx);

    // write results back to global memory
    // (after the reduction, the maximum of the elements is in sval[0] and sidx[0])
    if(tid == 0)
        pivotidx[bid] = sidx[0];
}

/** Returns the thread block sizes used for the singularity check and pivot update**/
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

/** This function tests if one of the specialized kernels should be used.
    Returns 1 when the use of the small kernel will give better performance,
    Returns 2 when the use of the panel kernel will give better performance,
    Returns 0 when it would be better to use the normal code. **/
template <bool ISBATCHED, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
int select_spkernel(const rocblas_int m, const rocblas_int n, const bool pivot)
{
    int ker = 0;

    if(ISBATCHED)
    {
        // Batch pivoting case (real precisions)
        if(pivot)
        {
            if(n <= 28)
            {
                ker = (m <= 140) ? 1 : 2;
            }
            else if(n <= 44)
            {
                ker = (m <= 256) ? 1 : 2;
            }
            else if((n <= 52 && (m <= 392 || m > 504)) || (n > 52 && n <= 60 && (m <= 392 || m > 624))
                    || (n > 60 && n <= 68 && (m <= 296 || m > 864)))
            {
                ker = (m <= 256) ? 1 : 2;
            }
            else if((n > 68 && n <= 76 && m >= n && (m <= 296 || m > 592))
                    || (n > 76 && n <= 92 && m >= n && (m <= 244 || m > 848))
                    || (n > 92 && n <= 108 && m >= n && (m <= 256 || m > 592))
                    || (n > 108 && m >= n && (m <= 164 || m > 736)))
            {
                ker = 2;
            }
        }
        // Batch non-pivoting case (real precisions)
        else
        {
            if(n <= 68)
            {
                ker = (m <= 512) ? 1 : 2;
            }
            else if(n > 68 && m >= n)
            {
                ker = 2;
            }
        }
    }
    else
    {
        // Normal pivoting case (real precisions)
        if(pivot)
        {
            if(n <= 20)
            {
                ker = (m <= 32) ? 1 : 2;
            }
            else if((n <= 28 && m >= 6) || (n > 28 && n <= 36 && m >= 8 && m <= 512)
                    || (n > 36 && n <= 44 && m >= 10 && m <= 512)
                    || (n > 44 && n <= 52 && m >= 16 && m <= 512)
                    || (n > 52 && n <= 60 && m >= 44 && m <= 344))
            {
                ker = (m < n) ? 1 : 2;
            }
            else if((n > 60 && n <= 68 && m >= n && m <= 344) || (n > 68 && m >= n && m <= 256))
            {
                ker = 2;
            }
        }
        // Normal non-pivoting case (real precisions)
        else
        {
            if((n <= 20) || (n <= 28 && m >= 6 && m <= 912))
            {
                ker = (m <= 512) ? 1 : 2;
            }
            else if((n > 28 && n <= 36 && m >= 6 && m <= 512)
                    || (n > 36 && n <= 44 && m >= 10 && m <= 512))
            {
                ker = (m < n || m > 66) ? 1 : 2;
            }
            else if(n > 44 && n <= 52 && m >= 24 && m <= 512)
            {
                ker = (m < n || m > 128) ? 1 : 2;
            }
            else if((n > 52 && n <= 60 && m >= n && m <= 256)
                    || (n > 60 && n <= 92 && m >= n && m <= 196) || (n > 92 && m >= n && m <= 128))
            {
                ker = 2;
            }
        }
    }

    if(ker == 1 && (m > GETF2_SSKER_MAX_M || n > GETF2_SSKER_MAX_N))
    {
        ker = 2;
    }

    return ker;
}

/** Complex type version **/
template <bool ISBATCHED, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
int select_spkernel(const rocblas_int m, const rocblas_int n, const bool pivot)
{
    int ker = 0;

    if(ISBATCHED)
    {
        // Batch pivoting case (complex precisions)
        if(pivot)
        {
            if((n <= 20) || (n <= 28 && m <= 656) || (n > 28 && n <= 36 && m <= 504))
            {
                ker = (m <= 64) ? 1 : 2;
            }
            else if(n > 36 && n <= 44 && m >= 6 && m <= 624)
            {
                ker = (m < n || (m > 110 && m <= 132)) ? 1 : 2;
            }
            else if(n > 44 && n <= 52 && m >= 8 && m <= 504)
            {
                ker = (m <= 228) ? 1 : 2;
            }
            else if(n > 52 && n <= 60 && m >= 10 && m <= 504 && (m <= 256 || m > 448))
            {
                ker = (m <= 256) ? 1 : 2;
            }
            else if((n > 60 && n <= 68 && m >= n && m <= 656 && (m <= 148 || m > 328))
                    || (n > 68 && n <= 76 && m >= n && m <= 896 && (m <= 148 || m > 204))
                    || (n > 76 && n <= 124 && m >= n && (m <= 128 || m > 328))
                    || (n > 124 && m >= n && m > 228))
            {
                ker = 2;
            }
        }
        // Batch non-pivoting case (complex precisions)
        else
        {
            if((n <= 20) || (n <= 28 && m <= 752) || (n > 28 && n <= 36 && m <= 608))
            {
                ker = (m <= 512) ? 1 : 2;
            }
            else if((n > 36 && n <= 52 && m >= n && m <= 608)
                    || (n > 52 && n <= 68 && m >= n && m <= 736) || (n > 68 && m >= n))
            {
                ker = 2;
            }
        }
    }
    else
    {
        // Normal pivoting case (complex precisions)
        if(pivot)
        {
            if(n <= 12)
            {
                ker = m < n ? 1 : 2;
            }
            else if((n <= 20 && m >= 6) || (n > 20 && n <= 28 && m >= 10 && m <= 512))
            {
                ker = (m < n) ? 1 : 2;
            }
            else if((n > 28 && n <= 36 && m >= n && m <= 368)
                    || (n > 36 && n <= 60 && m >= n && m <= 336) || (n > 60 && m >= n && m <= 256))
            {
                ker = 2;
            }
        }
        // Normal non-pivoting case (complex precisions)
        else
        {
            if(n <= 12)
            {
                ker = (m < n || (m > 28 && m <= 512)) ? 1 : 2;
            }
            else if(n <= 20 && m <= 928)
            {
                ker = (m < n || (m > 128 && m <= 512)) ? 1 : 2;
            }
            else if(n > 20 && n <= 28 && m >= 8 && m <= 704)
            {
                ker = (m < n || (m > 288 && m <= 512)) ? 1 : 2;
            }
            else if((n > 28 && n <= 36 && m >= n && m <= 480)
                    || (n > 36 && n <= 52 && m >= n && m <= 256)
                    || (n > 52 && n <= 60 && m >= n && m <= 208)
                    || (n > 60 && n <= 68 && m >= n && m <= 172) || (n > 68 && m >= n && m <= 128))
            {
                ker = 2;
            }
        }
    }

    if(ker == 1 && (m > GETF2_SSKER_MAX_M || n > GETF2_SSKER_MAX_N))
    {
        ker = 2;
    }

    return ker;
}

/** Returns the thread block sizes used for the scale+update of trailing matrix**/
inline void getf2_get_ger_blksize(const rocblas_int m,
                                  const rocblas_int n,
                                  rocblas_int* dimx,
                                  rocblas_int* dimy)
{
    rocblas_int dim = 1024;

#ifdef OPTIMAL
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
#endif

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
                                   size_t* size_pivotidx,
                                   bool inblocked = false)
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
    bool nomem = (m <= GETF2_SPKER_MAX_M && n <= GETF2_SPKER_MAX_N
                  && select_spkernel<ISBATCHED, T>(m, n, pivot) && !inblocked);

    // no workspace needed if using optimized kernel for small sizes
    if(nomem)
    {
        *size_scalars = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        return;
    }
#endif

    // inblocked = true when called from inside blocked algorithms like GETRF.

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
                                        const rocblas_stride stridePI = 0)
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
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    // initialize permutation array if needed
    if(permut_idx)
    {
        blocks = (m - 1) / 256 + 1;
        threads = dim3(256, 1, 1);
        grid = dim3(blocks, batch_count, 1);
        ROCSOLVER_LAUNCH_KERNEL(getf2_permut_init<T>, grid, threads, 0, stream, m, permut_idx,
                                stridePI);
    }

#ifdef OPTIMAL
    if(m <= GETF2_SPKER_MAX_M && n <= GETF2_SPKER_MAX_N)
    {
        int spker = select_spkernel<ISBATCHED, T>(m, n, pivot);

        // Use specialized kernels for small matrices
        if(spker == 1)
        {
            return getf2_run_small<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                      info, batch_count, pivot, offset, permut_idx, stridePI);
        }

        // use specialized kernels for small skinny matrices (panel factorization)
        if(spker == 2)
        {
            return getf2_run_panel<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP,
                                      info, batch_count, pivot, offset, permut_idx, stridePI);
        }
    }
#endif

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

    for(rocblas_int j = 0; j < dim; ++j)
    {
        if(pivot)
        {
            // find pivot. Use Fortran 1-based indexing (to follow LAPACK)
            ROCSOLVER_LAUNCH_KERNEL((getf2_iamax<T>), gridMax, threadsMax, 0, stream, m - j, A,
                                    shiftA + idx2D(j, j, lda), strideA, pivotidx);

            // adjust pivot indices, apply row interchanges and check singularity
            ROCSOLVER_LAUNCH_KERNEL(getf2_check_singularity<T>, gridPivot, threadsPivot, 0, stream,
                                    n, j, A, shiftA, lda, strideA, ipiv, shiftP, strideP, pivotval,
                                    pivotidx, info, offset, permut_idx, stridePI);
        }
        else
            // check singularity
            ROCSOLVER_LAUNCH_KERNEL(getf2_npvt_check_singularity<T>, gridPivot, threadsPivot, 0,
                                    stream, j, A, shiftA, lda, strideA, pivotval, info, offset);

        mm = m - j - 1;
        nn = n - j - 1;

        // get thread block size for matrix update
        getf2_get_ger_blksize(mm, nn, &sger_thds_x, &sger_thds_y);

        //if working with a general matrix or without optimizations:
        if(sger_thds_x == 1)
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

        //if working with optimizations and matrix with few columns
#ifdef OPTIMAL
        else
        {
            // scale and update trailing matrix with local function
            getf2_run_scale_update(handle, mm, nn, pivotval, A, shiftA + idx2D(j, j, lda), lda,
                                   strideA, batch_count, sger_thds_x, sger_thds_y);
        }
#endif
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
