/************************************************************************
 * Derived from
 * Gotlub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
 * and
 * Hari & Kovac (2019). On the Convergence of Complex Jacobi Methods.
 *     Linear and Multilinear Algebra 69(3), p. 489-514.
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_syev_heev.hpp"
#include "rocsolver/rocsolver.h"

/** Apply rotation from the left (based on LASR). **/
template <typename T, typename S>
__device__ void syevj_lasr_left(const rocblas_int n,
                                const rocblas_int i,
                                const rocblas_int ii,
                                S c,
                                T s1,
                                T s2,
                                T* A,
                                const rocblas_int lda)
{
    T temp1, temp2;

    for(rocblas_int j = 0; j < n; ++j)
    {
        temp1 = A[i + j * lda];
        temp2 = A[ii + j * lda];
        A[i + j * lda] = c * temp1 + s1 * temp2;
        A[ii + j * lda] = -s2 * temp1 + c * temp2;
    }
}
/** Apply rotation from the right (based on LASR). **/
template <typename T, typename S>
__device__ void syevj_lasr_right(const rocblas_int n,
                                 const rocblas_int j,
                                 const rocblas_int jj,
                                 S c,
                                 T s1,
                                 T s2,
                                 T* A,
                                 const rocblas_int lda)
{
    T temp1, temp2;

    for(rocblas_int i = 0; i < n; ++i)
    {
        temp1 = A[i + j * lda];
        temp2 = A[i + jj * lda];
        A[i + j * lda] = c * temp1 + s2 * temp2;
        A[i + jj * lda] = -s1 * temp1 + c * temp2;
    }
}

/** Compute new off-diagonal squared Frobenius norm. **/
template <typename T, typename S>
__device__ S syevj_update_norm(const rocblas_int tid,
                               const rocblas_int n,
                               const rocblas_int half_n,
                               const rocblas_int i,
                               const rocblas_int ii,
                               T* A,
                               const rocblas_int lda,
                               S* resarr)
{
    S temp1, temp2;

    S local_res = 0;
    for(rocblas_int j = 0; j < n; ++j)
    {
        temp1 = (i < n && i != j ? lengthsq<S>(A[i + j * lda]) : 0);
        temp2 = (ii < n && ii != j ? lengthsq<S>(A[ii + j * lda]) : 0);
        local_res += temp1 + temp2;
    }
    resarr[tid] = local_res;
    __syncthreads();

    local_res = 0;
    for(rocblas_int j = 0; j < half_n; j++)
        local_res += resarr[j];
    return local_res;
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_kernel(const rocblas_evect evect,
                                   const rocblas_fill uplo,
                                   const rocblas_int n,
                                   U AA,
                                   const rocblas_int shiftA,
                                   const rocblas_int lda,
                                   const rocblas_stride strideA,
                                   const S abstol,
                                   S* residual,
                                   const rocblas_int max_sweeps,
                                   rocblas_int* n_sweeps,
                                   S* WW,
                                   const rocblas_stride strideW,
                                   rocblas_int* info,
                                   const rocblas_int batch_count,
                                   T* AcpyA,
                                   S* resarrA,
                                   rocblas_int* tbarrA)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, k;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    if(tid >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* W = WW + bid * strideW;
    S* resarr = resarrA + bid * even_n;
    rocblas_int* top = tbarrA + bid * even_n;
    rocblas_int* bottom = top + half_n;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (first by column/row then sum)
    Acpy[tid + tid * n] = A[tid + tid * lda];
    if(evect != rocblas_evect_none)
        A[tid + tid * lda] = 1;

    S local_res = 0;
    if(uplo == rocblas_fill_upper)
    {
        T temp;
        for(j = tid + 1; j < n; j++)
        {
            temp = A[tid + j * lda];
            local_res += 2 * lengthsq<S>(temp);
            Acpy[tid + j * n] = temp;
            Acpy[j + tid * n] = conj(temp);

            if(evect != rocblas_evect_none)
            {
                A[tid + j * lda] = 0;
                A[j + tid * lda] = 0;
            }
        }
    }
    else
    {
        T temp;
        for(i = tid + 1; i < n; i++)
        {
            temp = A[i + tid * lda];
            local_res += 2 * lengthsq<S>(temp);
            Acpy[i + tid * n] = temp;
            Acpy[tid + i * n] = conj(temp);

            if(evect != rocblas_evect_none)
            {
                A[i + tid * lda] = 0;
                A[tid + i * lda] = 0;
            }
        }
    }
    resarr[tid] = local_res;
    __syncthreads();

    local_res = 0;
    for(i = 0; i < n; i++)
        local_res += resarr[i];
    __syncthreads();

    // quick return if norm is already small
    if(local_res <= abstol * abstol)
        return;

    // only need half_n threads at this point
    if(tid >= half_n)
        return;

    // initialize top/bottom
    top[tid] = 2 * tid;
    bottom[tid] = 2 * tid + 1;

    // execute sweeps
    while(sweeps < max_sweeps && local_res > abstol * abstol)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        S c, t, mag;
        T s1, s2, aij;
        for(k = 0; k < n - 1; k++)
        {
            i = top[tid];
            j = bottom[tid];

            if(i < n && j < n)
            {
                aij = Acpy[i + j * n];
                mag = length<S>(aij);

                // calculate rotation J
                if(mag == 0)
                {
                    c = 1;
                    s1 = s2 = 0;
                }
                else
                {
                    t = std::real(Acpy[i + i * n] - Acpy[j + j * n]);
                    t = (t >= 0 ? 2 * mag : -2 * mag) / (abs(t) + sqrt(t * t + 4 * mag * mag));

                    c = 1.0 / sqrt(1 + t * t);
                    s1 = t * c * aij / mag;
                    s2 = conj(s1);
                }

                // apply J' from the left
                syevj_lasr_left(n, i, j, c, s1, s2, Acpy, n);

                // update eigenvectors
                if(evect != rocblas_evect_none)
                    syevj_lasr_right(n, i, j, c, s1, s2, A, lda);
            }
            __syncthreads();

            if(i < n && j < n)
            {
                // apply J from the right
                syevj_lasr_right(n, i, j, c, s1, s2, Acpy, n);

                // round aij and aji to zero
                Acpy[j + i * n] = Acpy[i + j * n] = 0;
            }

            // cycle top/bottom pairs
            if(tid == 1)
                i = bottom[0];
            else if(tid > 1)
                i = top[tid - 1];
            if(tid == half_n - 1)
                j = top[half_n - 1];
            else
                j = bottom[tid + 1];
            __syncthreads();

            top[tid] = i;
            bottom[tid] = j;
            __syncthreads();
        }

        // update norm
        local_res = syevj_update_norm(tid, n, half_n, i, j, Acpy, n, resarr);

        sweeps++;
    }

    // update residual, n_sweeps, and info()
    if(tid == 0)
    {
        residual[bid] = sqrt(local_res);
        n_sweeps[bid] = sweeps;
        if(sweeps > max_sweeps)
            info[bid] = n + 1;
        else
            info[bid] = 0;
    }

    // update W and then sort eigenvalues and eigenvectors by selection sort
    for(i = tid; i < n; i += half_n)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if(evect == rocblas_evect_none && tid > 0)
        return;

    rocblas_int m;
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += half_n)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
            }
        }
    }
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevj_heevj_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_Acpy,
                                         size_t* size_resarr,
                                         size_t* size_tbarr)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_Acpy = 0;
        *size_resarr = 0;
        *size_tbarr = 0;
        return;
    }

    rocblas_int even_n = n + n % 2;

    // size of temporary workspace for copying A
    *size_Acpy = sizeof(T) * n * n * batch_count;

    // size of array for per-thread residuals
    *size_resarr = sizeof(S) * n * batch_count;

    // size of array for temporary top/bottom pairs
    *size_tbarr = sizeof(rocblas_int) * even_n * batch_count;
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevj_heevj_argCheck(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || max_sweeps <= 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              const S abstol,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              const rocblas_stride strideW,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* Acpy,
                                              S* resarr,
                                              rocblas_int* tbarr)
{
    ROCSOLVER_ENTER("syevj_heevj", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n <= 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        // scalar case
        if(n == 1)
            ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream, evect, A,
                                    strideA, W, strideW, batch_count);

        return rocblas_status_success;
    }

    // absolute tolerance for evaluating when the algorithm has converged
    S atol = (abstol <= 0) ? 2 * get_safemin<T>() : abstol;

    rocblas_int blocks = (n - 1) / BS1 + 1;
    dim3 grid(blocks, batch_count, 1);
    dim3 threads(BS1, 1, 1);

    ROCSOLVER_LAUNCH_KERNEL(syevj_kernel<T>, grid, threads, 0, stream, evect, uplo, n, A, shiftA,
                            lda, strideA, atol, residual, max_sweeps, n_sweeps, W, strideW, info,
                            batch_count, Acpy, resarr, tbarr);

    return rocblas_status_success;
}
