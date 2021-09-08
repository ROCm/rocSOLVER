/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

// number of threads for the lasyf kernel (currently not tunable)
#define LASYF_MAX_THDS 256

/** Device function to compute y = y - Ax (gemv) **/
template <typename T>
__device__ void lasyf_gemv(const rocblas_int tid,
                           const rocblas_int m,
                           const rocblas_int n,
                           T* A,
                           const rocblas_int lda,
                           T* x,
                           const rocblas_int incx,
                           T* y,
                           const rocblas_int incy,
                           T* temp)
{
    for(int i = tid; i < m; i += LASYF_MAX_THDS)
    {
        temp[tid] = y[i * incy];
        for(int j = 0; j < n; j++)
            temp[tid] -= A[i + j * lda] * x[j * incx];
        y[i * incy] = temp[tid];
    }
    __syncthreads();
}

/** Device function to compute C = C - A B' (gemm) **/
template <typename T>
__device__ void lasyf_gemm(const rocblas_int tid,
                           const rocblas_int m,
                           const rocblas_int n,
                           const rocblas_int k,
                           T* A,
                           const rocblas_int lda,
                           T* B,
                           const rocblas_int ldb,
                           T* C,
                           const rocblas_int ldc,
                           T* temp)
{
    for(int e = tid; e < m * n; e += LASYF_MAX_THDS)
    {
        int i = e % m;
        int j = e / m;
        temp[tid] = C[i + j * ldc];
        for(int l = 0; l < k; l++)
            temp[tid] -= A[i + l * lda] * B[j + l * ldb];
        C[i + j * ldc] = temp[tid];
    }
    __syncthreads();
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LASYF_MAX_THDS)
    lasyf_kernel_upper(const rocblas_int n,
                       const rocblas_int nb,
                       rocblas_int* kb,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA,
                       T* WA)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;

    int tid = hipThreadIdx_x;

    using S = decltype(std::real(T{}));
    const S alpha = (S)((1.0 + sqrt(17.0)) / 8.0);
    const int ldw = n;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * nb);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ rocblas_int info;
    int k = n - 1;
    int kstep, kp, kk, kw, kkw;

    // shared arrays for iamax
    __shared__ T sval[LASYF_MAX_THDS];
    __shared__ rocblas_int sidx[LASYF_MAX_THDS];
    int idx, i, j;
    S absakk, colmax, rowmax;

    if(tid == 0)
        info = 0;

    kw = nb + k - n;
    while(k >= 0 && (k > n - nb || nb == n))
    {
        // copy column k of A to column kw of W and update
        for(idx = tid; idx <= k; idx += LASYF_MAX_THDS)
            W[idx + kw * ldw] = A[idx + k * lda];
        __syncthreads();
        if(k < n - 1)
            lasyf_gemv(tid, k + 1, n - k - 1, A + (k + 1) * lda, lda, W + k + (kw + 1) * ldw, ldw,
                       W + kw * ldw, 1, sval);

        kstep = 1;
        absakk = aabs<S>(W[k + kw * ldw]);

        // find max off-diagonal entry in column k
        iamax<LASYF_MAX_THDS>(tid, k, W + kw * ldw, 1, sval, sidx);
        __syncthreads();
        i = sidx[0] - 1;
        colmax = aabs<S>(sval[0]);
        __syncthreads();

        if(max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && info == 0)
                info = k;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // copy column i of A to column kw-1 of W and update
                for(idx = tid; idx <= i; idx += LASYF_MAX_THDS)
                    W[idx + (kw - 1) * ldw] = A[idx + i * lda];
                for(idx = tid; idx < k - i; idx += LASYF_MAX_THDS)
                    W[(i + idx + 1) + (kw - 1) * ldw] = A[i + (i + idx + 1) * lda];
                __syncthreads();
                if(k < n - 1)
                    lasyf_gemv(tid, k + 1, n - k - 1, A + (k + 1) * lda, lda,
                               W + i + (kw + 1) * ldw, ldw, W + (kw - 1) * ldw, 1, sval);

                // find max off-diagonal entry in row i
                iamax<LASYF_MAX_THDS>(tid, k - i, W + (i + 1) + (kw - 1) * ldw, 1, sval, sidx);
                __syncthreads();
                j = i + sidx[0];
                rowmax = aabs<S>(sval[0]);
                __syncthreads();

                if(i > 0)
                {
                    iamax<LASYF_MAX_THDS>(tid, i, W + (kw - 1) * ldw, 1, sval, sidx);
                    __syncthreads();
                    j = sidx[0] - 1;
                    rowmax = max(rowmax, aabs<S>(sval[0]));
                    __syncthreads();
                }

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(W[i + (kw - 1) * ldw]) >= alpha * rowmax)
                {
                    // interchange rows and columns k and i (1-by-1 block)
                    kp = i;

                    // copy column kw-1 of W to column kw of W
                    for(idx = tid; idx <= k; idx += LASYF_MAX_THDS)
                        W[idx + kw * ldw] = W[idx + (kw - 1) * ldw];
                    __syncthreads();
                }
                else
                {
                    // interchange rows and columns k-1 and i (2-by-2 block)
                    kp = i;
                    kstep = 2;
                }
            }

            kk = k - kstep + 1;
            kkw = nb + kk - n;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                    A[kp + kp * lda] = A[kk + kk * lda];

                for(idx = tid; idx < kk - kp - 1; idx += LASYF_MAX_THDS)
                    A[kp + (kp + idx + 1) * lda] = A[(kp + idx + 1) + kk * lda];
                for(idx = tid; idx < kp; idx += LASYF_MAX_THDS)
                    A[idx + kp * lda] = A[idx + kk * lda];
                __syncthreads();
                for(idx = tid; idx < n - k - 1; idx += LASYF_MAX_THDS)
                    swap(A[kk + (k + idx + 1) * lda], A[kp + (k + idx + 1) * lda]);
                for(idx = tid; idx < n - kk; idx += LASYF_MAX_THDS)
                    swap(W[kk + (kkw + idx) * ldw], W[kp + (kkw + idx) * ldw]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                T r1 = T(1) / W[k + kw * ldw];
                if(tid == 0)
                    A[k + k * lda] = W[k + kw * ldw];
                for(idx = tid; idx < k; idx += LASYF_MAX_THDS)
                    A[idx + k * lda] = r1 * W[idx + kw * ldw];
                __syncthreads();
            }
            else
            {
                // 2-by-2 pivot block

                if(k > 1)
                {
                    T d21 = W[(k - 1) + kw * ldw];
                    T d11 = W[k + kw * ldw] / d21;
                    T d22 = W[(k - 1) + (kw - 1) * ldw] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(idx = tid; idx <= k - 2; idx += LASYF_MAX_THDS)
                    {
                        A[idx + (k - 1) * lda]
                            = d21 * (d11 * W[idx + (kw - 1) * ldw] - W[idx + kw * ldw]);
                        A[idx + k * lda] = d21 * (d22 * W[idx + kw * ldw] - W[idx + (kw - 1) * ldw]);
                    }
                }

                if(tid == 0)
                {
                    A[(k - 1) + (k - 1) * lda] = W[(k - 1) + (kw - 1) * ldw];
                    A[(k - 1) + k * lda] = W[(k - 1) + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];
                }
                __syncthreads();
            }
        }

        // update ipiv (1-based index to match LAPACK)
        if(tid == 0)
        {
            if(kstep == 1)
                ipiv[k] = kp + 1;
            else
            {
                ipiv[k] = -(kp + 1);
                ipiv[k - 1] = -(kp + 1);
            }
        }

        k -= kstep;
        kw = nb + k - n;
    }

    // update A from [0,0] to [k,k], nb columns at a time
    for(j = (k / nb) * nb; j >= 0; j -= nb)
    {
        idx = min(nb, k - j + 1); // jb
        for(i = j; i < j + idx; i++)
            lasyf_gemv(tid, i - j + 1, n - k - 1, A + j + (k + 1) * lda, lda,
                       W + i + (kw + 1) * ldw, ldw, A + j + i * lda, 1, sval);
        lasyf_gemm(tid, j, idx, n - k - 1, A + (k + 1) * lda, lda, W + j + (kw + 1) * ldw, ldw,
                   A + j * lda, lda, sval);
    }

    // partially undo interchanges to put U12 in standard form
    j = k + 1;
    while(j < n - 1)
    {
        __syncthreads();
        kk = j; // jj
        kp = ipiv[j]; // jp
        if(kp < 0)
        {
            kp = -kp - 1;
            j++;
        }
        else
            kp = kp - 1;

        j++;
        if(kp != kk && j < n)
        {
            for(idx = tid; idx < n - j; idx += LASYF_MAX_THDS)
                swap(A[kp + (j + idx) * lda], A[kk + (j + idx) * lda]);
        }
    }

    if(tid == 0)
    {
        kb[bid] = n - k - 1;
        infoA[bid] = info;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LASYF_MAX_THDS)
    lasyf_kernel_lower(const rocblas_int n,
                       const rocblas_int nb,
                       rocblas_int* kb,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA,
                       T* WA)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;

    int tid = hipThreadIdx_x;

    using S = decltype(std::real(T{}));
    const S alpha = (S)((1.0 + sqrt(17.0)) / 8.0);
    const int ldw = n;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * nb);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ rocblas_int info;
    int k = 0;
    int kstep, kp, kk;

    // shared arrays for iamax
    __shared__ T sval[LASYF_MAX_THDS];
    __shared__ rocblas_int sidx[LASYF_MAX_THDS];
    int idx, i, j;
    S absakk, colmax, rowmax;

    if(tid == 0)
        info = 0;

    while(k < n && (k < nb - 1 || nb == n))
    {
        // copy column k of A to column k of W and update
        for(idx = tid; idx < n - k; idx += LASYF_MAX_THDS)
            W[(k + idx) + k * ldw] = A[(k + idx) + k * lda];
        __syncthreads();
        lasyf_gemv(tid, n - k, k, A + k, lda, W + k, ldw, W + k + k * ldw, 1, sval);

        kstep = 1;
        absakk = aabs<S>(W[k + k * ldw]);

        // find max off-diagonal entry in column k
        iamax<LASYF_MAX_THDS>(tid, n - k - 1, W + (k + 1) + k * ldw, 1, sval, sidx);
        __syncthreads();
        i = k + sidx[0];
        colmax = aabs<S>(sval[0]);
        __syncthreads();

        if(max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && info == 0)
                info = k;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // copy column i of A to column k+1 of W and update
                for(idx = tid; idx < i - k; idx += LASYF_MAX_THDS)
                    W[(k + idx) + (k + 1) * ldw] = A[i + (k + idx) * lda];
                for(idx = tid; idx < n - i; idx += LASYF_MAX_THDS)
                    W[(i + idx) + (k + 1) * ldw] = A[(i + idx) + i * lda];
                __syncthreads();
                lasyf_gemv(tid, n - k, k, A + k, lda, W + i, ldw, W + k + (k + 1) * ldw, 1, sval);

                // find max off-diagonal entry in row i
                iamax<LASYF_MAX_THDS>(tid, i - k, W + k + (k + 1) * ldw, 1, sval, sidx);
                __syncthreads();
                j = k - 1 + sidx[0];
                rowmax = aabs<S>(sval[0]);
                __syncthreads();

                if(i < n - 1)
                {
                    iamax<LASYF_MAX_THDS>(tid, n - i - 1, W + (i + 1) + (k + 1) * ldw, 1, sval, sidx);
                    __syncthreads();
                    j = i + sidx[0];
                    rowmax = max(rowmax, aabs<S>(sval[0]));
                    __syncthreads();
                }

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(W[i + (k + 1) * ldw]) >= alpha * rowmax)
                {
                    // interchange rows and columns k and i (1-by-1 block)
                    kp = i;

                    // copy column kw-1 of W to column kw of W
                    for(idx = tid; idx < n - k; idx += LASYF_MAX_THDS)
                        W[(k + idx) + k * ldw] = W[(k + idx) + (k + 1) * ldw];
                    __syncthreads();
                }
                else
                {
                    // interchange rows and columns k+1 and i (2-by-2 block)
                    kp = i;
                    kstep = 2;
                }
            }

            kk = k + kstep - 1;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                    A[kp + kp * lda] = A[kk + kk * lda];

                for(idx = tid; idx < kp - kk - 1; idx += LASYF_MAX_THDS)
                    A[kp + (kk + idx + 1) * lda] = A[(kk + idx + 1) + kk * lda];
                for(idx = tid; idx < n - kp - 1; idx += LASYF_MAX_THDS)
                    A[(kp + idx + 1) + kp * lda] = A[(kp + idx + 1) + kk * lda];
                __syncthreads();
                for(idx = tid; idx < k; idx += LASYF_MAX_THDS)
                    swap(A[kk + idx * lda], A[kp + idx * lda]);
                for(idx = tid; idx <= kk; idx += LASYF_MAX_THDS)
                    swap(W[kk + idx * ldw], W[kp + idx * ldw]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                T r1 = T(1) / W[k + k * ldw];
                if(tid == 0)
                    A[k + k * lda] = W[k + k * ldw];
                for(idx = tid; idx < n - k - 1; idx += LASYF_MAX_THDS)
                    A[(k + idx + 1) + k * lda] = r1 * W[(k + idx + 1) + k * ldw];
                __syncthreads();
            }
            else
            {
                // 2-by-2 pivot block

                if(k < n - 2)
                {
                    T d21 = W[(k + 1) + k * ldw];
                    T d11 = W[(k + 1) + (k + 1) * ldw] / d21;
                    T d22 = W[k + k * ldw] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(idx = k + 2 + tid; idx < n; idx += LASYF_MAX_THDS)
                    {
                        A[idx + k * lda] = d21 * (d11 * W[idx + k * ldw] - W[idx + (k + 1) * ldw]);
                        A[idx + (k + 1) * lda]
                            = d21 * (d22 * W[idx + (k + 1) * ldw] - W[idx + k * ldw]);
                    }
                }

                if(tid == 0)
                {
                    A[k + k * lda] = W[k + k * ldw];
                    A[(k + 1) + k * lda] = W[(k + 1) + k * ldw];
                    A[(k + 1) + (k + 1) * lda] = W[(k + 1) + (k + 1) * ldw];
                }
                __syncthreads();
            }
        }

        // update ipiv (1-based index to match LAPACK)
        if(tid == 0)
        {
            if(kstep == 1)
                ipiv[k] = kp + 1;
            else
            {
                ipiv[k] = -(kp + 1);
                ipiv[k + 1] = -(kp + 1);
            }
        }

        k += kstep;
    }

    // update A from [k,k] to [n-1,n-1], nb columns at a time
    for(j = k; j < n; j += nb)
    {
        idx = min(nb, n - j); // jb
        for(i = j; i < j + idx; i++)
            lasyf_gemv(tid, j + idx - i, k, A + i, lda, W + i, ldw, A + i + i * lda, 1, sval);
        if(j + idx < n)
            lasyf_gemm(tid, n - j - idx, idx, k, A + (j + idx), lda, W + j, ldw,
                       A + (j + idx) + j * lda, lda, sval);
    }

    // partially undo interchanges to put L21 in standard form
    j = k - 1;
    while(j > 0)
    {
        __syncthreads();
        kk = j; // jj
        kp = ipiv[j]; // jp
        if(kp < 0)
        {
            kp = -kp - 1;
            j--;
        }
        else
            kp = kp - 1;

        j--;
        if(kp != kk && j >= 0)
        {
            for(idx = tid; idx <= j; idx += LASYF_MAX_THDS)
                swap(A[kp + idx * lda], A[kk + idx * lda]);
        }
    }

    if(tid == 0)
    {
        kb[bid] = k;
        infoA[bid] = info;
    }
}

template <typename T>
void rocsolver_lasyf_getMemorySize(const rocblas_int n,
                                   const rocblas_int nb,
                                   const rocblas_int batch_count,
                                   size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || nb == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // size of workspace
    *size_work = sizeof(T) * n * nb * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        U kb,
                                        T A,
                                        U ipiv,
                                        U info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nb < 0 || nb > n || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((batch_count && !kb) || (n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        rocblas_int* kb,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* work)
{
    ROCSOLVER_ENTER("lasyf", "uplo:", uplo, "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n == 0 || nb == 0)
    {
        // set info = 0
        rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BLOCKSIZE, 1, 1);
        hipLaunchKernelGGL(reset_info, gridReset, threadsReset, 0, stream, kb, batch_count, 0);
        hipLaunchKernelGGL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    dim3 grid(1, batch_count, 1);
    dim3 threads(LASYF_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
        hipLaunchKernelGGL(lasyf_kernel_upper<T>, grid, threads, 0, stream, n, nb, kb, A, shiftA,
                           lda, strideA, ipiv, strideP, info, work);
    else
        hipLaunchKernelGGL(lasyf_kernel_lower<T>, grid, threads, 0, stream, n, nb, kb, A, shiftA,
                           lda, strideA, ipiv, strideP, info, work);

    return rocblas_status_success;
}
