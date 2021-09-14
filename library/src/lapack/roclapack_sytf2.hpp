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

#include "rocblas.hpp"
#include "rocsolver.h"

// number of threads for the sytf2 kernel (currently not tunable)
#define SYTF2_MAX_THDS 256

/** Device function to execute an optimized reduction to find the index of the
    maximum element of a given vector (iamax) **/
template <typename T>
__device__ void sytf2_iamax(const rocblas_int tid,
                            const rocblas_int n,
                            T* A,
                            const rocblas_int incA,
                            T* sval,
                            rocblas_int* sidx)
{
    using S = decltype(std::real(T{}));

    // local memory setup
    T val1, val2;
    rocblas_int idx1, idx2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    idx1 = INT_MAX;
    for(int i = tid; i < n; i += SYTF2_MAX_THDS)
    {
        val2 = A[i * incA];
        idx2 = i + 1; // add one to make it 1-based index
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            val1 = val2;
            idx1 = idx2;
        }
    }
    sval[tid] = val1;
    sidx[tid] = idx1;
    __syncthreads();

    if(n <= 1)
        return;

    /** <========= Next do the reduction on the shared memory array =========>
        (We need to execute the for loop
            for(j = SYTF2_MAX_THDS; j > 0; j>>=1)
        to have half of the active threads at each step
        reducing two elements in the shared array.
        As SYTF2_MAX_THDS is fixed to 256, we can unroll the loop manually) **/

    if(tid < 128)
    {
        val2 = sval[tid + 128];
        idx2 = sidx[tid + 128];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }
    __syncthreads();

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < 64)
    {
        val2 = sval[tid + 64];
        idx2 = sidx[tid + 64];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 32];
        idx2 = sidx[tid + 32];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 16];
        idx2 = sidx[tid + 16];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 8];
        idx2 = sidx[tid + 8];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 4];
        idx2 = sidx[tid + 4];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 2];
        idx2 = sidx[tid + 2];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 1];
        idx2 = sidx[tid + 1];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }
    __syncthreads();

    // after the reduction, the maximum of the elements is in sval[0] and sidx[0]
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTF2_MAX_THDS)
    sytf2_kernel_upper(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;

    int tid = hipThreadIdx_x;

    using S = decltype(std::real(T{}));
    const S alpha = (S)((1.0 + sqrt(17.0)) / 8.0);

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ rocblas_int info;
    int k = n - 1;
    int kstep, kp, kk;

    // shared arrays for iamax
    __shared__ T sval[SYTF2_MAX_THDS];
    __shared__ rocblas_int sidx[SYTF2_MAX_THDS];
    int i, j;
    S absakk, colmax, rowmax;

    if(tid == 0)
        info = 0;

    while(k >= 0)
    {
        kstep = 1;
        absakk = aabs<S>(A[k + k * lda]);

        // find max off-diagonal entry in column k
        sytf2_iamax(tid, k, A + k * lda, 1, sval, sidx);
        i = sidx[0] - 1;
        colmax = aabs<S>(sval[0]);
        __syncthreads();

        if(max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && info == 0)
                info = k + 1;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // find max off-diagonal entry in row i
                sytf2_iamax(tid, k - i, A + i + (i + 1) * lda, lda, sval, sidx);
                j = i + sidx[0];
                rowmax = aabs<S>(sval[0]);
                __syncthreads();

                if(i > 0)
                {
                    sytf2_iamax(tid, i, A + i * lda, 1, sval, sidx);
                    j = sidx[0] - 1;
                    rowmax = max(rowmax, aabs<S>(sval[0]));
                    __syncthreads();
                }

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(A[i + i * lda]) >= alpha * rowmax)
                    // interchange rows and columns k and i (1-by-1 block)
                    kp = i;
                else
                {
                    // interchange rows and columns k-1 and i (2-by-2 block)
                    kp = i;
                    kstep = 2;
                }
            }

            kk = k - kstep + 1;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                {
                    swap(A[kk + kk * lda], A[kp + kp * lda]);
                    if(kstep == 2)
                        swap(A[(k - 1) + k * lda], A[kp + k * lda]);
                }

                for(i = tid; i < kp; i += SYTF2_MAX_THDS)
                    swap(A[i + kk * lda], A[i + kp * lda]);
                for(i = tid; i < kk - kp - 1; i += SYTF2_MAX_THDS)
                    swap(A[(kp + i + 1) + kk * lda], A[kp + (kp + i + 1) * lda]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                // perform rank 1 update of A from [0,0] to [k-1,k-1] (syr)
                T r1 = T(1) / A[k + k * lda];
                for(j = tid; j < k; j += SYTF2_MAX_THDS)
                {
                    T r2 = -r1 * A[j + k * lda];
                    for(i = 0; i <= j; i++)
                        A[i + j * lda] = A[i + j * lda] + A[i + k * lda] * r2;
                }
                __syncthreads();

                // update column k (scal)
                for(j = tid; j < k; j += SYTF2_MAX_THDS)
                    A[j + k * lda] *= r1;
            }
            else
            {
                // 2-by-2 pivot block

                if(k > 1)
                {
                    // perform rank 2 update of A from [0,0] to [k-2,k-2]
                    T wk, wkm1;
                    T d12 = A[(k - 1) + k * lda];
                    T d22 = A[(k - 1) + (k - 1) * lda] / d12;
                    T d11 = A[k + k * lda] / d12;
                    d12 = T(1) / ((d11 * d22 - T(1)) * d12);
                    for(j = k - 2 - tid; j >= 0; j -= SYTF2_MAX_THDS)
                    {
                        wkm1 = d12 * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                        wk = d12 * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);
                        for(i = j; i >= 0; i--)
                            A[i + j * lda]
                                = A[i + j * lda] - A[i + k * lda] * wk - A[i + (k - 1) * lda] * wkm1;
                    }
                    __syncthreads();

                    // update columns k and k-1
                    for(j = k - 2 - tid; j >= 0; j -= SYTF2_MAX_THDS)
                    {
                        wkm1 = d12 * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                        wk = d12 * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);
                        A[j + k * lda] = wk;
                        A[j + (k - 1) * lda] = wkm1;
                    }
                }
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
    }

    if(tid == 0)
        infoA[bid] = info;
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTF2_MAX_THDS)
    sytf2_kernel_lower(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;

    int tid = hipThreadIdx_x;

    using S = decltype(std::real(T{}));
    const S alpha = (S)((1.0 + sqrt(17.0)) / 8.0);

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ rocblas_int info;
    int k = 0;
    int kstep, kp, kk;

    // shared arrays for iamax
    __shared__ T sval[SYTF2_MAX_THDS];
    __shared__ rocblas_int sidx[SYTF2_MAX_THDS];
    int i, j;
    S absakk, colmax, rowmax;

    if(tid == 0)
        info = 0;

    while(k < n)
    {
        kstep = 1;
        absakk = aabs<S>(A[k + k * lda]);

        // find max off-diagonal entry in column k
        sytf2_iamax(tid, n - k - 1, A + (k + 1) + k * lda, 1, sval, sidx);
        i = k + sidx[0];
        colmax = aabs<S>(sval[0]);
        __syncthreads();

        if(max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && info == 0)
                info = k + 1;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // find max off-diagonal entry in row i
                sytf2_iamax(tid, i - k, A + i + k * lda, lda, sval, sidx);
                j = k - 1 + sidx[0];
                rowmax = aabs<S>(sval[0]);
                __syncthreads();

                if(i < n - 1)
                {
                    sytf2_iamax(tid, n - i - 1, A + (i + 1) + i * lda, 1, sval, sidx);
                    j = i + sidx[0];
                    rowmax = max(rowmax, aabs<S>(sval[0]));
                    __syncthreads();
                }

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(A[i + i * lda]) >= alpha * rowmax)
                    // interchange rows and columns k and i (1-by-1 block)
                    kp = i;
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
                {
                    swap(A[kk + kk * lda], A[kp + kp * lda]);
                    if(kstep == 2)
                        swap(A[(k + 1) + k * lda], A[kp + k * lda]);
                }

                for(i = tid; i < n - kp - 1; i += SYTF2_MAX_THDS)
                    swap(A[(kp + i + 1) + kk * lda], A[(kp + i + 1) + kp * lda]);
                for(i = tid; i < kp - kk - 1; i += SYTF2_MAX_THDS)
                    swap(A[(kk + i + 1) + kk * lda], A[kp + (kk + i + 1) * lda]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                if(k < n - 1)
                {
                    // perform rank 1 update of A from [k+1,k+1] to [n-1,n-1] (syr)
                    T r1 = T(1) / A[k + k * lda];
                    for(j = tid; j < n - k - 1; j += SYTF2_MAX_THDS)
                    {
                        T r2 = -r1 * A[(k + j + 1) + k * lda];
                        for(i = j; i < n - k - 1; i++)
                            A[(k + i + 1) + (k + j + 1) * lda]
                                = A[(k + i + 1) + (k + j + 1) * lda] + A[(k + i + 1) + k * lda] * r2;
                    }
                    __syncthreads();

                    // update column k (scal)
                    for(j = tid; j < n - k - 1; j += SYTF2_MAX_THDS)
                        A[(k + j + 1) + k * lda] *= r1;
                }
            }
            else
            {
                // 2-by-2 pivot block

                if(k < n - 2)
                {
                    // perform rank 2 update of A from [k+2,k+2] to [n-1,n-1]
                    T wk, wkp1;
                    T d21 = A[(k + 1) + k * lda];
                    T d11 = A[(k + 1) + (k + 1) * lda] / d21;
                    T d22 = A[k + k * lda] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(j = k + 2 + tid; j < n; j += SYTF2_MAX_THDS)
                    {
                        wk = d21 * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                        wkp1 = d21 * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);
                        for(i = j; i < n; i++)
                            A[i + j * lda]
                                = A[i + j * lda] - A[i + k * lda] * wk - A[i + (k + 1) * lda] * wkp1;
                    }
                    __syncthreads();

                    // update columns k and k+1
                    for(j = k + 2 + tid; j < n; j += SYTF2_MAX_THDS)
                    {
                        wk = d21 * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                        wkp1 = d21 * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);
                        A[j + k * lda] = wk;
                        A[j + (k + 1) * lda] = wkp1;
                    }
                }
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

    if(tid == 0)
        infoA[bid] = info;
}

template <typename T>
rocblas_status rocsolver_sytf2_sytrf_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              rocblas_int* ipiv,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_sytf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("sytf2", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n == 0)
    {
        // set info = 0
        rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BLOCKSIZE, 1, 1);
        hipLaunchKernelGGL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    dim3 grid(1, batch_count, 1);
    dim3 threads(SYTF2_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
        hipLaunchKernelGGL(sytf2_kernel_upper<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                           strideA, ipiv, strideP, info);
    else
        hipLaunchKernelGGL(sytf2_kernel_lower<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                           strideA, ipiv, strideP, info);

    return rocblas_status_success;
}
