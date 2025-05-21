/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************** Main kernels *********************************************************/
/**************************************************************************************/

template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[j + i * ldf] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * Vp[i + j * ldv];
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + i * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * Vp[(n - k + i) + j * ldv];
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + (n - k + i) * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
    }
}

template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[j + i * ldf] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * conj(Vp[i + j * ldv]);
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + i * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * conj(Vp[(n - k + i) + j * ldv]);
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + (n - k + i) * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void set_tau(const rocblas_int k, T* tau, const rocblas_stride strideT)
{
    const auto b = hipBlockIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T* tp = tau + b * strideT;
        tp[i] = -tp[i];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_forward(const rocblas_storev storev,
                                           const rocblas_int n,
                                           const rocblas_int k,
                                           U VA,
                                           const rocblas_int shiftV,
                                           const rocblas_int ldv,
                                           const rocblas_stride strideV,
                                           T* tauA,
                                           const rocblas_stride strideT,
                                           T* FA,
                                           const rocblas_int shiftF,
                                           const rocblas_int ldfA,
                                           const rocblas_stride strideF)
{
    const rocblas_int bid = hipBlockIdx_y;
    const rocblas_int tid = hipThreadIdx_x;
    const rocblas_int tid_inc = hipBlockDim_x;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* tau = tauA + bid * strideT;
    T* Ftemp = load_ptr_batch<T>(FA, bid, shiftF, strideF);

    // shared memory setup
    extern __shared__ double lmem[];
    T* work = reinterpret_cast<T*>(lmem);
    T* F = work + k;
    rocblas_int ldf = k;

    // copy F to shared memory
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // --------- MAIN BODY ---------
    for(rocblas_int kk = 1; kk < k; kk++)
    {
        const rocblas_int mm = kk;
        const rocblas_int nn = n - 1 - kk;

        T* Fx = F + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + (kk + 1) + kk * ldv;

            // gemv (conjugate transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk + (kk + 1) * ldv;

            // gemv (no transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = i; j < mm; j++)
                temp += F[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_backward(const rocblas_storev storev,
                                            const rocblas_int n,
                                            const rocblas_int k,
                                            U VA,
                                            const rocblas_int shiftV,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            T* tauA,
                                            const rocblas_stride strideT,
                                            T* FA,
                                            const rocblas_int shiftF,
                                            const rocblas_int ldfA,
                                            const rocblas_stride strideF)
{
    const rocblas_int bid = hipBlockIdx_y;
    const rocblas_int tid = hipThreadIdx_x;
    const rocblas_int tid_inc = hipBlockDim_x;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* tau = tauA + bid * strideT;
    T* Ftemp = load_ptr_batch<T>(FA, bid, shiftF, strideF);

    // shared memory setup
    extern __shared__ double lmem[];
    T* work = reinterpret_cast<T*>(lmem);
    T* F = work + k;
    rocblas_int ldf = k;

    // copy F to shared memory
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = 0; j <= i; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // --------- MAIN BODY ---------
    for(rocblas_int kk = k - 2; kk >= 0; kk--)
    {
        const rocblas_int mm = k - kk - 1;
        const rocblas_int nn = n - k + kk;

        T* Fm = F + (kk + 1) + (kk + 1) * ldf;
        T* Fx = F + (kk + 1) + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk * ldv;

            // gemv (conjugate transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + kk;

            // gemv (no transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = 0; j <= i; j++)
                temp += Fm[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = 0; j <= i; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

/******************* Host functions *********************************************/
/*******************************************************************************/

template <bool BATCHED, typename T>
void rocsolver_larft_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of re-usable workspace
    *size_work = sizeof(T) * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_larft_argCheck(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int ldv,
                                        const rocblas_int ldf,
                                        T V,
                                        U tau,
                                        U F)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    if(n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if((row && ldv < k) || (!row && ldv < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !V) || !tau || !F)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status larft_recursive_forward(rocblas_handle handle,
                                       const rocblas_storev storev,
                                       const rocblas_int n,
                                       const rocblas_int k,
                                       U V,
                                       const rocblas_int shiftV,
                                       const rocblas_int ldv,
                                       const rocblas_stride strideV,
                                       T* tau,
                                       const rocblas_stride strideT,
                                       T* F,
                                       const rocblas_int shiftF,
                                       const rocblas_int ldf,
                                       const rocblas_stride strideF,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       T* work,
                                       T** workArr,
                                       size_t sharedMemPerBlock)
{
    if(k < 2 || n < 2)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    size_t lmemsize = sizeof(T) * (k + 1) * k;
    if(k <= LARFT_SWITCHSIZE && lmemsize <= sharedMemPerBlock)
    {
        ROCSOLVER_LAUNCH_KERNEL(larft_kernel_forward, dim3(1, batch_count),
                                dim3(LARFT_SWITCHSIZE, 1), lmemsize, stream, storev, n, k, V,
                                shiftV, ldv, strideV, tau, strideT, F, shiftF, ldf, strideF);
        return rocblas_status_success;
    }

    rocblas_int blocks = (k - 1) / BS2 + 1;
    rocblas_int l = k / 2;
    T one = 1;
    T minone = -1;

    if(storev == rocblas_column_wise) // QR
    {
        // F_1_1
        larft_recursive_forward(handle, storev, n, l, V, shiftV, ldv, strideV, tau, strideT, F,
                                shiftF, ldf, strideF, batch_count, scalars, work, workArr,
                                sharedMemPerBlock);
        // F_2_2
        larft_recursive_forward(handle, storev, n - l, k - l, V, shiftV + (l * ldv + l), ldv,
                                strideV, tau + l, strideT, F, shiftF + (l * ldf + l), ldf, strideF,
                                batch_count, scalars, work, workArr, sharedMemPerBlock);

        // F_1_2 = V_2_1^T
        ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, rocblas_operation_conjugate_transpose,
                                k - l, l, V, shiftV + l, ldv, strideV, F, shiftF + l * ldf, ldf,
                                strideF);

        // F_1_2 = F_1_2 * V_2_2
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_unit, l, k - l, &one, 0, V, shiftV + l * ldv + l, ldv,
                         strideV, F, shiftF + l * ldf, ldf, strideF, batch_count, workArr);

        // F_1_2 = V_3_1^T * V_3_2 + F_1_2
        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, l,
                       k - l, n - k, &one, V, shiftV + k, 1, ldv, strideV, V, shiftV + l * ldv + k,
                       1, ldv, strideV, &one, F, shiftF + ldf * l, 1, ldf, strideF, batch_count,
                       workArr);

        // F_1_2 = -F_1_1 * F_1_2
        rocblasCall_trmm(handle, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &minone, 0, F, shiftF, ldf, strideF,
                         F, shiftF + l * ldf, ldf, strideF, batch_count, workArr);

        // F_1_2 = F_1_2 * F_2_2
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &one, 0, F, shiftF + ldf * l + l, ldf,
                         strideF, F, shiftF + l * ldf, ldf, strideF, batch_count, workArr);
    }
    else // LQ
    {
        // F_1_1
        larft_recursive_forward(handle, storev, n, l, V, shiftV, ldv, strideV, tau, strideT, F,
                                shiftF, ldf, strideF, batch_count, scalars, work, workArr,
                                sharedMemPerBlock);
        // F_2_2
        larft_recursive_forward(handle, storev, n - l, k - l, V, shiftV + (l * ldv + l), ldv,
                                strideV, tau + l, strideT, F, shiftF + (l * ldf + l), ldf, strideF,
                                batch_count, scalars, work, workArr, sharedMemPerBlock);

        // F_1_2 = V_1_2
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                                stream, l, k - l, V, shiftV + l * ldv, ldv, strideV, F,
                                shiftF + l * ldf, ldf, strideF);

        // F_1_2 = F_1_2 * V_2_2^T
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_upper,
                         rocblas_operation_conjugate_transpose, rocblas_diagonal_unit, l, k - l,
                         &one, 0, V, shiftV + l * ldv + l, ldv, strideV, F, shiftF + l * ldf, ldf,
                         strideF, batch_count, workArr);

        // F_1_2 = V_1_3 * V_2_3^T + F_1_2
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, l,
                       k - l, n - k, &one, V, shiftV + ldv * k, 1, ldv, strideV, V,
                       shiftV + k * ldv + l, 1, ldv, strideV, &one, F, shiftF + ldf * l, 1, ldf,
                       strideF, batch_count, workArr);

        // F_1_2 = -F_1_1 * F_1_2
        rocblasCall_trmm(handle, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &minone, 0, F, shiftF, ldf, strideF,
                         F, shiftF + l * ldf, ldf, strideF, batch_count, workArr);

        // F_1_2 = F_1_2 * F_2_2
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &one, 0, F, shiftF + ldf * l + l, ldf,
                         strideF, F, shiftF + l * ldf, ldf, strideF, batch_count, workArr);
    }
    return rocblas_status_success;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status larft_recursive_backward(rocblas_handle handle,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F,
                                        const rocblas_int shiftF,
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T** workArr,
                                        size_t sharedMemPerBlock)
{
    if(k < 2 || n < 2)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    size_t lmemsize = sizeof(T) * (k + 1) * k;
    if(k <= LARFT_SWITCHSIZE && lmemsize <= sharedMemPerBlock)
    {
        ROCSOLVER_LAUNCH_KERNEL(larft_kernel_backward, dim3(1, batch_count),
                                dim3(LARFT_SWITCHSIZE, 1), lmemsize, stream, storev, n, k, V,
                                shiftV, ldv, strideV, tau, strideT, F, shiftF, ldf, strideF);
        return rocblas_status_success;
    }

    rocblas_int blocks = (k - 1) / BS2 + 1;
    rocblas_int l = k / 2;
    T one = 1;
    T minone = -1;

    if(storev == rocblas_column_wise) // QL
    {
        // F_1_1
        larft_recursive_backward(handle, storev, n - l, k - l, V, shiftV, ldv, strideV, tau,
                                 strideT, F, shiftF, ldf, strideF, batch_count, scalars, work,
                                 workArr, sharedMemPerBlock);
        // F_2_2
        larft_recursive_backward(handle, storev, n, l, V, shiftV + (k - l) * ldv, ldv, strideV,
                                 tau + (k - l), strideT, F, shiftF + (k - l) * ldf + (k - l), ldf,
                                 strideF, batch_count, scalars, work, workArr, sharedMemPerBlock);

        // F_1_2 = V_2_1^T
        ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, rocblas_operation_conjugate_transpose,
                                k - l, l, V, shiftV + (n - k) + ldv * (k - l), ldv, strideV, F,
                                shiftF + (k - l), ldf, strideF);

        // F_2_1 = F_2_1 * V_2_1
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_unit, l, k - l, &one, 0, V, shiftV + (n - k), ldv,
                         strideV, F, shiftF + (k - l), ldf, strideF, batch_count, workArr);

        // F_2_1 = V_2_2^T * V_2_1 + F_2_1
        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, l,
                       k - l, n - k, &one, V, shiftV + ldv * (k - l), 1, ldv, strideV, V, shiftV, 1,
                       ldv, strideV, &one, F, shiftF + (k - l), 1, ldf, strideF, batch_count,
                       workArr);

        // F_2_1 = -F_2_2 * F_2_1
        rocblasCall_trmm(handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &minone, 0, F,
                         shiftF + (k - l) * ldf + (k - l), ldf, strideF, F, shiftF + (k - l), ldf,
                         strideF, batch_count, workArr);

        // F_2_1 = F_2_1 * F_1_1
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &one, 0, F, shiftF, ldf, strideF, F,
                         shiftF + (k - l), ldf, strideF, batch_count, workArr);
    }
    else // RQ
    {
        // F_1_1
        larft_recursive_backward(handle, storev, n - l, k - l, V, shiftV, ldv, strideV, tau,
                                 strideT, F, shiftF, ldf, strideF, batch_count, scalars, work,
                                 workArr, sharedMemPerBlock);
        // F_2_2
        larft_recursive_backward(handle, storev, n, l, V, shiftV + (k - l), ldv, strideV,
                                 tau + (k - l), strideT, F, shiftF + (k - l) + ldf * (k - l), ldf,
                                 strideF, batch_count, scalars, work, workArr, sharedMemPerBlock);

        // F_2_1 = V_2_2
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                                stream, l, k - l, V, shiftV + (k - l) + ldv * (n - k), ldv, strideV,
                                F, shiftF + (k - l), ldf, strideF);

        // F_2_1 = F_2_1 * V_1_2^T
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_lower,
                         rocblas_operation_conjugate_transpose, rocblas_diagonal_unit, l, k - l,
                         &one, 0, V, shiftV + ldv * (n - k), ldv, strideV, F, shiftF + (k - l), ldf,
                         strideF, batch_count, workArr);

        // F_2_1 = V_2_1 * V_1_1^T + F_2_1
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, l,
                       k - l, n - k, &one, V, shiftV + (k - l), 1, ldv, strideV, V, shiftV, 1, ldv,
                       strideV, &one, F, shiftF + (k - l), 1, ldf, strideF, batch_count, workArr);

        // F_2_1 = -F_2_2 * F_2_1
        rocblasCall_trmm(handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &minone, 0, F,
                         shiftF + (k - l) + ldf * (k - l), ldf, strideF, F, shiftF + (k - l), ldf,
                         strideF, batch_count, workArr);

        // F_2_1 = F_2_1 * F_1_1
        rocblasCall_trmm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_non_unit, l, k - l, &one, 0, F, shiftF, ldf, strideF, F,
                         shiftF + (k - l), ldf, strideF, batch_count, workArr);
    }
    return rocblas_status_success;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_template(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F,
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T** workArr)
{
    ROCSOLVER_ENTER("larft", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_stride stridew = rocblas_stride(k);
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_fill uplo;
    rocblas_operation trans;

    // Fix diagonal of T, make zero the not used triangular part,
    // setup tau (changing signs) and account for the non-stored 1's on the
    // householder vectors
    rocblas_int blocks = (k - 1) / 32 + 1;
    ROCSOLVER_LAUNCH_KERNEL(set_triangular, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                            stream, n, k, V, shiftV, ldv, strideV, tau, strideT, F, ldf, strideF,
                            direct, storev);
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    if(direct == rocblas_forward_direction)
    {
        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        ROCBLAS_CHECK(larft_recursive_forward<T, U, COMPLEX>(
            handle, storev, n, k, V, shiftV, ldv, strideV, tau, strideT, F, 0, ldf, strideF,
            batch_count, scalars, work, workArr, props.sharedMemPerBlock));
    }
    else
    {
        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        ROCBLAS_CHECK(larft_recursive_backward<T, U, COMPLEX>(
            handle, storev, n, k, V, shiftV, ldv, strideV, tau, strideT, F, 0, ldf, strideF,
            batch_count, scalars, work, workArr, props.sharedMemPerBlock));
    }

    // restore tau
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
