/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_orgtr_ungtr.hpp"
#include "auxiliary/rocauxiliary_steqr.hpp"
#include "auxiliary/rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "roclapack_sytrd_hetrd.hpp"
#include "rocsolver/rocsolver.h"

/** Set results for the scalar case (n=1) **/
template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void scalar_case(const rocblas_evect evect,
                                  U AA,
                                  const rocblas_stride strideA,
                                  T* DD,
                                  const rocblas_stride strideD,
                                  rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        T* A = load_ptr_batch<T>(AA, b, 0, strideA);
        T* D = DD + b * strideD;
        D[0] = std::real(A[0]);

        if(evect == rocblas_evect_original)
            A[0] = T(1);
    }
}

template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void scalar_case(const rocblas_evect evect,
                                  U AA,
                                  const rocblas_stride strideA,
                                  S* DD,
                                  const rocblas_stride strideD,
                                  rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        T* A = load_ptr_batch<T>(AA, b, 0, strideA);
        S* D = DD + b * strideD;
        D[0] = A[0].real();

        if(evect == rocblas_evect_original)
            A[0] = T(1);
    }
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syev_heev_getMemorySize(const rocblas_evect evect,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int batch_count,
                                       size_t* size_scalars,
                                       size_t* size_work_stack,
                                       size_t* size_Abyx_norms_tmptr,
                                       size_t* size_tmptau_trfact,
                                       size_t* size_tau,
                                       size_t* size_workArr)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_stack = 0;
        *size_Abyx_norms_tmptr = 0;
        *size_tmptau_trfact = 0;
        *size_tau = 0;
        *size_workArr = 0;
        return;
    }

    size_t unused;
    size_t w1 = 0, w2 = 0, w3 = 0;
    size_t a1 = 0, a2 = 0;
    size_t t1 = 0, t2 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w1, &a1, &t1,
                                                    size_workArr);

    if(evect == rocblas_evect_original)
    {
        // extra requirements for orgtr/ungtr
        rocsolver_orgtr_ungtr_getMemorySize<BATCHED, T>(uplo, n, batch_count, &unused, &w2, &a2,
                                                        &t2, &unused);

        // extra requirements for computing eigenvalues and vectors (steqr)
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &w3);
    }
    else
    {
        // extra requirements for computing only the eigenvalues (sterf)
        rocsolver_sterf_getMemorySize<T>(n, batch_count, &w2);
    }

    // get max values
    *size_work_stack = std::max({w1, w2, w3});
    *size_Abyx_norms_tmptr = std::max(a1, a2);
    *size_tmptau_trfact = std::max(t1, t2);

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syev_heev_argCheck(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            T A,
                                            const rocblas_int lda,
                                            S* D,
                                            S* E,
                                            rocblas_int* info,
                                            const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !E) || (n && !D) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
rocblas_status rocsolver_syev_heev_template(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            W A,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            S* D,
                                            const rocblas_stride strideD,
                                            S* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count,
                                            T* scalars,
                                            void* work_stack,
                                            T* Abyx_norms_tmptr,
                                            T* tmptau_trfact,
                                            T* tau,
                                            T** workArr)
{
    ROCSOLVER_ENTER("syev_heev", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threads, 0, stream, evect, A, strideA, D,
                                strideD, batch_count);
        return rocblas_status_success;
    }

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                            strideE, tau, n, batch_count, scalars, (T*)work_stack,
                                            Abyx_norms_tmptr, tmptau_trfact, workArr);

    if(evect != rocblas_evect_original)
    {
        // only compute eigenvalues
        rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                    (rocblas_int*)work_stack);
    }
    else
    {
        // update orthogonal matrix
        rocsolver_orgtr_ungtr_template<BATCHED, STRIDED, T>(
            handle, uplo, n, A, shiftA, lda, strideA, tau, n, batch_count, scalars, (T*)work_stack,
            Abyx_norms_tmptr, tmptau_trfact, workArr);

        // compute eigenvalues and eigenvectors
        rocsolver_steqr_template<T>(handle, evect, n, D, 0, strideD, E, 0, strideE, A, shiftA, lda,
                                    strideA, info, batch_count, work_stack);
    }

    return rocblas_status_success;
}
