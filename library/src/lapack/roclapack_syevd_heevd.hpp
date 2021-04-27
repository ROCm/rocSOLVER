/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "roclapack_syev_heev.hpp"

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevd_heevd_getMemorySize(const rocblas_evect evect,
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
    if(n == 0 || batch_count == 0)
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
    rocsolver_sytrd_hetrd_getMemorySize<T, BATCHED>(n, batch_count, size_scalars, &w1, &a1, &t1,
                                                    size_workArr);

    if(evect == rocblas_evect_original)
    {
        // TODO: Implement this part
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

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
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
    ROCSOLVER_ENTER("syevd_heevd", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info = 0
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        hipLaunchKernelGGL(scalar_case<T>, gridReset, threads, 0, stream, evect, A, strideA, D,
                           strideD, batch_count);
        return rocblas_status_success;
    }

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE,
                                   tau, n, batch_count, scalars, (T*)work_stack, Abyx_norms_tmptr,
                                   tmptau_trfact, workArr);

    if(evect != rocblas_evect_original)
    {
        // only compute eigenvalues
        rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                    (rocblas_int*)work_stack);
    }
    else
    {
        // TODO: Implement this part
        return rocblas_status_not_implemented;
    }

    return rocblas_status_success;
}
