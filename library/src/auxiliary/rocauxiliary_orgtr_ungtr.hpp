/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocauxiliary_orgql_ungql.hpp"
#include "rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <bool BATCHED, typename T>
void rocsolver_orgtr_ungtr_getMemorySize(const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_Abyx_tmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_Abyx_tmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    size_t w1 = sizeof(T) * batch_count * (n - 1) * n / 2;
    size_t w2;
    if(uplo == rocblas_fill_upper)
    {
        // requirements for calling orgql/ungql
        rocsolver_orgql_ungql_getMemorySize<BATCHED, T>(n - 1, n - 1, n - 1, batch_count,
                                                        size_scalars, &w2, size_Abyx_tmptr,
                                                        size_trfact, size_workArr);
    }

    else
    {
        // requirements for calling orgqr/ungqr
        rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(n - 1, n - 1, n - 1, batch_count,
                                                        size_scalars, &w2, size_Abyx_tmptr,
                                                        size_trfact, size_workArr);
    }
    *size_work = max(w1, w2);
}

template <typename T, typename U>
rocblas_status rocsolver_orgtr_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        U ipiv)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgtr_ungtr_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* Abyx_tmptr,
                                              T* trfact,
                                              T** workArr)
{
    ROCSOLVER_ENTER("orgtr_ungtr", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_stride strideW = rocblas_stride(n - 1) * n / 2; // number of elements to copy
    rocblas_int ldw = n - 1;
    rocblas_int blocks = (n - 2) / BS2 + 1;

    if(uplo == rocblas_fill_upper)
    {
        // shift the householder vectors provided by gebrd as they come above the
        // first superdiagonal and must be shifted left

        // copy
        ROCSOLVER_LAUNCH_KERNEL(copyshift_left<T>, dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, true, n - 1, A, shiftA, lda, strideA,
                                work, 0, ldw, strideW);

        // shift
        ROCSOLVER_LAUNCH_KERNEL(copyshift_left<T>, dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, false, n - 1, A, shiftA, lda, strideA,
                                work, 0, ldw, strideW);

        // result
        rocsolver_orgql_ungql_template<BATCHED, STRIDED, T>(
            handle, n - 1, n - 1, n - 1, A, shiftA, lda, strideA, ipiv, strideP, batch_count,
            scalars, work, Abyx_tmptr, trfact, workArr);
    }

    else
    {
        // shift the householder vectors provided by gebrd as they come below the
        // first subdiagonal and must be shifted right

        // copy
        ROCSOLVER_LAUNCH_KERNEL(copyshift_right<T>, dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, true, n - 1, A, shiftA, lda, strideA,
                                work, 0, ldw, strideW);

        // shift
        ROCSOLVER_LAUNCH_KERNEL(copyshift_right<T>, dim3(blocks, blocks, batch_count),
                                dim3(BS2, BS2), 0, stream, false, n - 1, A, shiftA, lda, strideA,
                                work, 0, ldw, strideW);

        // result
        rocsolver_orgqr_ungqr_template<BATCHED, STRIDED, T>(
            handle, n - 1, n - 1, n - 1, A, shiftA + idx2D(1, 1, lda), lda, strideA, ipiv, strideP,
            batch_count, scalars, work, Abyx_tmptr, trfact, workArr);
    }

    return rocblas_status_success;
}
