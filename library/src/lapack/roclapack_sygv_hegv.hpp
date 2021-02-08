/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_sygv_hegv_getMemorySize(const rocblas_eform itype,
                                       const rocblas_evect jobz,
                                       const rocblas_int n,
                                       const rocblas_int batch_count,
                                       size_t* size_scalars)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;
}

template <typename S, typename T>
rocblas_status rocsolver_sygv_hegv_argCheck(rocblas_handle handle,
                                            const rocblas_eform itype,
                                            const rocblas_evect jobz,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            const rocblas_int lda,
                                            const rocblas_int ldb,
                                            T A,
                                            T B,
                                            S D,
                                            S E,
                                            rocblas_int* info,
                                            const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
        return rocblas_status_invalid_value;
    if(jobz != rocblas_evect_none && jobz != rocblas_evect_original)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !B) || (n && !D) || (n && !E) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_sygv_hegv_template(rocblas_handle handle,
                                            const rocblas_eform itype,
                                            const rocblas_evect jobz,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            U B,
                                            const rocblas_int shiftB,
                                            const rocblas_int ldb,
                                            const rocblas_stride strideB,
                                            S* D,
                                            const rocblas_stride strideD,
                                            S* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count,
                                            T* scalars)
{
    ROCSOLVER_ENTER("sygv_hegv", "itype:", itype, "jobz:", jobz, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "bc:", batch_count);

    return rocblas_status_not_implemented;
}
