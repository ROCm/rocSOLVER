/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_sygs2_hegs2_getMemorySize(const rocblas_int n,
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

template <typename T>
rocblas_status rocsolver_sygs2_hegs2_argCheck(rocblas_handle handle,
                                              const rocblas_eform itype,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              const rocblas_int ldb,
                                              T A,
                                              T B,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
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
    if((n && !A) || (n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_sygs2_hegs2_template(rocblas_handle handle,
                                              const rocblas_eform itype,
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
                                              const rocblas_int batch_count,
                                              T* scalars)
{
    ROCSOLVER_ENTER("sygs2_hegs2", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
