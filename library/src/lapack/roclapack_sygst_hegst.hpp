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
#include "roclapack_sygs2_hegs2.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_sygst_hegst_getMemorySize(const rocblas_int n,
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

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_sygst_hegst_template(rocblas_handle handle,
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
    ROCSOLVER_ENTER("sygst_hegst", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
