/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver.h"

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevx_heevx_argCheck(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_erange erange,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              const S vl,
                                              const S vu,
                                              const rocblas_int il,
                                              const rocblas_int iu,
                                              rocblas_int* nev,
                                              S* W,
                                              T Z,
                                              const rocblas_int ldz,
                                              rocblas_int* ifail,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_original && evect != rocblas_evect_none)
        return rocblas_status_invalid_value;
    if(erange != rocblas_erange_all && erange != rocblas_erange_value
       && erange != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || (evect != rocblas_evect_none && ldz < n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !nev) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if(evect != rocblas_evect_none && ((n && !Z) || (n && !ifail)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevx_heevx_getMemorySize(const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_workArr)
{
    // to be completed
    *size_scalars = 0;
    *size_workArr = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevx_heevx_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_erange erange,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              const S vl,
                                              const S vu,
                                              const rocblas_int il,
                                              const rocblas_int iu,
                                              rocblas_int* nev,
                                              S* W,
                                              const rocblas_stride strideW,
                                              U Z,
                                              const rocblas_int shiftZ,
                                              const rocblas_int ldz,
                                              const rocblas_stride strideZ,
                                              rocblas_int* ifail,
                                              const rocblas_stride strideF,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T** workArr)
{
    ROCSOLVER_ENTER("syevx_heevx", "evect:", evect, "erange:", erange, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "vl:", vl, "vu:", vu, "il:", il, "iu:", iu,
                    "shiftZ:", shiftZ, "ldz:", ldz, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
