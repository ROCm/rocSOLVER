/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_ormlq_unmlq.hpp"
#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_gels.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver.h"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_gels_outofplace_getMemorySize(const rocblas_int m,
                                             const rocblas_int n,
                                             const rocblas_int nrhs,
                                             const rocblas_int batch_count,
                                             size_t* size_scalars,
                                             size_t* size_work_x_temp,
                                             size_t* size_workArr_temp_arr,
                                             size_t* size_diag_trfac_invA,
                                             size_t* size_trfact_workTrmm_invA_arr,
                                             size_t* size_ipiv,
                                             size_t* size_savedB,
                                             bool* optim_mem)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_diag_trfac_invA = 0;
        *size_trfact_workTrmm_invA_arr = 0;
        *size_ipiv = 0;
        *size_savedB = 0;
        *optim_mem = true;
        return;
    }

    size_t unused;

    rocsolver_gels_getMemorySize<BATCHED, STRIDED, T>(
        m, n, nrhs, batch_count, size_scalars, size_work_x_temp, size_workArr_temp_arr,
        size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, &unused, optim_mem);

    *size_ipiv = sizeof(T) * std::min(m, n) * batch_count;
    *size_savedB = sizeof(T) * std::max(m, n) * nrhs * batch_count;
}

template <bool COMPLEX, typename T>
rocblas_status rocsolver_gels_outofplace_argCheck(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  T A,
                                                  const rocblas_int lda,
                                                  T B,
                                                  const rocblas_int ldb,
                                                  T X,
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if((COMPLEX && trans == rocblas_operation_transpose)
       || (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(trans == rocblas_operation_none && (ldb < m || ldx < n))
        return rocblas_status_invalid_size;
    if(trans != rocblas_operation_none && (ldb < n || ldx < m))
        return rocblas_status_invalid_size;
    if(m < 0 || n < 0 || nrhs < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    rocblas_int rowsB = (trans == rocblas_operation_none ? m : n);
    rocblas_int rowsX = (trans == rocblas_operation_none ? n : m);
    if((m * n && !A) || (rowsB * nrhs && !B) || (rowsX * nrhs && !X) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gels_outofplace_template(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  U A,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  U B,
                                                  const rocblas_int shiftB,
                                                  const rocblas_int ldb,
                                                  const rocblas_stride strideB,
                                                  U X,
                                                  const rocblas_int shiftX,
                                                  const rocblas_int ldx,
                                                  const rocblas_stride strideX,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count,
                                                  T* scalars,
                                                  T* work_x_temp,
                                                  T* workArr_temp_arr,
                                                  T* diag_trfac_invA,
                                                  T** trfact_workTrmm_invA_arr,
                                                  T* ipiv,
                                                  T* savedB,
                                                  bool optim_mem)
{
    ROCSOLVER_ENTER("gels_outofplace", "trans:", trans, "m:", m, "n:", n, "nrhs:", nrhs,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "shiftX:", shiftX, "ldx:", ldx, "bc:", batch_count);

    return rocblas_status_not_implemented;
}
