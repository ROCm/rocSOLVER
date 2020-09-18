/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGTR_UNGTR_HPP
#define ROCLAPACK_ORGTR_UNGTR_HPP

#include "rocauxiliary_orgql_ungql.hpp"
#include "rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orgtr_ungtr_getMemorySize(const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t *size_1, size_t *size_2,
                                         size_t *size_3, size_t *size_4,
                                         size_t *size_5) {
  *size_1 = *size_2 = *size_3 = *size_4 = *size_5 = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_orgtr_argCheck(const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int lda, T A, U ipiv) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    return rocblas_status_invalid_value;

  // 2. invalid size
  if (n < 0 || lda < n)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((n && !A) || (n > 1 && !ipiv))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgtr_ungtr_template(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, U A,
    const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP,
    const rocblas_int batch_count, T *scalars, T *work, T **workArr, T *trfact,
    T *workTrmm) {
  return rocblas_status_not_implemented;
}

#endif
