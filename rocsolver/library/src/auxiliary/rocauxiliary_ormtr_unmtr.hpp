/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMTR_UNMTR_HPP
#define ROCLAPACK_ORMTR_UNMTR_HPP

#include "rocauxiliary_ormql_unmql.hpp"
#include "rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_ormtr_unmtr_getMemorySize(
    const rocblas_side side, const rocblas_fill uplo, const rocblas_int m,
    const rocblas_int n, const rocblas_int batch_count, size_t *size_1,
    size_t *size_2, size_t *size_3, size_t *size_4, size_t *size_5) {
  *size_1 = *size_2 = *size_3 = *size_4 = *size_5 = 0;
}

template <bool COMPLEX, typename T, typename U>
rocblas_status
rocsolver_ormtr_argCheck(const rocblas_side side, const rocblas_fill uplo,
                         const rocblas_operation trans, const rocblas_int m,
                         const rocblas_int n, const rocblas_int lda,
                         const rocblas_int ldc, T A, T C, U ipiv) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  if (side != rocblas_side_left && side != rocblas_side_right)
    return rocblas_status_invalid_value;
  if (trans != rocblas_operation_none && trans != rocblas_operation_transpose &&
      trans != rocblas_operation_conjugate_transpose)
    return rocblas_status_invalid_value;
  if ((COMPLEX && trans == rocblas_operation_transpose) ||
      (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
    return rocblas_status_invalid_value;
  if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    return rocblas_status_invalid_value;
  bool left = (side == rocblas_side_left);

  // 2. invalid size
  rocblas_int nq = left ? m : n;
  if (m < 0 || n < 0 || ldc < m || lda < nq)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((nq > 0 && !A) || (nq > 0 && !ipiv) || (m * n && !C))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U,
          bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_template(
    rocblas_handle handle, const rocblas_side side, const rocblas_fill uplo,
    const rocblas_operation trans, const rocblas_int m, const rocblas_int n,
    U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP, U C,
    const rocblas_int shiftC, const rocblas_int ldc,
    const rocblas_stride strideC, const rocblas_int batch_count, T *scalars,
    T *work, T **workArr, T *trfact, T *workTrmm) {
  return rocblas_status_not_implemented;
}

#endif
