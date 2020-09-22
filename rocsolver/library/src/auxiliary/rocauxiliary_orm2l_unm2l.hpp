/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORM2L_UNM2L_HPP
#define ROCLAPACK_ORM2L_UNM2L_HPP

#include "rocauxiliary_lacgv.hpp"
#include "rocauxiliary_larf.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orm2l_unm2l_getMemorySize(const rocblas_side side,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t *size_1, size_t *size_2,
                                         size_t *size_3, size_t *size_4) {
  *size_1 = *size_2 = *size_3 = *size_4 = 0;
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_orm2l_ormql_argCheck(
    const rocblas_side side, const rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int k, const rocblas_int lda,
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
  bool left = (side == rocblas_side_left);

  // 2. invalid size
  if (m < 0 || n < 0 || k < 0 || ldc < m)
    return rocblas_status_invalid_size;
  if (left && (lda < m || k > m))
    return rocblas_status_invalid_size;
  if (!left && (lda < n || k > n))
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((m * n && !C) || (k && !ipiv) || (left && m * k && !A) ||
      (!left && n * k && !A))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_orm2l_unm2l_template(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP, U C,
    const rocblas_int shiftC, const rocblas_int ldc,
    const rocblas_stride strideC, const rocblas_int batch_count, T *scalars,
    T *work, T **workArr, T *diag) {
  return rocblas_status_not_implemented;
}

#endif
