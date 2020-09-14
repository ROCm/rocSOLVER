/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORG2L_UNG2L_HPP
#define ROCLAPACK_ORG2L_UNG2L_HPP

#include "rocauxiliary_lacgv.hpp"
#include "rocauxiliary_larf.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_org2l_ung2l_getMemorySize(const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t *size_1, size_t *size_2,
                                         size_t *size_3) {
  *size_1 = *size_2 = *size_3 = 0;
}

template <typename T>
void rocsolver_org2l_ung2l_getMemorySize(const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t *size) {
  *size = 0;
}

template <typename T, typename U>
rocblas_status
rocsolver_org2l_orgql_argCheck(const rocblas_int m, const rocblas_int n,
                               const rocblas_int k, const rocblas_int lda, T A,
                               U ipiv) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  // N/A

  // 2. invalid size
  if (m < 0 || n < 0 || m < n || k < 0 || k > n || lda < m)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((k && !ipiv) || (m * n && !A))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_org2l_ung2l_template(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP,
    const rocblas_int batch_count, T *scalars, T *work, T **workArr) {
  return rocblas_status_not_implemented;
}

#endif
