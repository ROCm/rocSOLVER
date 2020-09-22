/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQL2_H
#define ROCLAPACK_GEQL2_H

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_geql2_getMemorySize(const rocblas_int m, const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2,
                                   size_t *size_3, size_t *size_4) {
  // if quick return no workspace needed
  if (m == 0 || n == 0 || batch_count == 0) {
    *size_1 = 0;
    *size_2 = 0;
    *size_3 = 0;
    *size_4 = 0;
    return;
  }

  size_t s1, s2;
  rocsolver_larf_getMemorySize<T, BATCHED>(rocblas_side_left, m, n, batch_count,
                                           size_1, &s1, size_3);
  rocsolver_larfg_getMemorySize<T>(n, batch_count, &s2, size_4);
  *size_2 = max(s1, s2);
}

template <typename T, typename U>
rocblas_status
rocsolver_geql2_geqlf_argCheck(const rocblas_int m, const rocblas_int n,
                               const rocblas_int lda, T A, U ipiv,
                               const rocblas_int batch_count = 1) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  // N/A

  // 2. invalid size
  if (m < 0 || n < 0 || lda < m || batch_count < 0)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((m * n && !A) || (m * n && !ipiv))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_geql2_template(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, U A,
    const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP,
    const rocblas_int batch_count, T *scalars, T *work, T **workArr, T *diag) {
  // quick return
  if (m == 0 || n == 0 || batch_count == 0)
    return rocblas_status_success;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  rocblas_int dim = min(m, n); // total number of pivots

  for (rocblas_int j = 0; j < dim; j++) {
    // generate Householder reflector to work on column j
    rocsolver_larfg_template(
        handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
        shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
        strideP, batch_count, work, diag);

    // insert one in A(m-j-1,n-j-1) tobuild/apply the householder matrix
    hipLaunchKernelGGL(
        set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, diag, 0,
        1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1, true);

    // conjugate tau
    if (COMPLEX)
      rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP,
                                  batch_count);

    // Apply Householder reflector to the rest of matrix from the left
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                            shiftA + idx2D(0, n - j - 1, lda), 1, strideA,
                            (ipiv + dim - j - 1), strideP, A, shiftA, lda,
                            strideA, batch_count, scalars, work, workArr);

    // restore original value of A(m-j-1,n-j-1)
    hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1),
                       0, stream, diag, 0, 1, A,
                       shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA,
                       1);

    // restore tau
    if (COMPLEX)
      rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP,
                                  batch_count);
  }

  return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQL2_H */
