/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELQ2_H
#define ROCLAPACK_GELQ2_H

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_gelq2_getMemorySize(const rocblas_int m, const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2,
                                   size_t *size_3, size_t *size_4) {
  size_t s1, s2;
  rocsolver_larf_getMemorySize<T, BATCHED>(rocblas_side_right, m, n,
                                           batch_count, size_1, &s1, size_3);
  rocsolver_larfg_getMemorySize<T>(n, batch_count, size_4, &s2);
  *size_2 = max(s1, s2);
}

template <typename T, typename U>
rocblas_status
rocsolver_gelq2_gelqf_argCheck(const rocblas_int m, const rocblas_int n,
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
rocblas_status rocsolver_gelq2_template(
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
  rocblas_int blocks = (n - 1) / 1024 + 1;

  for (rocblas_int j = 0; j < dim; ++j) {
    // conjugate the jth row of A
    if (COMPLEX)
      rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda),
                                  lda, strideA, batch_count);

    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle,
                             // order of reflector
                             n - j,
                             // value of alpha
                             A, shiftA + idx2D(j, j, lda),
                             // vector x to work on
                             A, shiftA + idx2D(j, min(j + 1, n - 1), lda),
                             // inc of x
                             lda, strideA,
                             // tau
                             (ipiv + j), strideP, batch_count, diag, work);

    // insert one in A(j,j) tobuild/apply the householder matrix
    hipLaunchKernelGGL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                       stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda,
                       strideA, 1, true);

    // Apply Householder reflector to the rest of matrix from the right
    if (j < m - 1) {
      rocsolver_larf_template(handle,
                              // side
                              rocblas_side_right,
                              // number of rows of matrix to modify
                              m - j - 1,
                              // number of columns of matrix to modify
                              n - j,
                              // householder vector x
                              A, shiftA + idx2D(j, j, lda),
                              // inc of x
                              lda, strideA,
                              // householder scalar (alpha)
                              (ipiv + j), strideP,
                              // matrix to work on
                              A, shiftA + idx2D(j + 1, j, lda),
                              // leading dimension
                              lda, strideA, batch_count, scalars, work,
                              workArr);
    }

    // restore original value of A(j,j)
    hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1),
                       0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda,
                       strideA, 1);

    // restore the jth row of A
    if (COMPLEX)
      rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda),
                                  lda, strideA, batch_count);
  }

  return rocblas_status_success;
}

#endif /* ROCLAPACK_GELQ2_H */
