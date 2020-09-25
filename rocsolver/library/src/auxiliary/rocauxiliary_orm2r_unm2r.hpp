/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORM2R_UNM2R_HPP
#define ROCLAPACK_ORM2R_UNM2R_HPP

#include "rocauxiliary_lacgv.hpp"
#include "rocauxiliary_larf.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orm2r_unm2r_getMemorySize(const rocblas_side side,
                                         const rocblas_int m,
                                         const rocblas_int n,
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

  // memory requirements to call larf
  rocsolver_larf_getMemorySize<T, BATCHED>(side, m, n, batch_count, size_1,
                                           size_2, size_3);

  // size of temporary array for diagonal elemements
  *size_4 = sizeof(T) * batch_count;
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_orm2r_ormqr_argCheck(
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
  if (left && (k > m || lda < m))
    return rocblas_status_invalid_size;
  if (!left && (k > n || lda < n))
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((m * n && !C) || (k && !ipiv) || (left && m * k && !A) ||
      (!left && n * k && !A))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_orm2r_unm2r_template(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP, U C,
    const rocblas_int shiftC, const rocblas_int ldc,
    const rocblas_stride strideC, const rocblas_int batch_count, T *scalars,
    T *work, T **workArr, T *diag) {
  // quick return
  if (!n || !m || !k || !batch_count)
    return rocblas_status_success;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // determine limits and indices
  bool left = (side == rocblas_side_left);
  bool transpose = (trans != rocblas_operation_none);
  rocblas_int start, step, ncol, nrow, ic, jc;
  if (left) {
    ncol = n;
    jc = 0;
    if (transpose) {
      start = -1;
      step = 1;
    } else {
      start = k;
      step = -1;
    }
  } else {
    nrow = m;
    ic = 0;
    if (transpose) {
      start = k;
      step = -1;
    } else {
      start = -1;
      step = 1;
    }
  }

  // conjugate tau
  if (COMPLEX && transpose)
    rocsolver_lacgv_template<T>(handle, k, ipiv, 0, 1, strideP, batch_count);

  rocblas_int i;
  for (rocblas_int j = 1; j <= k; ++j) {
    i = start + step * j; // current householder vector
    if (left) {
      nrow = m - i;
      ic = i;
    } else {
      ncol = n - i;
      jc = i;
    }

    // insert one in A(i,i) tobuild/apply the householder matrix
    hipLaunchKernelGGL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                       stream, diag, 0, 1, A, shiftA + idx2D(i, i, lda), lda,
                       strideA, 1, true);

    // Apply current Householder reflector
    rocsolver_larf_template(handle, side, // side
                            nrow,         // number of rows of matrix to modify
                            ncol, // number of columns of matrix to modify
                            A,
                            shiftA + idx2D(i, i, lda), // householder vector x
                            1, strideA,                // inc of x
                            (ipiv + i), strideP, // householder scalar (alpha)
                            C, shiftC + idx2D(ic, jc, ldc), // matrix to work on
                            ldc, strideC,                   // leading dimension
                            batch_count, scalars, work, workArr);

    // restore original value of A(i,i)
    hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1),
                       0, stream, diag, 0, 1, A, shiftA + idx2D(i, i, lda), lda,
                       strideA, 1);
  }

  // restore tau
  if (COMPLEX && transpose)
    rocsolver_lacgv_template<T>(handle, k, ipiv, 0, 1, strideP, batch_count);

  return rocblas_status_success;
}

#endif
