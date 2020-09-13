/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQLF_H
#define ROCLAPACK_GEQLF_H

#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_geql2.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_geqlf_getMemorySize(const rocblas_int m, const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2,
                                   size_t *size_3, size_t *size_4,
                                   size_t *size_5) {
  size_t s1, s2, s3, dum, s4 = 0;
  rocsolver_geql2_getMemorySize<T, BATCHED>(m, n, batch_count, size_1, &s1,
                                            size_3, size_4);
  if (m <= GEQLF_GEQL2_SWITCHSIZE || n <= GEQLF_GEQL2_SWITCHSIZE) {
    *size_2 = s1;
    *size_5 = 0;
  } else {
    rocblas_int jb = GEQLF_GEQL2_BLOCKSIZE;
    rocsolver_larft_getMemorySize<T>(jb, batch_count, &s2);
    rocsolver_larfb_getMemorySize<T, BATCHED>(rocblas_side_left, m, n - jb, jb,
                                              batch_count, &s3, &dum, &s4);
    *size_2 = max(s1, max(s2, s3));
    *size_5 = sizeof(T) * jb * jb * batch_count;
  }
  *size_4 = max(*size_4, s4);

  // size of workArr is double to accomodate
  // the TRMM calls in the batched case
  if (BATCHED)
    *size_3 *= 2;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status
rocsolver_geqlf_template(rocblas_handle handle, const rocblas_int m,
                         const rocblas_int n, U A, const rocblas_int shiftA,
                         const rocblas_int lda, const rocblas_stride strideA,
                         T *ipiv, const rocblas_stride strideP,
                         const rocblas_int batch_count, T *scalars, T *work,
                         T **workArr, T *diag, T *trfact) {
  // quick return
  if (m == 0 || n == 0 || batch_count == 0)
    return rocblas_status_success;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
  // algorithm
  if (m <= GEQLF_GEQL2_SWITCHSIZE || n <= GEQLF_GEQL2_SWITCHSIZE)
    return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA,
                                       ipiv, strideP, batch_count, scalars,
                                       work, workArr, diag);

  rocblas_int k = min(m, n); // total number of pivots
  rocblas_int nb = GEQLF_GEQL2_BLOCKSIZE;
  rocblas_int ki = ((k - GEQLF_GEQL2_SWITCHSIZE - 1) / nb) * nb;
  rocblas_int kk = min(k, ki + nb);
  rocblas_int jb, j = k - kk + ki;
  rocblas_int mu = m, nu = n;

  rocblas_int ldw = GEQLF_GEQL2_BLOCKSIZE;
  rocblas_stride strideW = rocblas_stride(ldw) * ldw;

  while (j >= k - kk) {
    // Factor diagonal and subdiagonal blocks
    jb = min(k - j, nb); // number of columns in the block
    rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A,
                                shiftA + idx2D(0, n - k + j, lda), lda, strideA,
                                (ipiv + j), strideP, batch_count, scalars, work,
                                workArr, diag);

    // apply transformation to the rest of the matrix
    if (n - k + j > 0) {

      // compute block reflector
      rocsolver_larft_template<T>(handle, rocblas_backward_direction,
                                  rocblas_column_wise, m - k + j + jb, jb, A,
                                  shiftA + idx2D(0, n - k + j, lda), lda,
                                  strideA, (ipiv + j), strideP, trfact, ldw,
                                  strideW, batch_count, scalars, work, workArr);

      // apply the block reflector
      rocsolver_larfb_template<BATCHED, STRIDED, T>(
          handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
          rocblas_backward_direction, rocblas_column_wise, m - k + j + jb,
          n - k + j, jb, A, shiftA + idx2D(0, n - k + j, lda), lda, strideA,
          trfact, 0, ldw, strideW, A, shiftA, lda, strideA, batch_count, work,
          workArr, diag);
    }
    j -= nb;
    mu = m - k + j + jb;
    nu = n - k + j + jb;
  }

  // factor last block
  if (mu > 0 && nu > 0)
    rocsolver_geql2_template<T>(handle, mu, nu, A, shiftA, lda, strideA, ipiv,
                                strideP, batch_count, scalars, work, workArr,
                                diag);

  return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQLF_H */
