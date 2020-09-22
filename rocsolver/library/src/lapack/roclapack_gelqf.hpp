/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELQF_H
#define ROCLAPACK_GELQF_H

#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_gelq2.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_gelqf_getMemorySize(const rocblas_int m, const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2,
                                   size_t *size_3, size_t *size_4,
                                   size_t *size_5) {
  // if quick return no workspace needed
  if (m == 0 || n == 0 || batch_count == 0) {
    *size_1 = 0;
    *size_2 = 0;
    *size_3 = 0;
    *size_4 = 0;
    *size_5 = 0;
    return;
  }

  size_t s1, s2, s3, unused, s4 = 0;
  rocsolver_gelq2_getMemorySize<T, BATCHED>(m, n, batch_count, size_1, &s1,
                                            size_3, size_4);
  if (m <= GEQRF_GEQR2_SWITCHSIZE || n <= GEQRF_GEQR2_SWITCHSIZE) {
    *size_2 = s1;
    *size_5 = 0;
  } else {
    rocblas_int jb = GEQRF_GEQR2_BLOCKSIZE;
    rocsolver_larft_getMemorySize<T, BATCHED>(jb, batch_count, &unused, &s2,
                                              &unused);
    rocsolver_larfb_getMemorySize<T, BATCHED>(rocblas_side_right, m - jb, n, jb,
                                              batch_count, &s4, &s3, &unused);
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
rocsolver_gelqf_template(rocblas_handle handle, const rocblas_int m,
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
  if (m <= GEQRF_GEQR2_SWITCHSIZE || n <= GEQRF_GEQR2_SWITCHSIZE)
    return rocsolver_gelq2_template<T>(handle, m, n, A, shiftA, lda, strideA,
                                       ipiv, strideP, batch_count, scalars,
                                       work, workArr, diag);

  rocblas_int dim = min(m, n); // total number of pivots
  rocblas_int jb, j = 0;

  rocblas_int ldw = GEQRF_GEQR2_BLOCKSIZE;
  rocblas_stride strideW = rocblas_stride(ldw) * ldw;

  while (j < dim - GEQRF_GEQR2_SWITCHSIZE) {
    // Factor diagonal and subdiagonal blocks
    jb = min(dim - j, GEQRF_GEQR2_BLOCKSIZE); // number of rows in the block
    rocsolver_gelq2_template<T>(handle, jb, n - j, A, shiftA + idx2D(j, j, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count,
                                scalars, work, workArr, diag);

    // apply transformation to the rest of the matrix
    if (j + jb < m) {

      // compute block reflector
      rocsolver_larft_template<T>(
          handle, rocblas_forward_direction, rocblas_row_wise, n - j, jb, A,
          shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, trfact,
          ldw, strideW, batch_count, scalars, work, workArr);

      // apply the block reflector
      rocsolver_larfb_template<BATCHED, STRIDED, T>(
          handle, rocblas_side_right, rocblas_operation_none,
          rocblas_forward_direction, rocblas_row_wise, m - j - jb, n - j, jb, A,
          shiftA + idx2D(j, j, lda), lda, strideA, trfact, 0, ldw, strideW, A,
          shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count, diag, work,
          workArr);
    }
    j += GEQRF_GEQR2_BLOCKSIZE;
  }

  // factor last block
  if (j < dim)
    rocsolver_gelq2_template<T>(
        handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
        (ipiv + j), strideP, batch_count, scalars, work, workArr, diag);

  return rocblas_status_success;
}

#endif /* ROCLAPACK_GELQF_H */
