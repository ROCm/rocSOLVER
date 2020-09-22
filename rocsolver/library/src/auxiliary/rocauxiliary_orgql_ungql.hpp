/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGQL_UNGQL_HPP
#define ROCLAPACK_ORGQL_UNGQL_HPP

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_org2l_ung2l.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orgql_ungql_getMemorySize(
    const rocblas_int m, const rocblas_int n, const rocblas_int k,
    const rocblas_int batch_count, size_t *size_1, size_t *size_2,
    size_t *size_3, size_t *size_4, size_t *size_5) {
  size_t s1, s2, s3, unused;
  rocsolver_org2l_ung2l_getMemorySize<T, BATCHED>(m, n, batch_count, size_1,
                                                  size_2, size_3);

  if (k <= ORGxx_UNGxx_SWITCHSIZE) {
    *size_4 = 0;
    *size_5 = 0;
  } else {
    // size of workspace
    // maximum of what is needed by org2l, larft and larfb
    rocblas_int jb = ORGxx_UNGxx_BLOCKSIZE;
    rocblas_int j = ((k - ORGxx_UNGxx_SWITCHSIZE - 1) / jb) * jb;
    rocblas_int kk = min(k, j + jb);
    rocsolver_org2l_ung2l_getMemorySize<T>(max(m - kk, jb), n, batch_count,
                                           &s1);
    rocsolver_larft_getMemorySize<T>(jb, batch_count, &s2);
    rocsolver_larfb_getMemorySize<T, BATCHED>(
        rocblas_side_left, m - jb, n, jb, batch_count, &s3, &unused, size_5);

    *size_2 = max(max(s1, s2), s3);

    // size of temporary array for triangular factor
    *size_4 = sizeof(T) * jb * jb * batch_count;
  }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgql_ungql_template(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP,
    const rocblas_int batch_count, T *scalars, T *work, T **workArr, T *trfact,
    T *workTrmm) {
  // quick return
  if (!n || !m || !batch_count)
    return rocblas_status_success;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // if the matrix is small, use the unblocked variant of the algorithm
  if (k <= ORGxx_UNGxx_SWITCHSIZE)
    return rocsolver_org2l_ung2l_template<T>(
        handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count,
        scalars, work, workArr);

  rocblas_int ldw = ORGxx_UNGxx_BLOCKSIZE;
  rocblas_stride strideW = rocblas_stride(ldw) * ldw;

  // start of the unblocked block
  rocblas_int jb = ORGxx_UNGxx_BLOCKSIZE;
  rocblas_int kk = min(k, ((k - ORGxx_UNGxx_SWITCHSIZE + jb - 1) / jb) * jb);

  // start of first blocked block
  rocblas_int j = k - kk;

  rocblas_int blocksy, blocksx;

  // compute the unblocked part and set to zero the
  // corresponding left submatrix
  if (kk < m) {
    blocksx = (kk - 1) / 32 + 1;
    blocksy = (n - kk - 1) / 32 + 1;
    hipLaunchKernelGGL(set_zero<T>, dim3(blocksx, blocksy, batch_count),
                       dim3(32, 32), 0, stream, kk, n - kk, A,
                       shiftA + idx2D(m - kk, 0, lda), lda, strideA);

    rocsolver_org2l_ung2l_template<T>(handle, m - kk, n - kk, k - kk, A, shiftA,
                                      lda, strideA, ipiv, strideP, batch_count,
                                      scalars, work, workArr);
  }

  // compute the blocked part
  while (j < k) {
    // first update the already computed part
    // applying the current block reflector using larft + larfb
    if (n - k + j > 0) {
      rocsolver_larft_template<T>(handle, rocblas_backward_direction,
                                  rocblas_column_wise, m - k + j + jb, jb, A,
                                  shiftA + idx2D(0, n - k + j, lda), lda,
                                  strideA, (ipiv + j), strideP, trfact, ldw,
                                  strideW, batch_count, scalars, work, workArr);

      rocsolver_larfb_template<BATCHED, STRIDED, T>(
          handle, rocblas_side_left, rocblas_operation_none,
          rocblas_backward_direction, rocblas_column_wise, m - k + j + jb,
          n - k + j, jb, A, shiftA + idx2D(0, n - k + j, lda), lda, strideA,
          trfact, 0, ldw, strideW, A, shiftA, lda, strideA, batch_count, work,
          workArr, workTrmm);
    }

    // now compute the current block and set to zero
    // the corresponding bottom submatrix
    if (j > 0) {
      blocksx = (k - j - jb - 1) / 32 + 1;
      blocksy = (jb - 1) / 32 + 1;
      hipLaunchKernelGGL(set_zero<T>, dim3(blocksx, blocksy, batch_count),
                         dim3(32, 32), 0, stream, k - j - jb, jb, A,
                         shiftA + idx2D(m - k + j + jb, n - k + j, lda), lda,
                         strideA);
    }
    rocsolver_org2l_ung2l_template<T>(
        handle, m - k + j + jb, jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
        lda, strideA, (ipiv + j), strideP, batch_count, scalars, work, workArr);

    j += jb;
  }

  return rocblas_status_success;
}

#endif
