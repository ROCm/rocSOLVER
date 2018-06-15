/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_GETRF_HPP
#define ROCLAPACK_GETRF_HPP

#include <hip/hip_runtime.h>
#include <rocblas.hpp>

#include "ideal_sizes.hpp"
#include "roclapack_getf2.hpp"
#include "roclapack_laswp.hpp"

#define GETRF_INPMINONE 0
#define GETRF_INPONE 1

__global__ void getrf_indices(rocblas_int n, rocblas_int j, rocblas_int *ipiv) {
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tid < n) {
    ipiv[j + tid] += j;
  }
}

template <typename T>
rocblas_status rocsolver_getrf_template(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, T *A, rocblas_int lda,
                                        rocblas_int *ipiv) {

  // if the matrix is small, use the unblocked variant
  if (m < GETRF_GETF2_SWITCHSIZE || n < GETRF_GETF2_SWITCHSIZE) {
    return rocsolver_getf2_template<T>(handle, m, n, A, lda, ipiv);
  }

  if (m == 0 || n == 0) {
    // quick return
    return rocblas_status_success;
  } else if (m < 0) {
    // less than zero dimensions in a matrix?!
    return rocblas_status_invalid_size;
  } else if (n < 0) {
    // less than zero dimensions in a matrix?!
    return rocblas_status_invalid_size;
  } else if (lda < max(1, m)) {
    // mismatch of provided first matrix dimension
    return rocblas_status_invalid_size;
  }

  T inpsResHost[2];
  inpsResHost[GETRF_INPMINONE] = static_cast<T>(-1);
  inpsResHost[GETRF_INPONE] = static_cast<T>(1);

  // allocate a tiny bit of memory on device to avoid going onto CPU and needing
  // to synchronize.
  T *inpsResGPU;
  hipMalloc(&inpsResGPU, 2 * sizeof(T));
  hipMemcpy(inpsResGPU, &inpsResHost[0], 2 * sizeof(T), hipMemcpyHostToDevice);

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  for (rocblas_int j = 0; j < min(m, n); j += GETRF_GETF2_SWITCHSIZE) {

    const rocblas_int jb = min(min(m, n) - j, GETRF_GETF2_SWITCHSIZE);

    // Factor diagonal and subdiagonal blocks and test for exact singularity
    const rocblas_status substat = rocsolver_getf2_template<T>(
        handle, m - j, jb, &A[idx2D(j, j, lda)], lda, &ipiv[j]);
    if (substat != rocblas_status_success) {
      return substat;
    }

    // adjust pivot indices
    rocblas_int sizePivot = min(m, j + jb);
    rocblas_int blocksPivot = (sizePivot - 1) / 256 + 1;
    dim3 gridPivot(blocksPivot, 1, 1);
    dim3 threads(256, 1, 1);
    hipLaunchKernelGGL(getrf_indices, gridPivot, threads, 0, stream, sizePivot,
                       j, ipiv);

    // apply interchanges to columns 1 : j-1
    roclapack_laswp_template<T>(handle, j, A, lda, j, j + jb, ipiv, 1);

    if (j + jb < n) {
      // apply interchanges to columns j+jb : n
      roclapack_laswp_template<T>(handle, (n - j - jb),
                                  &A[idx2D(1, j + jb - 1, lda)], lda, j, j + jb,
                                  ipiv, 1);

      if (j >= 1)
        return rocblas_status_success;

      // compute block row of U
      rocblas_trsm(
          handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
          rocblas_diagonal_unit, jb, (n - j - jb), &inpsResGPU[GETRF_INPONE],
          &A[idx2D(j, j, lda)], lda, &A[idx2D(j, j + jb - 1, lda)], lda);

      if (j + jb < m) {
        // update trailing submatrix
        rocblas_gemm(
            handle, rocblas_operation_none, rocblas_operation_none,
            (m - j - jb), (n - j - jb), jb, &inpsResGPU[GETRF_INPMINONE],
            &A[idx2D(j + jb - 1, j, lda)], lda, &A[idx2D(j, j + jb - 1, lda)],
            lda, &inpsResGPU[GETRF_INPONE],
            &A[idx2D(j + jb - 1, j + jb - 1, lda)], lda);
      }
    }
  }

  hipFree(inpsResGPU);

  return rocblas_status_success;
}

#undef GETRF_INPMINONE
#undef GETRF_INPONE

#endif /* ROCLAPACK_GETRF_HPP */
