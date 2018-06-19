/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_GETRS_HPP
#define ROCLAPACK_GETRS_HPP

#include "rocsolver-export.h"
#include <hip/hip_runtime.h>
#include <rocblas.hpp>

#include "ideal_sizes.hpp"
#include "roclapack_laswp.hpp"

#define GETRS_INPONE 0

template <typename T>
rocblas_status
rocsolver_getrs_template(rocblas_handle handle, rocblas_operation trans,
                         rocblas_int n, rocblas_int nrhs, const T *A,
                         rocblas_int lda, const rocblas_int *ipiv, T *B,
                         rocblas_int ldb) {

  // TODO remove const_cast here once rocBLAS is released with the correct API

  // check for possible input problems
  if (n < 0 || nrhs < 0 || lda < max(1, n) || ldb < max(1, n)) {
    cout << "Invalid size " << n << " " << nrhs << " " << lda << " " << ldb
         << endl;
    return rocblas_status_invalid_size;
  }

  // quick return
  if (n == 0 || nrhs == 0) {
    return rocblas_status_success;
  }

  T inpsResHost[2];
  inpsResHost[GETRS_INPONE] = static_cast<T>(1);

  // allocate a tiny bit of memory on device to avoid going onto CPU and needing
  // to synchronize.
  T *inpsResGPU;
  hipMalloc(&inpsResGPU, sizeof(T));
  hipMemcpy(inpsResGPU, &inpsResHost[0], sizeof(T), hipMemcpyHostToDevice);

  if (trans == rocblas_operation_none) {

    // solve A * X = B
    // first apply row interchanges to the right hand sides
    roclapack_laswp_template<T>(handle, nrhs, B, ldb, 1, n, ipiv, 1);

    // solve L*X - B, overwriting B with X
    rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_lower,
                    rocblas_operation_none, rocblas_diagonal_unit, n, nrhs,
                    &inpsResGPU[GETRS_INPONE], const_cast<T *>(A), lda, B, ldb);

    // solve U*X = B, overwriting B with X
    rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_upper,
                    rocblas_operation_none, rocblas_diagonal_non_unit, n, nrhs,
                    &inpsResGPU[GETRS_INPONE], const_cast<T *>(A), lda, B, ldb);
  } else {

    // solve A**T * X = B  or A**H * X = B
    // solve U**T *X = B or U**H *X = B, overwriting B with X
    rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_upper, trans,
                    rocblas_diagonal_non_unit, n, nrhs,
                    &inpsResGPU[GETRS_INPONE], const_cast<T *>(A), lda, B, ldb);

    // solve L**T *X = B, or L**H *X = B overwriting B with X
    rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_lower, trans,
                    rocblas_diagonal_unit, n, nrhs, &inpsResGPU[GETRS_INPONE],
                    const_cast<T *>(A), lda, B, ldb);

    // apply row interchanges to the solution vectors
    roclapack_laswp_template<T>(handle, nrhs, B, ldb, 1, n, ipiv, -1);
  }

  hipFree(inpsResGPU);

  return rocblas_status_success;
}

#undef GETRS_INPONE

#endif /* ROCLAPACK_GETRS_HPP */
