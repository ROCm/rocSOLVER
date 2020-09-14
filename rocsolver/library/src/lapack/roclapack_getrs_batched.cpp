/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrs_batched_impl(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, U A, const rocblas_int lda, const rocblas_int *ipiv,
    const rocblas_stride strideP, U B, const rocblas_int ldb,
    const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_getrs_argCheck(trans, n, nrhs, lda, ldb, A, B,
                                               ipiv, batch_count);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;
  rocblas_stride strideB = 0;

  // memory managment
  size_t size_1; // for TRSM x_temp
  size_t size_2; // for TRSM x_temp_arr
  size_t size_3; // for TRSM invA
  size_t size_4; // for TRSM invA_arr
  rocsolver_getrs_getMemorySize<true, T>(n, nrhs, batch_count, &size_1, &size_2,
                                         &size_3, &size_4);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *x_temp, *x_temp_arr, *invA, *invA_arr;
  // always allocate all required memory for TRSM optimal performance
  bool optim_mem = true;

  hipMalloc(&x_temp, size_1);
  hipMalloc(&x_temp_arr, size_2);
  hipMalloc(&invA, size_3);
  hipMalloc(&invA_arr, size_4);
  if ((size_1 && !x_temp) || (size_2 && !x_temp_arr) || (size_3 && !invA) ||
      (size_4 && !invA_arr))
    return rocblas_status_memory_error;

  // execution
  rocblas_status status = rocsolver_getrs_template<true, T>(
      handle, trans, n, nrhs, A, 0, lda, strideA, ipiv, strideP, B, 0, ldb,
      strideB, batch_count, x_temp, x_temp_arr, invA, invA_arr, optim_mem);

  hipFree(x_temp);
  hipFree(x_temp_arr);
  hipFree(invA);
  hipFree(invA_arr);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_sgetrs_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, float *const A[], const rocblas_int lda,
    const rocblas_int *ipiv, const rocblas_stride strideP, float *const B[],
    const rocblas_int ldb, const rocblas_int batch_count) {
  return rocsolver_getrs_batched_impl<float>(
      handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

extern "C" rocblas_status rocsolver_dgetrs_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, double *const A[], const rocblas_int lda,
    const rocblas_int *ipiv, const rocblas_stride strideP, double *const B[],
    const rocblas_int ldb, const rocblas_int batch_count) {
  return rocsolver_getrs_batched_impl<double>(
      handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

extern "C" rocblas_status rocsolver_cgetrs_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_float_complex *const A[],
    const rocblas_int lda, const rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_float_complex *const B[],
    const rocblas_int ldb, const rocblas_int batch_count) {
  return rocsolver_getrs_batched_impl<rocblas_float_complex>(
      handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

extern "C" rocblas_status rocsolver_zgetrs_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_double_complex *const A[],
    const rocblas_int lda, const rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_double_complex *const B[],
    const rocblas_int ldb, const rocblas_int batch_count) {
  return rocsolver_getrs_batched_impl<rocblas_double_complex>(
      handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}
