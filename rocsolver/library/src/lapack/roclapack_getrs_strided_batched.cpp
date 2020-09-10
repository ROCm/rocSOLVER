/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrs_strided_batched_impl(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, U A, const rocblas_int lda,
    const rocblas_stride strideA, const rocblas_int *ipiv,
    const rocblas_stride strideP, U B, const rocblas_int ldb,
    const rocblas_stride strideB, const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_getrs_argCheck(trans, n, nrhs, lda, ldb, A, B,
                                               ipiv, batch_count);
  if (st != rocblas_status_continue)
    return st;

  // memory managment
  // this function does not requiere memory work space
  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE

  // execution
  return rocsolver_getrs_template<T>(handle, trans, n, nrhs, A, 0, lda, strideA,
                                     ipiv, strideP, B, 0, ldb, strideB,
                                     batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrs_strided_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, float *A, const rocblas_int lda,
    const rocblas_stride strideA, const rocblas_int *ipiv,
    const rocblas_stride strideP, float *B, const rocblas_int ldb,
    const rocblas_stride strideB, const rocblas_int batch_count) {
  return rocsolver_getrs_strided_batched_impl<float>(
      handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB,
      batch_count);
}

extern "C" ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrs_strided_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, double *A, const rocblas_int lda,
    const rocblas_stride strideA, const rocblas_int *ipiv,
    const rocblas_stride strideP, double *B, const rocblas_int ldb,
    const rocblas_stride strideB, const rocblas_int batch_count) {
  return rocsolver_getrs_strided_batched_impl<double>(
      handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB,
      batch_count);
}

extern "C" ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrs_strided_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, const rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_float_complex *B,
    const rocblas_int ldb, const rocblas_stride strideB,
    const rocblas_int batch_count) {
  return rocsolver_getrs_strided_batched_impl<rocblas_float_complex>(
      handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB,
      batch_count);
}

extern "C" ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrs_strided_batched(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, const rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_double_complex *B,
    const rocblas_int ldb, const rocblas_stride strideB,
    const rocblas_int batch_count) {
  return rocsolver_getrs_strided_batched_impl<rocblas_double_complex>(
      handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB,
      batch_count);
}
