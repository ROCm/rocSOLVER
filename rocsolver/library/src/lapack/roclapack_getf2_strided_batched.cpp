/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getf2_strided_batched_impl(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, U A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_int *info,
    const rocblas_int batch_count, const int pivot) {

  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_getf2_getrf_argCheck(m, n, lda, A, ipiv, info, batch_count);
  if (st != rocblas_status_continue)
    return st;

  // memory managment
  using S = decltype(std::real(T{}));
  size_t size_1; // size of constants
  size_t size_2; // pivot values
  size_t size_3; // pivot indices
  size_t size_4; // workspace
  rocsolver_getf2_getMemorySize<T, S>(m, batch_count, &size_1, &size_2, &size_3,
                                      &size_4);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *pivot_idx, *pivot_val, *work;
  hipMalloc(&scalars, size_1);
  hipMalloc(&pivot_val, size_2);
  hipMalloc(&pivot_idx, size_3);
  hipMalloc(&work, size_4);
  if (!scalars || (size_2 && !pivot_idx) || (size_3 && !pivot_val) ||
      (size_4 && !work))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_getf2_template<true, T, S>(
      handle, m, n, A,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      lda, strideA, ipiv,
      0, // the vector is shifted 0 entries (will work on the entire vector)
      strideP, info, batch_count, pivot, (T *)scalars, (T *)pivot_val,
      (rocblas_int *)pivot_idx, (rocblas_index_value_t<S> *)work);

  hipFree(scalars);
  hipFree(pivot_val);
  hipFree(pivot_idx);
  hipFree(work);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetf2_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getf2_strided_batched_impl<float>(
      handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count, 1);
}

rocblas_status rocsolver_dgetf2_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getf2_strided_batched_impl<double>(
      handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count, 1);
}

rocblas_status rocsolver_cgetf2_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getf2_strided_batched_impl<rocblas_float_complex>(
      handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count, 1);
}

rocblas_status rocsolver_zgetf2_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getf2_strided_batched_impl<rocblas_double_complex>(
      handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count, 1);
}

rocblas_status rocsolver_sgetf2_npvt_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  rocblas_int *ipiv;
  return rocsolver_getf2_strided_batched_impl<float>(
      handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count, 0);
}

rocblas_status rocsolver_dgetf2_npvt_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  rocblas_int *ipiv;
  return rocsolver_getf2_strided_batched_impl<double>(
      handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count, 0);
}

rocblas_status rocsolver_cgetf2_npvt_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  rocblas_int *ipiv;
  return rocsolver_getf2_strided_batched_impl<rocblas_float_complex>(
      handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count, 0);
}

rocblas_status rocsolver_zgetf2_npvt_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  rocblas_int *ipiv;
  return rocsolver_getf2_strided_batched_impl<rocblas_double_complex>(
      handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count, 0);
}

} // extern C
