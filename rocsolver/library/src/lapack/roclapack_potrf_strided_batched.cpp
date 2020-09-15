/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potrf.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_potrf_strided_batched_impl(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, U A,
    const rocblas_int lda, const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_potf2_potrf_argCheck(uplo, n, lda, A, info, batch_count);
  if (st != rocblas_status_continue)
    return st;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3;
  size_t size_4;
  size_t size_5; // for TRSM
  size_t size_6; // for TRSM
  size_t size_7; // for TRSM
  size_t size_8; // for TRSM
  rocsolver_potrf_getMemorySize<false, T>(n, uplo, batch_count, &size_1,
                                          &size_2, &size_3, &size_4, &size_5,
                                          &size_6, &size_7, &size_8);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *pivotGPU, *iinfo, *x_temp, *x_temp_arr, *invA,
      *invA_arr;
  // always allocate all required memory for TRSM optimal performance
  bool optim_mem = true;

  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&pivotGPU, size_3);
  hipMalloc(&iinfo, size_4);
  hipMalloc(&x_temp, size_5);
  hipMalloc(&x_temp_arr, size_6);
  hipMalloc(&invA, size_7);
  hipMalloc(&invA_arr, size_8);
  if (!scalars || (size_2 && !work) || (size_3 && !pivotGPU) ||
      (size_4 && !iinfo) || (size_5 && !x_temp) || (size_6 && !x_temp_arr) ||
      (size_7 && !invA) || (size_8 && !invA_arr))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_potrf_template<false, S, T>(
      handle, uplo, n, A,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      lda, strideA, info, batch_count, (T *)scalars, (T *)work, (T *)pivotGPU,
      (rocblas_int *)iinfo, x_temp, x_temp_arr, invA, invA_arr, optim_mem);

  hipFree(scalars);
  hipFree(work);
  hipFree(pivotGPU);
  hipFree(iinfo);
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

extern "C" {

rocblas_status rocsolver_spotrf_strided_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    float *A, const rocblas_int lda, const rocblas_stride strideA,
    rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_potrf_strided_batched_impl<float, float>(
      handle, uplo, n, A, lda, strideA, info, batch_count);
}

rocblas_status rocsolver_dpotrf_strided_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    double *A, const rocblas_int lda, const rocblas_stride strideA,
    rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_potrf_strided_batched_impl<double, double>(
      handle, uplo, n, A, lda, strideA, info, batch_count);
}

rocblas_status rocsolver_cpotrf_strided_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_potrf_strided_batched_impl<float, rocblas_float_complex>(
      handle, uplo, n, A, lda, strideA, info, batch_count);
}

rocblas_status rocsolver_zpotrf_strided_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_potrf_strided_batched_impl<double, rocblas_double_complex>(
      handle, uplo, n, A, lda, strideA, info, batch_count);
}
}
