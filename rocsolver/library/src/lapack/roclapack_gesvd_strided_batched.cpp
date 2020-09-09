/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvd.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_strided_batched_impl(
    rocblas_handle handle, const rocblas_svect left_svect,
    const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n,
    W A, const rocblas_int lda, const rocblas_stride strideA, TT *S,
    const rocblas_stride strideS, T *U, const rocblas_int ldu,
    const rocblas_stride strideU, T *V, const rocblas_int ldv,
    const rocblas_stride strideV, TT *E, const rocblas_stride strideE,
    const rocblas_workmode fast_alg, rocblas_int *info,
    const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_gesvd_argCheck(left_svect, right_svect, m, n, A, lda, S, U, ldu,
                               V, ldv, E, info, batch_count);
  if (st != rocblas_status_continue)
    return st;

  // memory managment
  // size for constants
  size_t size_1;
  // size of reusable workspace
  size_t size_2;
  // size of array of pointers to reusable workspace (only for batched case)
  size_t size_3;
  // size of fixed workspace (it cannot be re-used by other internal
  // subroutines)
  size_t size_4;
  // size of the array for the householder scalars
  size_t size_5;
  rocsolver_gesvd_getMemorySize<false, T, TT>(left_svect, right_svect, m, n,
                                              batch_count, &size_1, &size_2,
                                              &size_3, &size_4, &size_5);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *workgral, *workArr, *workfunc, *tau;
  hipMalloc(&scalars, size_1);
  hipMalloc(&workgral, size_2);
  hipMalloc(&workArr, size_3);
  hipMalloc(&workfunc, size_4);
  hipMalloc(&tau, size_5);
  if ((size_1 && !scalars) || (size_2 && !workgral) || (size_3 && !workArr) ||
      (size_4 && !workfunc) || (size_5 && !tau))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_gesvd_template<false, true, T>(
      handle, left_svect, right_svect, m, n, A, 0, lda, strideA, S, strideS, U,
      ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, info, batch_count,
      (T *)scalars, workgral, (T **)workArr, (T *)workfunc, (T *)tau);

  hipFree(scalars);
  hipFree(workgral);
  hipFree(workArr);
  hipFree(workfunc);
  hipFree(tau);

  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvd_strided_batched(
    rocblas_handle handle, const rocblas_svect left_svect,
    const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n,
    float *A, const rocblas_int lda, const rocblas_stride strideA, float *S,
    const rocblas_stride strideS, float *U, const rocblas_int ldu,
    const rocblas_stride strideU, float *V, const rocblas_int ldv,
    const rocblas_stride strideV, float *E, const rocblas_stride strideE,
    const rocblas_workmode fast_alg, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_gesvd_strided_batched_impl<float>(
      handle, left_svect, right_svect, m, n, A, lda, strideA, S, strideS, U,
      ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_dgesvd_strided_batched(
    rocblas_handle handle, const rocblas_svect left_svect,
    const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n,
    double *A, const rocblas_int lda, const rocblas_stride strideA, double *S,
    const rocblas_stride strideS, double *U, const rocblas_int ldu,
    const rocblas_stride strideU, double *V, const rocblas_int ldv,
    const rocblas_stride strideV, double *E, const rocblas_stride strideE,
    const rocblas_workmode fast_alg, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_gesvd_strided_batched_impl<double>(
      handle, left_svect, right_svect, m, n, A, lda, strideA, S, strideS, U,
      ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_cgesvd_strided_batched(
    rocblas_handle handle, const rocblas_svect left_svect,
    const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n,
    rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, float *S, const rocblas_stride strideS,
    rocblas_float_complex *U, const rocblas_int ldu,
    const rocblas_stride strideU, rocblas_float_complex *V,
    const rocblas_int ldv, const rocblas_stride strideV, float *E,
    const rocblas_stride strideE, const rocblas_workmode fast_alg,
    rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_gesvd_strided_batched_impl<rocblas_float_complex>(
      handle, left_svect, right_svect, m, n, A, lda, strideA, S, strideS, U,
      ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, info, batch_count);
}

rocblas_status rocsolver_zgesvd_strided_batched(
    rocblas_handle handle, const rocblas_svect left_svect,
    const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n,
    rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, double *S, const rocblas_stride strideS,
    rocblas_double_complex *U, const rocblas_int ldu,
    const rocblas_stride strideU, rocblas_double_complex *V,
    const rocblas_int ldv, const rocblas_stride strideV, double *E,
    const rocblas_stride strideE, const rocblas_workmode fast_alg,
    rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_gesvd_strided_batched_impl<rocblas_double_complex>(
      handle, left_svect, right_svect, m, n, A, lda, strideA, S, strideS, U,
      ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, info, batch_count);
}

} // extern C
