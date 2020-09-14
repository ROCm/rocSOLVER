/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

template <typename T>
rocblas_status
rocsolver_getrs_impl(rocblas_handle handle, const rocblas_operation trans,
                     const rocblas_int n, const rocblas_int nrhs, T *A,
                     const rocblas_int lda, const rocblas_int *ipiv, T *B,
                     const rocblas_int ldb) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_getrs_argCheck(trans, n, nrhs, lda, ldb, A, B, ipiv);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;
  rocblas_stride strideB = 0;
  rocblas_stride strideP = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // for TRSM x_temp
  size_t size_2; // for TRSM x_temp_arr
  size_t size_3; // for TRSM invA
  size_t size_4; // for TRSM invA_arr
  rocsolver_getrs_getMemorySize<false, T>(n, nrhs, batch_count, &size_1,
                                          &size_2, &size_3, &size_4);

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
  rocblas_status status = rocsolver_getrs_template<false, T>(
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

extern "C" rocblas_status
rocsolver_sgetrs(rocblas_handle handle, const rocblas_operation trans,
                 const rocblas_int n, const rocblas_int nrhs, float *A,
                 const rocblas_int lda, const rocblas_int *ipiv, float *B,
                 const rocblas_int ldb) {
  return rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B,
                                     ldb);
}

extern "C" rocblas_status
rocsolver_dgetrs(rocblas_handle handle, const rocblas_operation trans,
                 const rocblas_int n, const rocblas_int nrhs, double *A,
                 const rocblas_int lda, const rocblas_int *ipiv, double *B,
                 const rocblas_int ldb) {
  return rocsolver_getrs_impl<double>(handle, trans, n, nrhs, A, lda, ipiv, B,
                                      ldb);
}

extern "C" rocblas_status rocsolver_cgetrs(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_int *ipiv, rocblas_float_complex *B, const rocblas_int ldb) {
  return rocsolver_getrs_impl<rocblas_float_complex>(handle, trans, n, nrhs, A,
                                                     lda, ipiv, B, ldb);
}

extern "C" rocblas_status rocsolver_zgetrs(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_int *ipiv, rocblas_double_complex *B, const rocblas_int ldb) {
  return rocsolver_getrs_impl<rocblas_double_complex>(handle, trans, n, nrhs, A,
                                                      lda, ipiv, B, ldb);
}
