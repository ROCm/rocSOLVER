/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormqr_unmqr.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status
rocsolver_ormqr_unmqr_impl(rocblas_handle handle, const rocblas_side side,
                           const rocblas_operation trans, const rocblas_int m,
                           const rocblas_int n, const rocblas_int k, T *A,
                           const rocblas_int lda, T *ipiv, T *C,
                           const rocblas_int ldc) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_orm2r_ormqr_argCheck<COMPLEX>(
      side, trans, m, n, k, lda, ldc, A, C, ipiv);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;
  rocblas_stride strideP = 0;
  rocblas_stride strideC = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3; // size of array of pointers to workspace
  size_t size_4; // size of triangular factor or diagonal elements
  size_t size_5; // size of workspace for TRMM calls
  rocsolver_ormqr_unmqr_getMemorySize<T, false>(
      side, m, n, k, batch_count, &size_1, &size_2, &size_3, &size_4, &size_5);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *workArr, *trfact, *workTrmm;
  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&workArr, size_3);
  hipMalloc(&trfact, size_4);
  hipMalloc(&workTrmm, size_5);
  if (!scalars || (size_2 && !work) || (size_3 && !workArr) ||
      (size_4 && !trfact) || (size_5 && !workTrmm))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_ormqr_unmqr_template<false, false, T>(
      handle, side, trans, m, n, k, A, 0, // shifted 0 entries
      lda, strideA, ipiv, strideP, C, 0, ldc, strideC, batch_count,
      (T *)scalars, (T *)work, (T **)workArr, (T *)trfact, (T *)workTrmm);

  hipFree(scalars);
  hipFree(work);
  hipFree(workArr);
  hipFree(trfact);
  hipFree(workTrmm);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sormqr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                const rocblas_int k, float *A,
                                const rocblas_int lda, float *ipiv, float *C,
                                const rocblas_int ldc) {
  return rocsolver_ormqr_unmqr_impl<float>(handle, side, trans, m, n, k, A, lda,
                                           ipiv, C, ldc);
}

rocblas_status rocsolver_dormqr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                const rocblas_int k, double *A,
                                const rocblas_int lda, double *ipiv, double *C,
                                const rocblas_int ldc) {
  return rocsolver_ormqr_unmqr_impl<double>(handle, side, trans, m, n, k, A,
                                            lda, ipiv, C, ldc);
}

rocblas_status rocsolver_cunmqr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                const rocblas_int k, rocblas_float_complex *A,
                                const rocblas_int lda,
                                rocblas_float_complex *ipiv,
                                rocblas_float_complex *C,
                                const rocblas_int ldc) {
  return rocsolver_ormqr_unmqr_impl<rocblas_float_complex>(
      handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_zunmqr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                const rocblas_int k, rocblas_double_complex *A,
                                const rocblas_int lda,
                                rocblas_double_complex *ipiv,
                                rocblas_double_complex *C,
                                const rocblas_int ldc) {
  return rocsolver_ormqr_unmqr_impl<rocblas_double_complex>(
      handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

} // extern C
