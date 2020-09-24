/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormtr_unmtr.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_impl(
    rocblas_handle handle, const rocblas_side side, const rocblas_fill uplo,
    const rocblas_operation trans, const rocblas_int m, const rocblas_int n,
    T *A, const rocblas_int lda, T *ipiv, T *C, const rocblas_int ldc) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_ormtr_argCheck<COMPLEX>(side, uplo, trans, m, n,
                                                        lda, ldc, A, C, ipiv);
  if (st != rocblas_status_continue)
    return st;

  // the matrices are shifted 0 entries (will work on the entire matrix)
  rocblas_int shiftA = 0;
  rocblas_int shiftC = 0;
  rocblas_int strideA = 0;
  rocblas_int strideP = 0;
  rocblas_int strideC = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3; // size of array of pointers to workspace
  size_t size_4; // size of temporary array for triangular factor
  size_t size_5; // workspace for TRMM calls
  rocsolver_ormtr_unmtr_getMemorySize<T, false>(side, uplo, m, n, batch_count,
                                                &size_1, &size_2, &size_3,
                                                &size_4, &size_5);

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
  rocblas_status status = rocsolver_ormtr_unmtr_template<false, false, T>(
      handle, side, uplo, trans, m, n, A, shiftA, lda, strideA, ipiv, strideP,
      C, shiftC, ldc, strideC, batch_count, (T *)scalars, (T *)work,
      (T **)workArr, (T *)trfact, (T *)workTrmm);

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

rocblas_status rocsolver_sormtr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                float *A, const rocblas_int lda, float *ipiv,
                                float *C, const rocblas_int ldc) {
  return rocsolver_ormtr_unmtr_impl<float>(handle, side, uplo, trans, m, n, A,
                                           lda, ipiv, C, ldc);
}

rocblas_status rocsolver_dormtr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                double *A, const rocblas_int lda, double *ipiv,
                                double *C, const rocblas_int ldc) {
  return rocsolver_ormtr_unmtr_impl<double>(handle, side, uplo, trans, m, n, A,
                                            lda, ipiv, C, ldc);
}

rocblas_status rocsolver_cunmtr(rocblas_handle handle, const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m, const rocblas_int n,
                                rocblas_float_complex *A, const rocblas_int lda,
                                rocblas_float_complex *ipiv,
                                rocblas_float_complex *C,
                                const rocblas_int ldc) {
  return rocsolver_ormtr_unmtr_impl<rocblas_float_complex>(
      handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc);
}

rocblas_status
rocsolver_zunmtr(rocblas_handle handle, const rocblas_side side,
                 const rocblas_fill uplo, const rocblas_operation trans,
                 const rocblas_int m, const rocblas_int n,
                 rocblas_double_complex *A, const rocblas_int lda,
                 rocblas_double_complex *ipiv, rocblas_double_complex *C,
                 const rocblas_int ldc) {
  return rocsolver_ormtr_unmtr_impl<rocblas_double_complex>(
      handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc);
}

} // extern C
