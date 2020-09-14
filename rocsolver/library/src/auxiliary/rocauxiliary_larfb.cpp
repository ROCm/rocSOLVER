/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfb.hpp"

template <typename T>
rocblas_status
rocsolver_larfb_impl(rocblas_handle handle, const rocblas_side side,
                     const rocblas_operation trans, const rocblas_direct direct,
                     const rocblas_storev storev, const rocblas_int m,
                     const rocblas_int n, const rocblas_int k, T *V,
                     const rocblas_int ldv, T *F, const rocblas_int ldf, T *A,
                     const rocblas_int lda) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_larfb_argCheck(side, trans, direct, storev, m,
                                               n, k, ldv, ldf, lda, V, A, F);
  if (st != rocblas_status_continue)
    return st;

  // the matrices are shifted 0 entries (will work on the entire matrix)
  rocblas_int shiftv = 0;
  rocblas_int shifta = 0;
  rocblas_int shiftf = 0;
  rocblas_stride stridev = 0;
  rocblas_stride stridea = 0;
  rocblas_stride stridef = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // size of workspace
  size_t size_2; // size of array of pointers to workspace
  rocsolver_larfb_getMemorySize<T, false>(side, m, n, k, batch_count, &size_1,
                                          &size_2);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *work, *workArr;
  hipMalloc(&work, size_1);
  hipMalloc(&workArr, size_2);
  if ((size_1 && !work) || (size_2 && !workArr))
    return rocblas_status_memory_error;

  //  execution
  rocblas_status status = rocsolver_larfb_template<false, false, T>(
      handle, side, trans, direct, storev, m, n, k, V, shiftv, ldv, stridev, F,
      shiftf, ldf, stridef, A, shifta, lda, stridea, batch_count, (T *)work,
      (T **)workArr);

  hipFree(work);
  hipFree(workArr);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfb(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_direct direct,
    const rocblas_storev storev, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, float *V, const rocblas_int ldv, float *T,
    const rocblas_int ldt, float *A, const rocblas_int lda) {
  return rocsolver_larfb_impl<float>(handle, side, trans, direct, storev, m, n,
                                     k, V, ldv, T, ldt, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfb(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_direct direct,
    const rocblas_storev storev, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, double *V, const rocblas_int ldv, double *T,
    const rocblas_int ldt, double *A, const rocblas_int lda) {
  return rocsolver_larfb_impl<double>(handle, side, trans, direct, storev, m, n,
                                      k, V, ldv, T, ldt, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_clarfb(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_direct direct,
    const rocblas_storev storev, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, rocblas_float_complex *V, const rocblas_int ldv,
    rocblas_float_complex *T, const rocblas_int ldt, rocblas_float_complex *A,
    const rocblas_int lda) {
  return rocsolver_larfb_impl<rocblas_float_complex>(
      handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarfb(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_direct direct,
    const rocblas_storev storev, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, rocblas_double_complex *V, const rocblas_int ldv,
    rocblas_double_complex *T, const rocblas_int ldt, rocblas_double_complex *A,
    const rocblas_int lda) {
  return rocsolver_larfb_impl<rocblas_double_complex>(
      handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}

} // extern C
