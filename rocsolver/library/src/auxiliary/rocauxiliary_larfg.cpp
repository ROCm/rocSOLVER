/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfg.hpp"

template <typename T>
rocblas_status rocsolver_larfg_impl(rocblas_handle handle, const rocblas_int n,
                                    T *alpha, T *x, const rocblas_int incx,
                                    T *tau) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_larfg_argCheck(n, incx, alpha, x, tau);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride stridex = 0;
  rocblas_stride strideP = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // size to store the norms
  size_t size_2; // size of workspace
  rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_1, &size_2);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *norms, *work;
  hipMalloc(&norms, size_1);
  hipMalloc(&work, size_2);
  if (!norms || (size_2 && !work))
    return rocblas_status_memory_error;

  // execution
  rocblas_status status = rocsolver_larfg_template<T>(
      handle, n, alpha, 0, // The pivot is the first pointed element
      x, 0,                // the vector is shifted 0 entries,
      incx, stridex, tau, strideP, batch_count, (T *)norms, (T *)work);

  hipFree(norms);
  hipFree(work);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float *alpha, float *x,
                                                 const rocblas_int incx,
                                                 float *tau) {
  return rocsolver_larfg_impl<float>(handle, n, alpha, x, incx, tau);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double *alpha, double *x,
                                                 const rocblas_int incx,
                                                 double *tau) {
  return rocsolver_larfg_impl<double>(handle, n, alpha, x, incx, tau);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_clarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_float_complex *alpha,
                                                 rocblas_float_complex *x,
                                                 const rocblas_int incx,
                                                 rocblas_float_complex *tau) {
  return rocsolver_larfg_impl<rocblas_float_complex>(handle, n, alpha, x, incx,
                                                     tau);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_double_complex *alpha,
                                                 rocblas_double_complex *x,
                                                 const rocblas_int incx,
                                                 rocblas_double_complex *tau) {
  return rocsolver_larfg_impl<rocblas_double_complex>(handle, n, alpha, x, incx,
                                                      tau);
}

} // extern C
