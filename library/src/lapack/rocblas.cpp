#include "rocblas.hpp"

template <>
rocblas_status rocblas_scal(rocblas_handle handle, rocblas_int n,
                            const float *alpha, float *x, rocblas_int incx) {
  return rocblas_sscal(handle, n, alpha, x, incx);
}

template <>
rocblas_status rocblas_scal(rocblas_handle handle, rocblas_int n,
                            const double *alpha, double *x, rocblas_int incx) {
  return rocblas_dscal(handle, n, alpha, x, incx);
}

template <>
rocblas_status rocblas_swap(rocblas_handle handle, rocblas_int n, float *x,
                            rocblas_int incx, float *y, rocblas_int incy) {
  return rocblas_sswap(handle, n, x, incx, y, incy);
}

template <>
rocblas_status rocblas_swap(rocblas_handle handle, rocblas_int n, double *x,
                            rocblas_int incx, double *y, rocblas_int incy) {
  return rocblas_dswap(handle, n, x, incx, y, incy);
}

template <>
rocblas_status rocblas_dot(rocblas_handle handle, rocblas_int n, const float *x,
                           rocblas_int incx, const float *y, rocblas_int incy,
                           float *result) {
  return rocblas_sdot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblas_dot(rocblas_handle handle, rocblas_int n,
                           const double *x, rocblas_int incx, const double *y,
                           rocblas_int incy, double *result) {
  return rocblas_ddot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const float *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_isamax(handle, n, x, incx, result);
}

template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const double *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_idamax(handle, n, x, incx, result);
}

template <>
rocblas_status rocblas_ger(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const float *alpha, const float *x, rocblas_int incx,
                           const float *y, rocblas_int incy, float *A,
                           rocblas_int lda) {
  return rocblas_sger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
rocblas_status rocblas_ger(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const double *alpha, const double *x,
                           rocblas_int incx, const double *y, rocblas_int incy,
                           double *A, rocblas_int lda) {
  return rocblas_dger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
rocblas_status rocblas_gemv(rocblas_handle handle, rocblas_operation transA,
                            rocblas_int m, rocblas_int n, const float *alpha,
                            const float *A, rocblas_int lda, const float *x,
                            rocblas_int incx, const float *beta, float *y,
                            rocblas_int incy) {
  return rocblas_sgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}

template <>
rocblas_status rocblas_gemv(rocblas_handle handle, rocblas_operation transA,
                            rocblas_int m, rocblas_int n, const double *alpha,
                            const double *A, rocblas_int lda, const double *x,
                            rocblas_int incx, const double *beta, double *y,
                            rocblas_int incy) {
  return rocblas_dgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}
