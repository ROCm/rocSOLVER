/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.hpp"

//nrm2

template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const float* x, const rocblas_int incx, float* result) {
  return rocblas_snrm2(handle, n, x, incx, result);
}

template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const double* x, const rocblas_int incx, double* result) {
  return rocblas_dnrm2(handle, n, x, incx, result);
}

//scal

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
rocblas_status rocblas_scal(rocblas_handle handle, rocblas_int n,
                            const rocblas_float_complex *alpha, rocblas_float_complex *x, rocblas_int incx) {
  return rocblas_cscal(handle, n, alpha, x, incx);
}

template <>
rocblas_status rocblas_scal(rocblas_handle handle, rocblas_int n,
                            const rocblas_double_complex *alpha, rocblas_double_complex *x, rocblas_int incx) {
  return rocblas_zscal(handle, n, alpha, x, incx);
}

//swap

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

//dot

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

//iamax

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
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const rocblas_float_complex *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_icamax(handle, n, x, incx, result);
}

template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const rocblas_double_complex *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_izamax(handle, n, x, incx, result);
}

//ger 

template <>
rocblas_status rocblas_ger<false>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const float *alpha, const float *x, rocblas_int incx,
                           const float *y, rocblas_int incy, float *A,
                           rocblas_int lda) {
  return rocblas_sger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
rocblas_status rocblas_ger<false>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const double *alpha, const double *x,
                           rocblas_int incx, const double *y, rocblas_int incy,
                           double *A, rocblas_int lda) {
  return rocblas_dger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

//geru

template <>
rocblas_status rocblas_ger<false>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                           const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *A,
                           rocblas_int lda) {
  return rocblas_cgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
rocblas_status rocblas_ger<false>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const rocblas_double_complex *alpha, const rocblas_double_complex *x,
                           rocblas_int incx, const rocblas_double_complex *y, rocblas_int incy,
                           rocblas_double_complex *A, rocblas_int lda) {
  return rocblas_zgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

//gerc

template <>
rocblas_status rocblas_ger<true>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                           const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *A,
                           rocblas_int lda) {
  return rocblas_cgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
rocblas_status rocblas_ger<true>(rocblas_handle handle, rocblas_int m, rocblas_int n,
                           const rocblas_double_complex *alpha, const rocblas_double_complex *x,
                           rocblas_int incx, const rocblas_double_complex *y, rocblas_int incy,
                           rocblas_double_complex *A, rocblas_int lda) {
  return rocblas_zgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}


//gemv

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

//gemm

template <>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k, const float *alpha,
                            const float *A, rocblas_int lda, const float *B,
                            rocblas_int ldb, const float *beta, float *C,
                            rocblas_int ldc) {
  return rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k, const double *alpha,
                            const double *A, rocblas_int lda, const double *B,
                            rocblas_int ldb, const double *beta, double *C,
                            rocblas_int ldc) {
  return rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k, const rocblas_float_complex *alpha,
                            const rocblas_float_complex *A, rocblas_int lda, const rocblas_float_complex *B,
                            rocblas_int ldb, const rocblas_float_complex *beta, rocblas_float_complex *C,
                            rocblas_int ldc) {
  return rocblas_cgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k, const rocblas_double_complex *alpha,
                            const rocblas_double_complex *A, rocblas_int lda, const rocblas_double_complex *B,
                            rocblas_int ldb, const rocblas_double_complex *beta, rocblas_double_complex *C,
                            rocblas_int ldc) {
  return rocblas_zgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

//trsm

template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const float *alpha, float *A, rocblas_int lda,
                            float *B, rocblas_int ldb) {
  return rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                       ldb);
}

template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const double *alpha, double *A, rocblas_int lda,
                            double *B, rocblas_int ldb) {
  return rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                       ldb);
}

template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const rocblas_float_complex *alpha, rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb) {
    return rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const rocblas_double_complex *alpha, rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb) {
    return rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

//trmm

template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                            rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            float *alpha, float *A, rocblas_int lda, float* B, rocblas_int ldb)
{
    return rocblas_strmm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}

template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                            rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            double *alpha, double *A, rocblas_int lda, double* B, rocblas_int ldb)
{
    return rocblas_dtrmm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
