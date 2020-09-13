/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.hpp"

template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const float *x, const rocblas_int incx,
                            float *result) {
  return rocblas_snrm2(handle, n, x, incx, result);
}
template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const double *x, const rocblas_int incx,
                            double *result) {
  return rocblas_dnrm2(handle, n, x, incx, result);
}

/*template <>
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
}*/

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
                            const rocblas_float_complex *alpha,
                            rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb) {
  return rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                       ldb);
}
template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const rocblas_double_complex *alpha,
                            rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb) {
  return rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,
                       ldb);
}

/*template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation trans,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            float *alpha, float *A, rocblas_int lda, float *B,
                            rocblas_int ldb) {
  return rocblas_strmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation trans,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            double *alpha, double *A, rocblas_int lda,
                            double *B, rocblas_int ldb) {
  return rocblas_dtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation trans,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            rocblas_float_complex *alpha,
                            rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb) {
  return rocblas_ctrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation trans,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            rocblas_double_complex *alpha,
                            rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb) {
  return rocblas_ztrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

// trtri
template <>
rocblas_status rocblas_trtri(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_diagonal diag, rocblas_int n,
                             const float *A, rocblas_int lda, float *invA,
                             rocblas_int ldinvA) {
  return rocblas_strtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
rocblas_status rocblas_trtri(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_diagonal diag, rocblas_int n,
                             const double *A, rocblas_int lda, double *invA,
                             rocblas_int ldinvA) {
  return rocblas_dtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
rocblas_status rocblas_trtri(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_diagonal diag, rocblas_int n,
                             const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *invA, rocblas_int ldinvA) {
  return rocblas_ctrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
rocblas_status rocblas_trtri(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_diagonal diag, rocblas_int n,
                             const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *invA, rocblas_int ldinvA) {
  return rocblas_ztrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}*/
