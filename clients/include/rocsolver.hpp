#ifndef ROCSOLVER_HPP
#define ROCSOLVER_HPP

#include "rocsolver.h"

template <typename T>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *A, rocblas_int lda);

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *A,
                                      rocblas_int lda) {
  return rocsolver_spotf2(handle, uplo, n, A, lda);
}

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *A,
                                      rocblas_int lda) {
  return rocsolver_dpotf2(handle, uplo, n, A, lda);
}

template <typename T>
inline rocblas_status rocsolver_getf2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info);

template <>
inline rocblas_status rocsolver_getf2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_sgetf2(handle, m, n, A, lda, ipiv, info);
}

template <>
inline rocblas_status rocsolver_getf2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_dgetf2(handle, m, n, A, lda, ipiv, info);
}

template <typename T>
inline rocblas_status rocsolver_getrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info);

template <>
inline rocblas_status rocsolver_getrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_sgetrf(handle, m, n, A, lda, ipiv, info);
}

template <>
inline rocblas_status rocsolver_getrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_dgetrf(handle, m, n, A, lda, ipiv, info);
}

template <typename T>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, const T *A, rocblas_int lda,
                const rocblas_int *ipiv, T *B, rocblas_int ldb);

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, const float *A, rocblas_int lda,
                const rocblas_int *ipiv, float *B, rocblas_int ldb) {
  return rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, const double *A, rocblas_int lda,
                const rocblas_int *ipiv, double *B, rocblas_int ldb) {
  return rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

#endif /* ROCSOLVER_HPP */
