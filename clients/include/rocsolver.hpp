#ifndef ROCSOLVER_HPP
#define ROCSOLVER_HPP

#include "rocsolver.h"

//larfg

template <typename T>
inline rocblas_status rocsolver_larfg(rocblas_handle handle, rocblas_int n, T *alpha, T *x, 
                                      rocblas_int incx, T *tau);

template <>
inline rocblas_status rocsolver_larfg(rocblas_handle handle, rocblas_int n, float *alpha, float *x, 
                                      rocblas_int incx, float *tau) {
  return rocsolver_slarfg(handle, n, alpha, x, incx, tau);
}

template <>
inline rocblas_status rocsolver_larfg(rocblas_handle handle, rocblas_int n, double *alpha, double *x, 
                                      rocblas_int incx, double *tau) {
  return rocsolver_dlarfg(handle, n, alpha, x, incx, tau);
}

//larf

template <typename T>
inline rocblas_status rocsolver_larf(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, T *x, 
                                      rocblas_int incx, T* alpha, T *A, rocblas_int lda);

template <>
inline rocblas_status rocsolver_larf(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, float *x, 
                                      rocblas_int incx, float *alpha, float *A, rocblas_int lda) {
  return rocsolver_slarf(handle, side, m, n, x, incx, alpha, A, lda);
}

template <>
inline rocblas_status rocsolver_larf(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, double *x, 
                                      rocblas_int incx, double *alpha, double *A, rocblas_int lda) {
  return rocsolver_dlarf(handle, side, m, n, x, incx, alpha, A, lda);
}

//larft

template <typename T>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocsolver_direct direct, rocblas_int n, rocblas_int k, T *V, 
                                      rocblas_int ldv, T* tau, T *F, rocblas_int ldt);

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocsolver_direct direct, rocblas_int n, rocblas_int k, float *V, 
                                      rocblas_int ldv, float *tau, float *F, rocblas_int ldt) {
  return rocsolver_slarft(handle, direct, n, k, V, ldv, tau, F, ldt);
}

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocsolver_direct direct, rocblas_int n, rocblas_int k, double *V, 
                                      rocblas_int ldv, double *tau, double *F, rocblas_int ldt) {
  return rocsolver_dlarft(handle, direct, n, k, V, ldv, tau, F, ldt);
}

//larfb

template <typename T>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocsolver_side side, rocsolver_operation trans, rocsolver_direct direct, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *V, 
                                      rocblas_int ldv, T *F, rocblas_int ldt, T *A, rocblas_int lda);

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocsolver_side side, rocsolver_operation trans, rocsolver_direct direct, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *V, 
                                      rocblas_int ldv, float *F, rocblas_int ldt, float *A, rocblas_int lda)
{
  return rocsolver_slarfb(handle, side, trans, direct, m, n, k, V, ldv, F, ldt, A, lda);
}

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocsolver_side side, rocsolver_operation trans, rocsolver_direct direct, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *V, 
                                      rocblas_int ldv, double *F, rocblas_int ldt, double *A, rocblas_int lda)
{
  return rocsolver_dlarfb(handle, side, trans, direct, m, n, k, V, ldv, F, ldt, A, lda);
}

//potf2

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

//getf2

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

//getf2_batched

template <typename T>
inline rocblas_status rocsolver_getf2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_getf2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_sgetf2_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getf2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dgetf2_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

//getf2_strided_batched

template <typename T>
inline rocblas_status rocsolver_getf2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_getf2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_sgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getf2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

//getrf

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

//getrf_batched

template <typename T>
inline rocblas_status rocsolver_getrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_getrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_sgetrf_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dgetrf_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

//getrf_strided_bacthed

template <typename T>
inline rocblas_status rocsolver_getrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_getrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_sgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

//getrs

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


//geqr2

template <typename T>
inline rocblas_status rocsolver_geqr2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      T *ipiv);

template <>
inline rocblas_status rocsolver_geqr2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      float *ipiv) {
  return rocsolver_sgeqr2(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_geqr2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      double *ipiv) {
  return rocsolver_dgeqr2(handle, m, n, A, lda, ipiv);
}

//geqr2_batched

template <typename T>
inline rocblas_status rocsolver_geqr2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_geqr2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgeqr2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqr2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgeqr2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

//geqr2_strided_batched

template <typename T>
inline rocblas_status rocsolver_geqr2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_geqr2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqr2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

//geqrf

template <typename T>
inline rocblas_status rocsolver_geqrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      T *ipiv);

template <>
inline rocblas_status rocsolver_geqrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      float *ipiv) {
  return rocsolver_sgeqrf(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_geqrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      double *ipiv) {
  return rocsolver_dgeqrf(handle, m, n, A, lda, ipiv);
}

//geqrf_batched

template <typename T>
inline rocblas_status rocsolver_geqrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_geqrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgeqrf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgeqrf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

//geqrf_strided_batched

template <typename T>
inline rocblas_status rocsolver_geqrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_geqrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}


#endif /* ROCSOLVER_HPP */
