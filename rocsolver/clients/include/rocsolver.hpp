#ifndef ROCSOLVER_HPP
#define ROCSOLVER_HPP

#include "rocsolver.h"

//lacgv

template <typename T>
inline rocblas_status rocsolver_lacgv(rocblas_handle handle, rocblas_int n, T *x, rocblas_int incx);

template <>
inline rocblas_status rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_float_complex *x, rocblas_int incx) {
  return rocsolver_clacgv(handle,n,x,incx);
}

template <>
inline rocblas_status rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_double_complex *x, rocblas_int incx) {
  return rocsolver_zlacgv(handle,n,x,incx);
}

//laswp

template <typename T>
inline rocblas_status rocsolver_laswp(rocblas_handle handle, rocblas_int n, T *A, rocblas_int lda,
                                      rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc);

template <>
inline rocblas_status rocsolver_laswp(rocblas_handle handle, rocblas_int n, float *A, rocblas_int lda,
                                      rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
  return rocsolver_slaswp(handle,n,A,lda,k1,k2,ipiv,inc);
}

template <>
inline rocblas_status rocsolver_laswp(rocblas_handle handle, rocblas_int n, double *A, rocblas_int lda,
                                      rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
  return rocsolver_dlaswp(handle,n,A,lda,k1,k2,ipiv,inc);
}

template <>
inline rocblas_status rocsolver_laswp(rocblas_handle handle, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
  return rocsolver_claswp(handle,n,A,lda,k1,k2,ipiv,inc);
}

template <>
inline rocblas_status rocsolver_laswp(rocblas_handle handle, rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
  return rocsolver_zlaswp(handle,n,A,lda,k1,k2,ipiv,inc);
}

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

template <>
inline rocblas_status rocsolver_larfg(rocblas_handle handle, rocblas_int n, rocblas_float_complex *alpha, rocblas_float_complex *x, 
                                      rocblas_int incx, rocblas_float_complex *tau) {
  return rocsolver_clarfg(handle, n, alpha, x, incx, tau);
}

template <>
inline rocblas_status rocsolver_larfg(rocblas_handle handle, rocblas_int n, rocblas_double_complex *alpha, rocblas_double_complex *x, 
                                      rocblas_int incx, rocblas_double_complex *tau) {
  return rocsolver_zlarfg(handle, n, alpha, x, incx, tau);
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

template <>
inline rocblas_status rocsolver_larf(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, rocblas_float_complex *x, 
                                      rocblas_int incx, rocblas_float_complex *alpha, rocblas_float_complex *A, rocblas_int lda) {
  return rocsolver_clarf(handle, side, m, n, x, incx, alpha, A, lda);
}

template <>
inline rocblas_status rocsolver_larf(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, rocblas_double_complex *x, 
                                      rocblas_int incx, rocblas_double_complex *alpha, rocblas_double_complex *A, rocblas_int lda) {
  return rocsolver_zlarf(handle, side, m, n, x, incx, alpha, A, lda);
}

//larft

template <typename T>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int n, rocblas_int k, T *V, 
                                      rocblas_int ldv, T* tau, T *F, rocblas_int ldt);

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int n, rocblas_int k, float *V, 
                                      rocblas_int ldv, float *tau, float *F, rocblas_int ldt) {
  return rocsolver_slarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int n, rocblas_int k, double *V, 
                                      rocblas_int ldv, double *tau, double *F, rocblas_int ldt) {
  return rocsolver_dlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int n, rocblas_int k, rocblas_float_complex *V, 
                                      rocblas_int ldv, rocblas_float_complex *tau, rocblas_float_complex *F, rocblas_int ldt) {
  return rocsolver_clarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

template <>
inline rocblas_status rocsolver_larft(rocblas_handle handle, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int n, rocblas_int k, rocblas_double_complex *V, 
                                      rocblas_int ldv, rocblas_double_complex *tau, rocblas_double_complex *F, rocblas_int ldt) {
  return rocsolver_zlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

//larfb

template <typename T>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocblas_side side, rocblas_operation trans, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, T *V, 
                                      rocblas_int ldv, T *F, rocblas_int ldt, T *A, rocblas_int lda);

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocblas_side side, rocblas_operation trans, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, float *V, 
                                      rocblas_int ldv, float *F, rocblas_int ldt, float *A, rocblas_int lda)
{
  return rocsolver_slarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocblas_side side, rocblas_operation trans, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, double *V, 
                                      rocblas_int ldv, double *F, rocblas_int ldt, double *A, rocblas_int lda)
{
  return rocsolver_dlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocblas_side side, rocblas_operation trans, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *V, 
                                      rocblas_int ldv, rocblas_float_complex *F, rocblas_int ldt, rocblas_float_complex *A, rocblas_int lda)
{
  return rocsolver_clarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

template <>
inline rocblas_status rocsolver_larfb(rocblas_handle handle, rocblas_side side, rocblas_operation trans, rocblas_direct direct, 
                                      rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *V, 
                                      rocblas_int ldv, rocblas_double_complex *F, rocblas_int ldt, rocblas_double_complex *A, rocblas_int lda)
{
  return rocsolver_zlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

//org2r & ung2r

template <typename T>
inline rocblas_status rocsolver_org2r_ung2r(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv);

template <>
inline rocblas_status rocsolver_org2r_ung2r(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv) {
  return rocsolver_sorg2r(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_org2r_ung2r(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv) {
  return rocsolver_dorg2r(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_org2r_ung2r(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv) {
  return rocsolver_cung2r(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_org2r_ung2r(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv) {
  return rocsolver_zung2r(handle, m, n, k, A, lda, Ipiv);
}

//orgqr & ungqr

template <typename T>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv);

template <>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv) {
  return rocsolver_sorgqr(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv) {
  return rocsolver_dorgqr(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv) {
  return rocsolver_cungqr(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv) {
  return rocsolver_zungqr(handle, m, n, k, A, lda, Ipiv);
}

//orgl2 & ungl2

template <typename T>
inline rocblas_status rocsolver_orgl2_ungl2(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv);

template <>
inline rocblas_status rocsolver_orgl2_ungl2(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv) {
  return rocsolver_sorgl2(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgl2_ungl2(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv) {
  return rocsolver_dorgl2(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgl2_ungl2(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv) {
  return rocsolver_cungl2(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgl2_ungl2(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv) {
  return rocsolver_zungl2(handle, m, n, k, A, lda, Ipiv);
}

//orglq & unglq

template <typename T>
inline rocblas_status rocsolver_orglq_unglq(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv);

template <>
inline rocblas_status rocsolver_orglq_unglq(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv) {
  return rocsolver_sorglq(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orglq_unglq(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv) {
  return rocsolver_dorglq(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orglq_unglq(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv) {
  return rocsolver_cunglq(handle, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orglq_unglq(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv) {
  return rocsolver_zunglq(handle, m, n, k, A, lda, Ipiv);
}

//ormbr & unmbr

template <typename T>
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle, rocblas_storev storev, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *A,
                                      rocblas_int lda, T *Ipiv, T *C, rocblas_int ldc);
 
template <>
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle, rocblas_storev storev, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                                      rocblas_int lda, float *Ipiv, float *C, rocblas_int ldc) {
    return rocsolver_sormbr(handle, storev, side, trans, m ,n ,k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle, rocblas_storev storev, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                                      rocblas_int lda, double *Ipiv, double *C, rocblas_int ldc) {
    return rocsolver_dormbr(handle, storev, side, trans, m ,n ,k, A, lda, Ipiv, C, ldc);
}
 
template <>
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle, rocblas_storev storev, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                                      rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *C, rocblas_int ldc) {
    return rocsolver_cunmbr(handle, storev, side, trans, m ,n ,k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle, rocblas_storev storev, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                                      rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *C, rocblas_int ldc) {
    return rocsolver_zunmbr(handle, storev, side, trans, m ,n ,k, A, lda, Ipiv, C, ldc);
}

//orgbr & ungbr

template <typename T>
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle, rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv);

template <>
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle, rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv) {
  return rocsolver_sorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle, rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv) {
  return rocsolver_dorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle, rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv) {
  return rocsolver_cungbr(handle, storev, m, n, k, A, lda, Ipiv);
}

template <>
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle, rocblas_storev storev, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv) {
  return rocsolver_zungbr(handle, storev, m, n, k, A, lda, Ipiv);
}

//orm2r & unm2r

template <typename T>
inline rocblas_status rocsolver_orm2r_unm2r(rocblas_handle handle, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv, T *C, rocblas_int ldc);

template <>
inline rocblas_status rocsolver_orm2r_unm2r(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv, float *C, rocblas_int ldc) {
  return rocsolver_sorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orm2r_unm2r(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv, double *C, rocblas_int ldc) {
  return rocsolver_dorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orm2r_unm2r(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *C, rocblas_int ldc) {
  return rocsolver_cunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orm2r_unm2r(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *C, rocblas_int ldc) {
  return rocsolver_zunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

//ormqr & unmqr

template <typename T>
inline rocblas_status rocsolver_ormqr_unmqr(rocblas_handle handle, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv, T *C, rocblas_int ldc);

template <>
inline rocblas_status rocsolver_ormqr_unmqr(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv, float *C, rocblas_int ldc) {
  return rocsolver_sormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormqr_unmqr(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv, double *C, rocblas_int ldc) {
  return rocsolver_dormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormqr_unmqr(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *C, rocblas_int ldc) {
  return rocsolver_cunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormqr_unmqr(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *C, rocblas_int ldc) {
  return rocsolver_zunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

//orml2 & unml2

template <typename T>
inline rocblas_status rocsolver_orml2_unml2(rocblas_handle handle, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv, T *C, rocblas_int ldc);

template <>
inline rocblas_status rocsolver_orml2_unml2(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv, float *C, rocblas_int ldc) {
  return rocsolver_sorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orml2_unml2(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv, double *C, rocblas_int ldc) {
  return rocsolver_dorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orml2_unml2(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *C, rocblas_int ldc) {
  return rocsolver_cunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_orml2_unml2(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *C, rocblas_int ldc) {
  return rocsolver_zunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

//ormlq & unmlq

template <typename T>
inline rocblas_status rocsolver_ormlq_unmlq(rocblas_handle handle, rocblas_side side, rocblas_operation trans, 
                                      rocblas_int m, rocblas_int n, rocblas_int k, T *A, 
                                      rocblas_int lda, T *Ipiv, T *C, rocblas_int ldc);

template <>
inline rocblas_status rocsolver_ormlq_unmlq(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, float *A,  
                                      rocblas_int lda, float *Ipiv, float *C, rocblas_int ldc) {
  return rocsolver_sormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormlq_unmlq(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, double *A,  
                                      rocblas_int lda, double *Ipiv, double *C, rocblas_int ldc) {
  return rocsolver_dormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormlq_unmlq(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,  
                                      rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *C, rocblas_int ldc) {
  return rocsolver_cunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

template <>
inline rocblas_status rocsolver_ormlq_unmlq(rocblas_handle handle, rocblas_side side, rocblas_operation trans,
                                      rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,  
                                      rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *C, rocblas_int ldc) {
  return rocsolver_zunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}


//potf2

template <typename T>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int *info);

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_spotf2(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_dpotf2(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_cpotf2(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_zpotf2(handle, uplo, n, A, lda, info);
}

//potf2_strided_batched

template <typename T>
inline rocblas_status rocsolver_potf2_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_potf2_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_spotf2_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dpotf2_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cpotf2_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zpotf2_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

//potf2_batched

template <typename T>
inline rocblas_status rocsolver_potf2_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *const A[], rocblas_int lda, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_potf2_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_spotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potf2_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}


//potrf

template <typename T>
inline rocblas_status rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int *info);

template <>
inline rocblas_status rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_spotrf(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_dpotrf(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_cpotrf(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *A,
                                      rocblas_int lda, rocblas_int *info) {
  return rocsolver_zpotrf(handle, uplo, n, A, lda, info);
}

//potrf_strided_batched

template <typename T>
inline rocblas_status rocsolver_potrf_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_potrf_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_spotrf_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dpotrf_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cpotrf_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_strided_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *A,
                                      rocblas_int lda, rocblas_int strideA, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zpotrf_strided_batched(handle, uplo, n, A, lda, strideA, info, batch_count);
}

//potrf_batched

template <typename T>
inline rocblas_status rocsolver_potrf_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, T *const A[], rocblas_int lda, rocblas_int *info, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_potrf_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, float *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_spotrf_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, double *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_dpotrf_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_float_complex *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cpotrf_batched(handle, uplo, n, A, lda, info, batch_count);
}

template <>
inline rocblas_status rocsolver_potrf_batched(rocblas_handle handle, rocblas_fill uplo,
                                      rocblas_int n, rocblas_double_complex *const A[],
                                      rocblas_int lda, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zpotrf_batched(handle, uplo, n, A, lda, info, batch_count);
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

template <>
inline rocblas_status rocsolver_getf2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_cgetf2(handle, m, n, A, lda, ipiv, info);
}

template <>
inline rocblas_status rocsolver_getf2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_zgetf2(handle, m, n, A, lda, ipiv, info);
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

template <>
inline rocblas_status rocsolver_getf2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cgetf2_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getf2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zgetf2_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
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

template <>
inline rocblas_status rocsolver_getf2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getf2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
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

template <>
inline rocblas_status rocsolver_getrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_cgetrf(handle, m, n, A, lda, ipiv, info);
}

template <>
inline rocblas_status rocsolver_getrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int *info) {
  return rocsolver_zgetrf(handle, m, n, A, lda, ipiv, info);
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

template <>
inline rocblas_status rocsolver_getrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cgetrf_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zgetrf_batched(handle, m, n, A, lda, ipiv, stridep, info, batch_count);
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

template <>
inline rocblas_status rocsolver_getrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_cgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

template <>
inline rocblas_status rocsolver_getrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_int *ipiv, rocblas_int stridep, rocblas_int *info, rocblas_int batch_count) {
  return rocsolver_zgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, info, batch_count);
}

//getrs

template <typename T>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, T *A, rocblas_int lda,
                rocblas_int *ipiv, T *B, rocblas_int ldb);

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, float *A, rocblas_int lda,
                rocblas_int *ipiv, float *B, rocblas_int ldb) {
  return rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, double *A, rocblas_int lda,
                rocblas_int *ipiv, double *B, rocblas_int ldb) {
  return rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_float_complex *A, rocblas_int lda,
                rocblas_int *ipiv, rocblas_float_complex *B, rocblas_int ldb) {
  return rocsolver_cgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

template <>
inline rocblas_status
rocsolver_getrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_double_complex *A, rocblas_int lda,
                rocblas_int *ipiv, rocblas_double_complex *B, rocblas_int ldb) {
  return rocsolver_zgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

//getrs_batched

template <typename T>
inline rocblas_status
rocsolver_getrs_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, T *const A[], rocblas_int lda,
                rocblas_int *ipiv, rocblas_int strideP, T *const B[], rocblas_int ldb, rocblas_int batch_count);

template <>
inline rocblas_status
rocsolver_getrs_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, float *const A[], rocblas_int lda,
                rocblas_int *ipiv, rocblas_int strideP, float *const B[], rocblas_int ldb, rocblas_int batch_count) {
    return rocsolver_sgetrs_batched(handle,trans,n,nrhs,A,lda,ipiv,strideP,B,ldb,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, double *const A[], rocblas_int lda,
                rocblas_int *ipiv, rocblas_int strideP, double *const B[], rocblas_int ldb, rocblas_int batch_count) {
    return rocsolver_dgetrs_batched(handle,trans,n,nrhs,A,lda,ipiv,strideP,B,ldb,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_float_complex *const A[], rocblas_int lda,
                rocblas_int *ipiv, rocblas_int strideP, rocblas_float_complex *const B[], rocblas_int ldb, rocblas_int batch_count) {
    return rocsolver_cgetrs_batched(handle,trans,n,nrhs,A,lda,ipiv,strideP,B,ldb,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_double_complex *const A[], rocblas_int lda,
                rocblas_int *ipiv, rocblas_int strideP, rocblas_double_complex *const B[], rocblas_int ldb, rocblas_int batch_count) {
    return rocsolver_zgetrs_batched(handle,trans,n,nrhs,A,lda,ipiv,strideP,B,ldb,batch_count);
}

//getrs_strided_batched

template <typename T>
inline rocblas_status
rocsolver_getrs_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, T *A, rocblas_int lda, rocblas_int strideA,
                rocblas_int *ipiv, rocblas_int strideP, T *B, rocblas_int ldb, rocblas_int strideB, rocblas_int batch_count);

template <>
inline rocblas_status
rocsolver_getrs_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, float *A, rocblas_int lda, rocblas_int strideA,
                rocblas_int *ipiv, rocblas_int strideP, float *B, rocblas_int ldb, rocblas_int strideB, rocblas_int batch_count) {
    return rocsolver_sgetrs_strided_batched(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, double *A, rocblas_int lda, rocblas_int strideA,
                rocblas_int *ipiv, rocblas_int strideP, double *B, rocblas_int ldb, rocblas_int strideB, rocblas_int batch_count) {
    return rocsolver_dgetrs_strided_batched(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                rocblas_int *ipiv, rocblas_int strideP, rocblas_float_complex *B, rocblas_int ldb, rocblas_int strideB, rocblas_int batch_count) {
    return rocsolver_cgetrs_strided_batched(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,batch_count);
}

template <>
inline rocblas_status
rocsolver_getrs_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                rocblas_int nrhs, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                rocblas_int *ipiv, rocblas_int strideP, rocblas_double_complex *B, rocblas_int ldb, rocblas_int strideB, rocblas_int batch_count) {
    return rocsolver_zgetrs_strided_batched(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,batch_count);
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

template <>
inline rocblas_status rocsolver_geqr2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_float_complex *ipiv) {
  return rocsolver_cgeqr2(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_geqr2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_double_complex *ipiv) {
  return rocsolver_zgeqr2(handle, m, n, A, lda, ipiv);
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

template <>
inline rocblas_status rocsolver_geqr2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgeqr2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqr2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgeqr2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
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

template <>
inline rocblas_status rocsolver_geqr2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqr2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
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

template <>
inline rocblas_status rocsolver_geqrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_float_complex *ipiv) {
  return rocsolver_cgeqrf(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_geqrf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_double_complex *ipiv) {
  return rocsolver_zgeqrf(handle, m, n, A, lda, ipiv);
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

template <>
inline rocblas_status rocsolver_geqrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgeqrf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqrf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgeqrf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
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

template <>
inline rocblas_status rocsolver_geqrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_geqrf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

//gelq2

template <typename T>
inline rocblas_status rocsolver_gelq2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      T *ipiv);

template <>
inline rocblas_status rocsolver_gelq2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      float *ipiv) {
  return rocsolver_sgelq2(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelq2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      double *ipiv) {
  return rocsolver_dgelq2(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelq2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_float_complex *ipiv) {
  return rocsolver_cgelq2(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelq2(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_double_complex *ipiv) {
  return rocsolver_zgelq2(handle, m, n, A, lda, ipiv);
}

//gelq2_batched

template <typename T>
inline rocblas_status rocsolver_gelq2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_gelq2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgelq2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgelq2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgelq2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgelq2_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

//gelq2_strided_batched

template <typename T>
inline rocblas_status rocsolver_gelq2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_gelq2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelq2_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

//gelqf

template <typename T>
inline rocblas_status rocsolver_gelqf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda,
                                      T *ipiv);

template <>
inline rocblas_status rocsolver_gelqf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda,
                                      float *ipiv) {
  return rocsolver_sgelqf(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelqf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda,
                                      double *ipiv) {
  return rocsolver_dgelqf(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelqf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                                      rocblas_float_complex *ipiv) {
  return rocsolver_cgelqf(handle, m, n, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_gelqf(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
                                      rocblas_double_complex *ipiv) {
  return rocsolver_zgelqf(handle, m, n, A, lda, ipiv);
}

//gelqf_batched

template <typename T>
inline rocblas_status rocsolver_gelqf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *const A[], rocblas_int lda,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_gelqf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *const A[], rocblas_int lda,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgelqf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *const A[], rocblas_int lda,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgelqf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *const A[], rocblas_int lda,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgelqf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *const A[], rocblas_int lda,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgelqf_batched(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

//gelqf_strided_batched

template <typename T>
inline rocblas_status rocsolver_gelqf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, T *A, rocblas_int lda, rocblas_int strideA,
                                      T *ipiv, rocblas_int stridep, rocblas_int batch_count);

template <>
inline rocblas_status rocsolver_gelqf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, float *A, rocblas_int lda, rocblas_int strideA,
                                      float *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_sgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, double *A, rocblas_int lda, rocblas_int strideA,
                                      double *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_dgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_float_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_cgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

template <>
inline rocblas_status rocsolver_gelqf_strided_batched(rocblas_handle handle, rocblas_int m,
                                      rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int strideA,
                                      rocblas_double_complex *ipiv, rocblas_int stridep, rocblas_int batch_count) {
  return rocsolver_zgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

#endif /* ROCSOLVER_HPP */
