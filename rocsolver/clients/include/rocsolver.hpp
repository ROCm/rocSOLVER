/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSOLVER_HPP
#define ROCSOLVER_HPP

#include "clientcommon.hpp"
#include "rocsolver.h"

// The following functions are not included in the public API and must be
// declared

#ifdef __cplusplus
extern "C" {
#endif

rocblas_status rocsolver_sgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            float* const A[],
                                            const rocblas_int lda,
                                            float* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_dgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            double* const A[],
                                            const rocblas_int lda,
                                            double* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_cgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            const rocblas_int lda,
                                            rocblas_float_complex* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_zgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            const rocblas_int lda,
                                            rocblas_double_complex* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   float* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   float* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count);

rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   double* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   double* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count);

rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_float_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_float_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count);

rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_double_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_double_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count);

#ifdef __cplusplus
}
#endif

/******************** LACGV ********************/
inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_float_complex* x, rocblas_int incx)
{
    return rocsolver_clacgv(handle, n, x, incx);
}

inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_double_complex* x, rocblas_int incx)
{
    return rocsolver_zlacgv(handle, n, x, incx);
}
/*****************************************************/

/******************** LASWP ********************/
inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_slaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_dlaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_claswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_zlaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}
/*****************************************************/

/******************** LARFG ********************/
inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      float* alpha,
                                      float* x,
                                      rocblas_int incx,
                                      float* tau)
{
    return rocsolver_slarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      double* alpha,
                                      double* x,
                                      rocblas_int incx,
                                      double* tau)
{
    return rocsolver_dlarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* alpha,
                                      rocblas_float_complex* x,
                                      rocblas_int incx,
                                      rocblas_float_complex* tau)
{
    return rocsolver_clarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* alpha,
                                      rocblas_double_complex* x,
                                      rocblas_int incx,
                                      rocblas_double_complex* tau)
{
    return rocsolver_zlarfg(handle, n, alpha, x, incx, tau);
}
/*****************************************************/

/******************** LARF ********************/
inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     float* x,
                                     rocblas_int incx,
                                     float* alpha,
                                     float* A,
                                     rocblas_int lda)
{
    return rocsolver_slarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     double* x,
                                     rocblas_int incx,
                                     double* alpha,
                                     double* A,
                                     rocblas_int lda)
{
    return rocsolver_dlarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex* x,
                                     rocblas_int incx,
                                     rocblas_float_complex* alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_clarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_double_complex* x,
                                     rocblas_int incx,
                                     rocblas_double_complex* alpha,
                                     rocblas_double_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_zlarf(handle, side, m, n, x, incx, alpha, A, lda);
}
/*****************************************************/

/******************** LARFT ********************/
inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* V,
                                      rocblas_int ldv,
                                      float* tau,
                                      float* F,
                                      rocblas_int ldt)
{
    return rocsolver_slarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* V,
                                      rocblas_int ldv,
                                      double* tau,
                                      double* F,
                                      rocblas_int ldt)
{
    return rocsolver_dlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* F,
                                      rocblas_int ldt)
{
    return rocsolver_clarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* tau,
                                      rocblas_double_complex* F,
                                      rocblas_int ldt)
{
    return rocsolver_zlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}
/*****************************************************/

/******************** LARFB ********************/
inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* V,
                                      rocblas_int ldv,
                                      float* F,
                                      rocblas_int ldt,
                                      float* A,
                                      rocblas_int lda)
{
    return rocsolver_slarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* V,
                                      rocblas_int ldv,
                                      double* F,
                                      rocblas_int ldt,
                                      double* A,
                                      rocblas_int lda)
{
    return rocsolver_dlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* F,
                                      rocblas_int ldt,
                                      rocblas_float_complex* A,
                                      rocblas_int lda)
{
    return rocsolver_clarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* F,
                                      rocblas_int ldt,
                                      rocblas_double_complex* A,
                                      rocblas_int lda)
{
    return rocsolver_zlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}
/***************************************************************/

/******************** BDSQR ********************/
inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      float* D,
                                      float* E,
                                      float* V,
                                      rocblas_int ldv,
                                      float* U,
                                      rocblas_int ldu,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_sbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      double* D,
                                      double* E,
                                      double* V,
                                      rocblas_int ldv,
                                      double* U,
                                      rocblas_int ldu,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_cbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}
/***************************************************************/

/******************** LABRD ********************/
inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      float* A,
                                      rocblas_int lda,
                                      float* D,
                                      float* E,
                                      float* tauq,
                                      float* taup,
                                      float* X,
                                      rocblas_int ldx,
                                      float* Y,
                                      rocblas_int ldy)
{
    return rocsolver_slabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      double* A,
                                      rocblas_int lda,
                                      double* D,
                                      double* E,
                                      double* tauq,
                                      double* taup,
                                      double* X,
                                      rocblas_int ldx,
                                      double* Y,
                                      rocblas_int ldy)
{
    return rocsolver_dlabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* tauq,
                                      rocblas_float_complex* taup,
                                      rocblas_float_complex* X,
                                      rocblas_int ldx,
                                      rocblas_float_complex* Y,
                                      rocblas_int ldy)
{
    return rocsolver_clabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* tauq,
                                      rocblas_double_complex* taup,
                                      rocblas_double_complex* X,
                                      rocblas_int ldx,
                                      rocblas_double_complex* Y,
                                      rocblas_int ldy)
{
    return rocsolver_zlabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}
/***************************************************************/

/******************** ORGxR_UNGxR ********************/
inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GQR ? rocsolver_sorgqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorg2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GQR ? rocsolver_dorgqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorg2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GQR ? rocsolver_cungqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cung2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GQR ? rocsolver_zungqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zung2r(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGLx_UNGLx ********************/
inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GLQ ? rocsolver_sorglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorgl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GLQ ? rocsolver_dorglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorgl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GLQ ? rocsolver_cunglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cungl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GLQ ? rocsolver_zunglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zungl2(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGxL_UNGxL ********************/
inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GQL ? rocsolver_sorgql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorg2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GQL ? rocsolver_dorgql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorg2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GQL ? rocsolver_cungql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cung2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GQL ? rocsolver_zungql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zung2l(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGBR_UNGBR ********************/
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return rocsolver_sorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return rocsolver_dorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return rocsolver_cungbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return rocsolver_zungbr(handle, storev, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGTR_UNGTR ********************/
inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return rocsolver_sorgtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return rocsolver_dorgtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return rocsolver_cungtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return rocsolver_zungtr(handle, uplo, n, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORMxR_UNMxR ********************/
inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_sormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_dormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_cunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_zunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMLx_UNMLx ********************/
inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_sormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_dormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_cunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_zunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMxL_UNMxL ********************/
inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_sormql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_dormql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_cunmql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_zunmql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMBR_UNMBR ********************/
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return rocsolver_sormbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return rocsolver_dormbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_cunmbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_zunmbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMTR_UNMTR ********************/
inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return rocsolver_sormtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return rocsolver_dormtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_cunmtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_zunmtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** POTF2_POTRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_spotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_spotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_spotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_spotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_dpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_dpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_dpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_dpotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_cpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_cpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_cpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_cpotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_zpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_zpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_zpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_zpotf2(handle, uplo, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_spotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_spotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_dpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_dpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_cpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_cpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_zpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_zpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}
/********************************************************/

/******************** GETF2_GETRF_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 float* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_sgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_sgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_sgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_sgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 double* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_dgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_dgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_dgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_dgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_cgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_cgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_cgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_cgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_zgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_zgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_zgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_zgetf2_npvt(handle, m, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 float* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_sgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_sgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 double* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_dgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_dgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_float_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_cgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_cgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_zgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_zgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}
/********************************************************/

/******************** GETF2_GETRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_sgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_sgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_sgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_sgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_dgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_dgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_dgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_dgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_cgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_cgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_cgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_cgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_zgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_zgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_zgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_zgetf2(handle, m, n, A, lda, ipiv, info);
}

// batched
inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_sgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_sgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_dgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_dgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_cgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_cgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_zgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_zgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_sgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_dgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_cgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_zgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

// batched
inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_sgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_cgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_zgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      float* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      double* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_cgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_zgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

// batched
inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      float* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_sgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      double* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_dgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_cgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_zgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}
/********************************************************/

/******************** GETRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      float* A1,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_sgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      double* A1,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_dgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* A1,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_cgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* A1,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_zgetri(handle, n, A, lda, ipiv, info);
}

// batched and outofplace_batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      float* const A1[],
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgetri_batched(handle, n, A, lda, ipiv, stP, info, bc)
        : rocsolver_sgetri_outofplace_batched(handle, n, A1, lda, ipiv, stP, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      double* const A1[],
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgetri_batched(handle, n, A, lda, ipiv, stP, info, bc)
        : rocsolver_dgetri_outofplace_batched(handle, n, A1, lda, ipiv, stP, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* const A1[],
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgetri_batched(handle, n, A, lda, ipiv, stP, info, bc)
        : rocsolver_cgetri_outofplace_batched(handle, n, A1, lda, ipiv, stP, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* const A1[],
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgetri_batched(handle, n, A, lda, ipiv, stP, info, bc)
        : rocsolver_zgetri_outofplace_batched(handle, n, A1, lda, ipiv, stP, A, lda, info, bc);
}
/********************************************************/

/******************** GEQR2_GEQRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_sgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_sgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_dgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_dgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_cgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_cgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_zgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_zgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgeqr2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_sgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_dgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_cgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_zgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

// ptr_batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_sgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_dgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_cgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_zgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}
/********************************************************/

/******************** GEQL2_GEQLF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_sgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_sgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_dgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_dgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_cgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_cgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_zgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_zgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgeql2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_sgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_dgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_cgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_zgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GELQ2_GELQF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_sgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_sgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_dgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_dgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_cgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_cgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_zgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_zgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgelq2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_sgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_dgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_cgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_zgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GEBD2_GEBRD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tauq,
                                            rocblas_stride stQ,
                                            float* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_sgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_sgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_sgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_sgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tauq,
                                            rocblas_stride stQ,
                                            double* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_dgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_dgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_dgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_dgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_float_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_cgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_cgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_cgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_cgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_double_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_zgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_zgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_zgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_zgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

// batched
inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tauq,
                                            rocblas_stride stQ,
                                            float* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_sgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_sgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tauq,
                                            rocblas_stride stQ,
                                            double* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_dgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_dgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_float_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_cgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_cgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_double_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_zgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_zgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}
/********************************************************/

#endif /* ROCSOLVER_HPP */
