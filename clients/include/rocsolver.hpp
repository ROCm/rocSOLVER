/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "rocsolver.h"

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

/***** Functions not included in the public API that must be declared *****/
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

rocblas_status rocsolver_sgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          float* B,
                                          const rocblas_int ldb,
                                          float* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_dgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          double* B,
                                          const rocblas_int ldb,
                                          double* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_cgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_zgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_sgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          float* B,
                                          const rocblas_int ldb,
                                          float* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_dgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          double* B,
                                          const rocblas_int ldb,
                                          double* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_cgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_zgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

#ifdef __cplusplus
}
#endif
/***************************************************/

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

/******************** LATRD ********************/
inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* A,
                                      rocblas_int lda,
                                      float* E,
                                      float* tau,
                                      float* W,
                                      rocblas_int ldw)
{
    return rocsolver_slatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* A,
                                      rocblas_int lda,
                                      double* E,
                                      double* tau,
                                      double* W,
                                      rocblas_int ldw)
{
    return rocsolver_dlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      float* E,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* W,
                                      rocblas_int ldw)
{
    return rocsolver_clatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      double* E,
                                      rocblas_double_complex* tau,
                                      rocblas_double_complex* W,
                                      rocblas_int ldw)
{
    return rocsolver_zlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
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

/******************** STERF ********************/
inline rocblas_status
    rocsolver_sterf(rocblas_handle handle, rocblas_int n, float* D, float* E, rocblas_int* info)
{
    return rocsolver_ssterf(handle, n, D, E, info);
}

inline rocblas_status
    rocsolver_sterf(rocblas_handle handle, rocblas_int n, double* D, double* E, rocblas_int* info)
{
    return rocsolver_dsterf(handle, n, D, E, info);
}
/********************************************************/

/******************** STEBZ ********************/
inline rocblas_status rocsolver_stebz(rocblas_handle handle,
                                      rocblas_erange range,
                                      rocblas_eorder order,
                                      rocblas_int n,
                                      float vl,
                                      float vu,
                                      rocblas_int il,
                                      rocblas_int iu,
                                      float abstol,
                                      float* D,
                                      float* E,
                                      rocblas_int* nev,
                                      rocblas_int* nsplit,
                                      float* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_int* info)
{
    return rocsolver_sstebz(handle, range, order, n, vl, vu, il, iu, abstol, D, E, nev, nsplit, W,
                            iblock, isplit, info);
}

inline rocblas_status rocsolver_stebz(rocblas_handle handle,
                                      rocblas_erange range,
                                      rocblas_eorder order,
                                      rocblas_int n,
                                      double vl,
                                      double vu,
                                      rocblas_int il,
                                      rocblas_int iu,
                                      double abstol,
                                      double* D,
                                      double* E,
                                      rocblas_int* nev,
                                      rocblas_int* nsplit,
                                      double* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_int* info)
{
    return rocsolver_dstebz(handle, range, order, n, vl, vu, il, iu, abstol, D, E, nev, nsplit, W,
                            iblock, isplit, info);
}
/********************************************************/

/******************** STEQR ********************/
inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_ssteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dsteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_csteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zsteqr(handle, evect, n, D, E, C, ldc, info);
}
/********************************************************/

/******************** STEDC ********************/
inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_sstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_cstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zstedc(handle, evect, n, D, E, C, ldc, info);
}
/********************************************************/

/******************** LASYF ********************/
inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_slasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_dlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_clasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_zlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}
/********************************************************/

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

/******************** POTRS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_spotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_spotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_dpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_dpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_cpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_cpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_zpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_zpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

// batched
inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_spotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_dpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_cpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_zpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}
/********************************************************/

/******************** POSV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_sposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_sposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_dposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_dposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_cposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_cposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_zposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_zposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_sposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_dposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_cposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_zposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}
/********************************************************/

/******************** POTRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potri(bool STRIDED,
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
        return rocsolver_spotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_spotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
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
        return rocsolver_dpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_dpotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
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
        return rocsolver_cpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_cpotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
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
        return rocsolver_zpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_zpotri(handle, uplo, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_spotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_dpotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_cpotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_zpotri_batched(handle, uplo, n, A, lda, info, batch_count);
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

/******************** GESV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_sgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_sgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_dgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_dgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_cgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_cgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_zgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_zgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_sgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_dgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_cgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
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
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_zgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}
/********************************************************/

/******************** GESV_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                float* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_sgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_sgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                double* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_dgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_dgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                rocblas_float_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_cgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_cgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                rocblas_double_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_zgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_zgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

// batched
inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                float* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                double* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                rocblas_float_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
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
                                                rocblas_double_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}
/********************************************************/

/******************** GETRI_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 float* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 float* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_sgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 double* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 double* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_dgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_float_complex* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_cgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_double_complex* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_zgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

// batched
inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 float* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 float* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_sgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 double* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 double* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_dgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_float_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_float_complex* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_cgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_double_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_double_complex* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_zgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}
/********************************************************/

/******************** GETRI_NPVT_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      float* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_sgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      double* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_dgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_cgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_zgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

// batched
inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      float* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_sgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      double* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_dgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_float_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_cgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_double_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_zgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}
/********************************************************/

/******************** GETRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
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

// batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_sgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_cgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_zgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/

/******************** GETRI_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           float* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_sgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           double* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_dgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_float_complex* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_cgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_double_complex* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_zgetri_npvt(handle, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           float* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_sgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           double* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_dgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_float_complex* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_cgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_double_complex* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_zgetri_npvt_batched(handle, n, A, lda, info, bc);
}
/********************************************************/

/******************** TRTRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_strtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_strtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dtrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_dtrtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_ctrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_ctrtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_ztrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_ztrtri(handle, uplo, diag, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_strtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dtrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_ctrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_ztrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
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

/******************** GERQ2_GERQF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
        return GERQF ? rocsolver_sgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_sgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
        return GERQF ? rocsolver_dgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_dgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
        return GERQF ? rocsolver_cgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_cgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
        return GERQF ? rocsolver_zgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_zgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgerq2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
    return GERQF ? rocsolver_sgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
    return GERQF ? rocsolver_dgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
    return GERQF ? rocsolver_cgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
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
    return GERQF ? rocsolver_zgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
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

/******************** GELS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_sgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_sgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_dgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_cgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_cgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_zgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_sgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_dgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_cgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_zgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}
/********************************************************/

/******************** GELS_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_sgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_sgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_dgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_dgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_float_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_cgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_cgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_double_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_zgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_zgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

// batched
inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_float_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_double_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
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

/******************** SYTD2/SYTRD_HETD2/HETRD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_ssytrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_ssytd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_ssytrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_ssytd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_dsytrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_dsytd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_dsytrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_dsytd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_chetrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_chetd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_chetrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_chetd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_zhetrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_zhetd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_zhetrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_zhetd2(handle, uplo, n, A, lda, D, E, tau);
}

// batched
inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_ssytrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_ssytd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_dsytrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_dsytd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_chetrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_chetd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_zhetrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_zhetd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}
/********************************************************/

/******************** SYGS2/SYGST_HEGS2/HEGST ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_ssygst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_ssygs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_ssygst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_ssygs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_dsygst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_dsygs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_dsygst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_dsygs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_chegst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_chegs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_chegst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_chegs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_zhegst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_zhegs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_zhegst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_zhegs2(handle, itype, uplo, n, A, lda, B, ldb);
}

// batched
inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_ssygst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_ssygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_dsygst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_dsygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_chegst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_chegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_zhegst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_zhegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}
/********************************************************/

/******************** SYEV/HEEV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_ssyev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_ssyev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_dsyev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_dsyev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_cheev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_cheev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_zheev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_zheev(handle, evect, uplo, n, A, lda, D, E, info);
}

// batched
inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_ssyev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_dsyev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_cheev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_zheev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}
/********************************************************/

/******************** SYEVD/HEEVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_ssyevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_ssyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_dsyevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_dsyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_cheevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_cheevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_zheevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_zheevd(handle, evect, uplo, n, A, lda, D, E, info);
}

// batched
inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssyevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsyevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_cheevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zheevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}
/********************************************************/

/******************** SYGV_HEGV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_ssygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_dsygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_float_complex* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_chegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_double_complex* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_zhegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

// batched
inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_ssygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_dsygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_float_complex* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_chegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_double_complex* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_zhegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}
/********************************************************/

/******************** SYGVD_HEGVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_ssygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_dsygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_chegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_zhegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

// batched
inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_chegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zhegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}
/********************************************************/

/******************** SYTF2_SYTRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
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
        return SYTRF
            ? rocsolver_ssytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_ssytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_ssytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_ssytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
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
        return SYTRF
            ? rocsolver_dsytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_dsytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_dsytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_dsytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
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
        return SYTRF
            ? rocsolver_csytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_csytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_csytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_csytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
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
        return SYTRF
            ? rocsolver_zsytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_zsytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_zsytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_zsytf2(handle, uplo, n, A, lda, ipiv, info);
}

// batched
inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_ssytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_ssytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_dsytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_dsytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_csytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_csytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_zsytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_zsytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/
