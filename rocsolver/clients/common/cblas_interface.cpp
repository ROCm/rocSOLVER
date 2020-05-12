/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include "cblas_interface.h"
#include "cblas.h"
#include "rocblas.h"
#include "utility.h"
#include <memory>
#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is
 * only used for testing not part of the GPU library
 */

#ifdef __cplusplus
extern "C" {
#endif

void strtri_(char *uplo, char *diag, int *n, float *A, int *lda, int *info);
void dtrtri_(char *uplo, char *diag, int *n, double *A, int *lda, int *info);
void ctrtri_(char *uplo, char *diag, int *n, rocblas_float_complex *A, int *lda,
             int *info);
void ztrtri_(char *uplo, char *diag, int *n, rocblas_double_complex *A,
             int *lda, int *info);

void sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);
void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
void cgetrf_(int *m, int *n, rocblas_float_complex *A, int *lda, int *ipiv,
             int *info);
void zgetrf_(int *m, int *n, rocblas_double_complex *A, int *lda, int *ipiv,
             int *info);

void spotrf_(char *uplo, int *m, float *A, int *lda, int *info);
void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
void cpotrf_(char *uplo, int *m, rocblas_float_complex *A, int *lda, int *info);
void zpotrf_(char *uplo, int *m, rocblas_double_complex *A, int *lda, int *info);

void spotf2_(char *uplo, int *n, float *A, int *lda, int *info);
void dpotf2_(char *uplo, int *n, double *A, int *lda, int *info);
void cpotf2_(char *uplo, int *n, rocblas_float_complex *A, int *lda, int *info);
void zpotf2_(char *uplo, int *n, rocblas_double_complex *A, int *lda, int *info);

void sgetf2_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);
void dgetf2_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
void cgetf2_(int *m, int *n, rocblas_float_complex *A, int *lda, int *ipiv,
             int *info);
void zgetf2_(int *m, int *n, rocblas_double_complex *A, int *lda, int *ipiv,
             int *info);

void sgetrs_(char *trans, int *n, int *nrhs, float *A, int *lda, int *ipiv,
             float *B, int *ldb, int *info);
void dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda, int *ipiv,
             double *B, int *ldb, int *info);
void cgetrs_(char *trans, int *n, int *nrhs, rocblas_float_complex *A, int *lda,
             int *ipiv, rocblas_float_complex *B, int *ldb, int *info);
void zgetrs_(char *trans, int *n, int *nrhs, rocblas_double_complex *A,
             int *lda, int *ipiv, rocblas_double_complex *B, int *ldb,
             int *info);

void slarfg_(int *n, float *alpha, float *x, int *incx, float *tau);
void dlarfg_(int *n, double *alpha, double *x, int *incx, double *tau);
void clarfg_(int *n, rocblas_float_complex *alpha, rocblas_float_complex *x, int *incx, rocblas_float_complex *tau);
void zlarfg_(int *n, rocblas_double_complex *alpha, rocblas_double_complex *x, int *incx, rocblas_double_complex *tau);

void slarf_(char *side, int *m, int *n, float *x, int *incx, float *alpha, float *A, int *lda, float *work);
void dlarf_(char *side, int *m, int *n, double *x, int *incx, double *alpha, double *A, int *lda, double *work);
void clarf_(char *side, int *m, int *n, rocblas_float_complex *x, int *incx, rocblas_float_complex *alpha, rocblas_float_complex *A, int *lda, rocblas_float_complex *work);
void zlarf_(char *side, int *m, int *n, rocblas_double_complex *x, int *incx, rocblas_double_complex *alpha, rocblas_double_complex *A, int *lda, rocblas_double_complex *work);

void slarft_(char *direct, char *storev, int *n, int *k, float *V, int *ldv, float *tau, float *T, int *ldt);
void dlarft_(char *direct, char *storev, int *n, int *k, double *V, int *ldv, double *tau, double *T, int *ldt);
void clarft_(char *direct, char *storev, int *n, int *k, rocblas_float_complex *V, int *ldv, rocblas_float_complex *tau, rocblas_float_complex *T, int *ldt);
void zlarft_(char *direct, char *storev, int *n, int *k, rocblas_double_complex *V, int *ldv, rocblas_double_complex *tau, rocblas_double_complex *T, int *ldt);

void slarfb_(char *side, char *trans, char *direct, char *storev, int *m, int *n, int *k, float *V, int *ldv, float *T, int *ldt, float *A, int *lda, float *W, int *ldw);
void dlarfb_(char *side, char *trans, char *direct, char *storev, int *m, int *n, int *k, double *V, int *ldv, double *T, int *ldt, double *A, int *lda, double *W, int *ldw);
void clarfb_(char *side, char *trans, char *direct, char *storev, int *m, int *n, int *k, rocblas_float_complex *V, int *ldv, rocblas_float_complex *T, int *ldt, rocblas_float_complex *A, int *lda, rocblas_float_complex *W, int *ldw);
void zlarfb_(char *side, char *trans, char *direct, char *storev, int *m, int *n, int *k, rocblas_double_complex *V, int *ldv, rocblas_double_complex *T, int *ldt, rocblas_double_complex *A, int *lda, rocblas_double_complex *W, int *ldw);

void sgeqr2_(int *m, int *n, float *A, int *lda, float *ipiv, float *work, int *info);
void dgeqr2_(int *m, int *n, double *A, int *lda, double *ipiv, double *work, int *info);
void cgeqr2_(int *m, int *n, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *info);
void zgeqr2_(int *m, int *n, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *info);
void sgeqrf_(int *m, int *n, float *A, int *lda, float *ipiv, float *work, int *lwork, int *info);
void dgeqrf_(int *m, int *n, double *A, int *lda, double *ipiv, double *work, int *lwork, int *info);
void cgeqrf_(int *m, int *n, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *lwork, int *info);
void zgeqrf_(int *m, int *n, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *lwork, int *info);

void sgelq2_(int *m, int *n, float *A, int *lda, float *ipiv, float *work, int *info);
void dgelq2_(int *m, int *n, double *A, int *lda, double *ipiv, double *work, int *info);
void cgelq2_(int *m, int *n, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *info);
void zgelq2_(int *m, int *n, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *info);
void sgelqf_(int *m, int *n, float *A, int *lda, float *ipiv, float *work, int *lwork, int *info);
void dgelqf_(int *m, int *n, double *A, int *lda, double *ipiv, double *work, int *lwork, int *info);
void cgelqf_(int *m, int *n, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *lwork, int *info);
void zgelqf_(int *m, int *n, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *lwork, int *info);

void clacgv_(int *n, rocblas_float_complex *x, int *incx);
void zlacgv_(int *n, rocblas_double_complex *x, int *incx);

void slaswp_(int *n, float *A, int *lda, int *k1, int *k2, int *ipiv, int *inc);
void dlaswp_(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *inc);
void claswp_(int *n, rocblas_float_complex *A, int *lda, int *k1, int *k2, int *ipiv, int *inc);
void zlaswp_(int *n, rocblas_double_complex *A, int *lda, int *k1, int *k2, int *ipiv, int *inc);

void sorg2r_(int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *work, int *info);
void dorg2r_(int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *work, int *info);
void cung2r_(int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *info);
void zung2r_(int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *info);
void sorgqr_(int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *work, int *lwork, int *info);
void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *work, int *lwork, int *info);
void cungqr_(int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *lwork, int *info);
void zungqr_(int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *lwork, int *info);

void sorm2r_(char *side, char *trans, int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *C, int *ldc, float *work, int *info);
void dorm2r_(char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *C, int *ldc, double *work, int *info);
void cunm2r_(char *side, char *trans, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, int *ldc, rocblas_float_complex *work, int *info);
void zunm2r_(char *side, char *trans, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, int *ldc, rocblas_double_complex *work, int *info);
void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *C, int *ldc, float *work, int *sizeW, int *info);
void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *C, int *ldc, double *work, int *sizeW, int *info);
void cunmqr_(char *side, char *trans, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, int *ldc, rocblas_float_complex *work, int *sizeW, int *info);
void zunmqr_(char *side, char *trans, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, int *ldc, rocblas_double_complex *work, int *sizeW, int *info);

void sorml2_(char *side, char *trans, int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *C, int *ldc, float *work, int *info);
void dorml2_(char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *C, int *ldc, double *work, int *info);
void cunml2_(char *side, char *trans, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, int *ldc, rocblas_float_complex *work, int *info);
void zunml2_(char *side, char *trans, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, int *ldc, rocblas_double_complex *work, int *info);
void sormlq_(char *side, char *trans, int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *C, int *ldc, float *work, int *sizeW, int *info);
void dormlq_(char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *C, int *ldc, double *work, int *sizeW, int *info);
void cunmlq_(char *side, char *trans, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, int *ldc, rocblas_float_complex *work, int *sizeW, int *info);
void zunmlq_(char *side, char *trans, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, int *ldc, rocblas_double_complex *work, int *sizeW, int *info);

void sorgl2_(int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *work, int *info);
void dorgl2_(int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *work, int *info);
void cungl2_(int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *info);
void zungl2_(int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *info);
void sorglq_(int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *work, int *lwork, int *info);
void dorglq_(int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *work, int *lwork, int *info);
void cunglq_(int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, int *lwork, int *info);
void zunglq_(int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, int *lwork, int *info);

void sorgbr_(char *vect, int *m, int *n, int *k, float *A, int *lda, float *Ipiv, float *work, int *size_w, int *info);
void dorgbr_(char *vect, int *m, int *n, int *k, double *A, int *lda, double *Ipiv, double *work, int *size_w, int *info);
void cungbr_(char *vect, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *Ipiv, rocblas_float_complex *work, int *size_w, int *info);
void zungbr_(char *vect, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *Ipiv, rocblas_double_complex *work, int *size_w, int *info);
void sormbr_(char *vect, char *side, char *trans, int *m, int *n, int *k, float *A, int *lda, float *ipiv, float *C, int *ldc, float *work, int *sizeW, int *info);
void dormbr_(char *vect, char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *ipiv, double *C, int *ldc, double *work, int *sizeW, int *info);
void cunmbr_(char *vect, char *side, char *trans, int *m, int *n, int *k, rocblas_float_complex *A, int *lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, int *ldc, rocblas_float_complex *work, int *sizeW, int *info);
void zunmbr_(char *vect, char *side, char *trans, int *m, int *n, int *k, rocblas_double_complex *A, int *lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, int *ldc, rocblas_double_complex *work, int *sizeW, int *info);

void sgebd2_(int *m, int *n, float *A, int *lda, float *D, float *E, float *tauq, float *taup, float *work, int *info);
void dgebd2_(int *m, int *n, double *A, int *lda, double *D, double *E, double *tauq, double *taup, double *work, int *info);
void cgebd2_(int *m, int *n, rocblas_float_complex *A, int *lda, float *D, float *E, rocblas_float_complex *tauq, rocblas_float_complex *taup, rocblas_float_complex *work, int *info);
void zgebd2_(int *m, int *n, rocblas_double_complex *A, int *lda, double *D, double *E, rocblas_double_complex *tauq, rocblas_double_complex *taup, rocblas_double_complex *work, int *info);

void sgebrd_(int *m, int *n, float *A, int *lda, float *D, float *E, float *tauq, float *taup, float *work, int *size_w, int *info);
void dgebrd_(int *m, int *n, double *A, int *lda, double *D, double *E, double *tauq, double *taup, double *work, int *size_w, int *info);
void cgebrd_(int *m, int *n, rocblas_float_complex *A, int *lda, rocblas_float_complex *D, rocblas_float_complex *E, rocblas_float_complex *tauq, rocblas_float_complex *taup, rocblas_float_complex *work, int *size_w, int *info);
void zgebrd_(int *m, int *n, rocblas_double_complex *A, int *lda, rocblas_double_complex *D, rocblas_double_complex *E, rocblas_double_complex *tauq, rocblas_double_complex *taup, rocblas_double_complex *work, int *size_w, int *info);



#ifdef __cplusplus
}
#endif

/*
 * ===========================================================================
 *    Auxiliary LAPACK
 * ===========================================================================
 */

//lacgv

template <>
void cblas_lacgv<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *x, rocblas_int incx) {
    clacgv_(&n,x,&incx);
}

template <>
void cblas_lacgv<rocblas_double_complex>(rocblas_int n, rocblas_double_complex *x, rocblas_int incx) {
    zlacgv_(&n,x,&incx);
}


//laswp

template <>
void cblas_laswp<float>(rocblas_int n, float *A, rocblas_int lda, rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
    slaswp_(&n,A,&lda,&k1,&k2,ipiv,&inc);
}

template <>
void cblas_laswp<double>(rocblas_int n, double *A, rocblas_int lda, rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
    dlaswp_(&n,A,&lda,&k1,&k2,ipiv,&inc);
}

template <>
void cblas_laswp<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
    claswp_(&n,A,&lda,&k1,&k2,ipiv,&inc);
}

template <>
void cblas_laswp<rocblas_double_complex>(rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_int k1, rocblas_int k2, rocblas_int *ipiv, rocblas_int inc) {
    zlaswp_(&n,A,&lda,&k1,&k2,ipiv,&inc);
}


//larfg

template <>
void cblas_larfg<float>(rocblas_int n, float *alpha, float *x, rocblas_int incx, float *tau) {
    slarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<double>(rocblas_int n, double *alpha, double *x, rocblas_int incx, double *tau) {
    dlarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *alpha, rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *tau) {
    clarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_double_complex>(rocblas_int n, rocblas_double_complex *alpha, rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex *tau) {
    zlarfg_(&n, alpha, x, &incx, tau);
}

//larf

template <>
void cblas_larf<float>(char side, rocblas_int m, rocblas_int n, float *x, rocblas_int incx, float *alpha, float *A, rocblas_int lda, float *work) {
    slarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<double>(char side, rocblas_int m, rocblas_int n, double *x, rocblas_int incx, double *alpha, double *A, rocblas_int lda, double *work) {
    dlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_float_complex>(char side, rocblas_int m, rocblas_int n, rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *alpha, rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *work) {
    clarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_double_complex>(char side, rocblas_int m, rocblas_int n, rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex *alpha, rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *work) {
    zlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

//larft

template <>
void cblas_larft<float>(char direct, char storev, rocblas_int n, rocblas_int k, float *V, rocblas_int ldv, float *tau, float *T, rocblas_int ldt) {
    slarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<double>(char direct, char storev, rocblas_int n, rocblas_int k, double *V, rocblas_int ldv, double *tau, double *T, rocblas_int ldt) {
    dlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_float_complex>(char direct, char storev, rocblas_int n, rocblas_int k, rocblas_float_complex *V, rocblas_int ldv, rocblas_float_complex *tau, rocblas_float_complex *T, rocblas_int ldt) {
    clarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_double_complex>(char direct, char storev, rocblas_int n, rocblas_int k, rocblas_double_complex *V, rocblas_int ldv, rocblas_double_complex *tau, rocblas_double_complex *T, rocblas_int ldt) {
    zlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

//larfb

template <>
void cblas_larfb<float>(char side, char trans, char direct, char storev, rocblas_int m, rocblas_int n, rocblas_int k, 
                        float *V, rocblas_int ldv, float *T, rocblas_int ldt, float *A, rocblas_int lda, float *W, rocblas_int ldw) {
    slarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<double>(char side, char trans, char direct, char storev, rocblas_int m, rocblas_int n, rocblas_int k, 
                         double *V, rocblas_int ldv, double *T, rocblas_int ldt, double *A, rocblas_int lda, double *W, rocblas_int ldw) {
    dlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_float_complex>(char side, char trans, char direct, char storev, rocblas_int m, rocblas_int n, rocblas_int k, 
                        rocblas_float_complex *V, rocblas_int ldv, rocblas_float_complex *T, rocblas_int ldt, rocblas_float_complex *A, rocblas_int lda,
                        rocblas_float_complex *W, rocblas_int ldw) {
    clarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_double_complex>(char side, char trans, char direct, char storev, rocblas_int m, rocblas_int n, rocblas_int k, 
                         rocblas_double_complex *V, rocblas_int ldv, rocblas_double_complex *T, rocblas_int ldt, rocblas_double_complex *A, rocblas_int lda,
                         rocblas_double_complex *W, rocblas_int ldw) {
    zlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

// orgqr & ungqr
template <>
void cblas_orgqr_ungqr<float>(rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  int lwork = n;
  sorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<double>(rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  int lwork = n;
  dorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  int lwork = n;
  cungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  int lwork = n;
  zungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2r & ung2r
template <>
void cblas_org2r_ung2r<float>(rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  sorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<double>(rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  dorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  cung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  zung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// ormqr & unmqr
template <>
void cblas_ormqr_unmqr<float>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *C, rocblas_int ldc, float *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  sormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<double>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *C, rocblas_int ldc, double *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  dormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_float_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, rocblas_int ldc, rocblas_float_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  cunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_double_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, rocblas_int ldc, rocblas_double_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  zunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2r & unm2r
template <>
void cblas_orm2r_unm2r<float>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *C, rocblas_int ldc, float *work) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  sorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<double>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *C, rocblas_int ldc, double *work) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  dorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_float_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, rocblas_int ldc, rocblas_float_complex *work) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  cunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_double_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, rocblas_int ldc, rocblas_double_complex *work) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  zunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormlq & unmlq
template <>
void cblas_ormlq_unmlq<float>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *C, rocblas_int ldc, float *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  sormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<double>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *C, rocblas_int ldc, double *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  dormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_float_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, rocblas_int ldc, rocblas_float_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  cunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_double_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, rocblas_int ldc, rocblas_double_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  zunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orml2 & unml2
template <>
void cblas_orml2_unml2<float>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *C, rocblas_int ldc, float *work) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  sorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<double>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *C, rocblas_int ldc, double *work) {
  int info;
  char sideC = 'R', transC = 'T';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  dorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_float_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, rocblas_int ldc, rocblas_float_complex *work) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  cunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_double_complex>(rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, rocblas_int ldc, rocblas_double_complex *work) {
  int info;
  char sideC = 'R', transC = 'C';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  zunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// orglq & unglq
template <>
void cblas_orglq_unglq<float>(rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  int lwork = m;
  sorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<double>(rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  int lwork = m;
  dorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  int lwork = m;
  cunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  int lwork = m;
  zunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// orgl2 & ungl2
template <>
void cblas_orgl2_ungl2<float>(rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  sorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<double>(rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  dorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  cungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  zungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// ormbr & unmbr
template <>
void cblas_ormbr_unmbr<float>(char storev, rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, float *A,
                               rocblas_int lda, float *ipiv, float *C, rocblas_int ldc, float *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  char vect;
  if (storev == 'C')
    vect = 'Q';
  else
    vect = 'P';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  sormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<double>(char storev, rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, double *A,
                               rocblas_int lda, double *ipiv, double *C, rocblas_int ldc, double *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'T';
  char vect;
  if (storev == 'C')
    vect = 'Q';
  else
    vect = 'P';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  dormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_float_complex>(char storev, rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *C, rocblas_int ldc, rocblas_float_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  char vect;
  if (storev == 'C')
    vect = 'Q';
  else
    vect = 'P';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  cunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_double_complex>(char storev, rocblas_side side, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *C, rocblas_int ldc, rocblas_double_complex *work, rocblas_int lwork) {
  int info;
  char sideC = 'R', transC = 'C';
  char vect;
  if (storev == 'C')
    vect = 'Q';
  else
    vect = 'P';
  if (side == rocblas_side_left)
    sideC = 'L';
  if (trans == rocblas_operation_none)
    transC = 'N';

  zunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}



/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// scal
template <>
void cblas_scal<float>(rocblas_int n, const float alpha, float *x,
                       rocblas_int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template <>
void cblas_scal<double>(rocblas_int n, const double alpha, double *x,
                        rocblas_int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template <>
void cblas_scal<rocblas_float_complex>(rocblas_int n,
                                       const rocblas_float_complex alpha,
                                       rocblas_float_complex *x,
                                       rocblas_int incx) {
  cblas_cscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<rocblas_double_complex>(rocblas_int n,
                                        const rocblas_double_complex alpha,
                                        rocblas_double_complex *x,
                                        rocblas_int incx) {
  cblas_zscal(n, &alpha, x, incx);
}

// copy
template <>
void cblas_copy<float>(rocblas_int n, float *x, rocblas_int incx, float *y,
                       rocblas_int incy) {
  cblas_scopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<double>(rocblas_int n, double *x, rocblas_int incx, double *y,
                        rocblas_int incy) {
  cblas_dcopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *x,
                                       rocblas_int incx,
                                       rocblas_float_complex *y,
                                       rocblas_int incy) {
  cblas_ccopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex *x,
                                        rocblas_int incx,
                                        rocblas_double_complex *y,
                                        rocblas_int incy) {
  cblas_zcopy(n, x, incx, y, incy);
}

// axpy
/*template <>
void cblas_axpy<rocblas_half>(rocblas_int n, rocblas_half alpha,
                              rocblas_half *x, rocblas_int incx,
                              rocblas_half *y, rocblas_int incy) {
  rocblas_int abs_incx = incx >= 0 ? incx : -incx;
  rocblas_int abs_incy = incy >= 0 ? incy : -incy;
  float alpha_float = half_to_float(alpha);
  std::unique_ptr<float[]> x_float(new float[n * abs_incx]());
  std::unique_ptr<float[]> y_float(new float[n * abs_incy]());
  for (int i = 0; i < n; i++) {
    x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
    y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
  }

  cblas_saxpy(n, alpha_float, x_float.get(), incx, y_float.get(), incy);

  for (int i = 0; i < n; i++) {
    x[i * abs_incx] = float_to_half(x_float[i * abs_incx]);
    y[i * abs_incy] = float_to_half(y_float[i * abs_incy]);
  }
}*/

template <>
void cblas_axpy<float>(rocblas_int n, float alpha, float *x, rocblas_int incx,
                       float *y, rocblas_int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<double>(rocblas_int n, double alpha, double *x,
                        rocblas_int incx, double *y, rocblas_int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_float_complex>(
    rocblas_int n, rocblas_float_complex alpha, rocblas_float_complex *x,
    rocblas_int incx, rocblas_float_complex *y, rocblas_int incy) {
  cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_double_complex>(
    rocblas_int n, rocblas_double_complex alpha, rocblas_double_complex *x,
    rocblas_int incx, rocblas_double_complex *y, rocblas_int incy) {
  cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// swap
template <>
void cblas_swap<float>(rocblas_int n, float *x, rocblas_int incx, float *y,
                       rocblas_int incy) {
  cblas_sswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<double>(rocblas_int n, double *x, rocblas_int incx, double *y,
                        rocblas_int incy) {
  cblas_dswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *x,
                                       rocblas_int incx,
                                       rocblas_float_complex *y,
                                       rocblas_int incy) {
  cblas_cswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex *x,
                                        rocblas_int incx,
                                        rocblas_double_complex *y,
                                        rocblas_int incy) {
  cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void cblas_dot<float>(rocblas_int n, const float *x, rocblas_int incx,
                      const float *y, rocblas_int incy, float *result) {
  *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
void cblas_dot<double>(rocblas_int n, const double *x, rocblas_int incx,
                       const double *y, rocblas_int incy, double *result) {
  *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
void cblas_dot<rocblas_float_complex>(rocblas_int n,
                                      const rocblas_float_complex *x,
                                      rocblas_int incx,
                                      const rocblas_float_complex *y,
                                      rocblas_int incy,
                                      rocblas_float_complex *result) {
  cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dot<rocblas_double_complex>(rocblas_int n,
                                       const rocblas_double_complex *x,
                                       rocblas_int incx,
                                       const rocblas_double_complex *y,
                                       rocblas_int incy,
                                       rocblas_double_complex *result) {
  cblas_zdotu_sub(n, x, incx, y, incy, result);
}

// nrm2
template <>
void cblas_nrm2<float, float>(rocblas_int n, const float *x, rocblas_int incx,
                              float *result) {
  *result = cblas_snrm2(n, x, incx);
}

template <>
void cblas_nrm2<double, double>(rocblas_int n, const double *x,
                                rocblas_int incx, double *result) {
  *result = cblas_dnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex *x,
                                              rocblas_int incx, float *result) {
  *result = cblas_scnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex *x,
                                                rocblas_int incx,
                                                double *result) {
  *result = cblas_dznrm2(n, x, incx);
}

// asum
template <>
void cblas_asum<float, float>(rocblas_int n, const float *x, rocblas_int incx,
                              float *result) {
  *result = cblas_sasum(n, x, incx);
}

template <>
void cblas_asum<double, double>(rocblas_int n, const double *x,
                                rocblas_int incx, double *result) {
  *result = cblas_dasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex *x,
                                              rocblas_int incx, float *result) {
  *result = cblas_scasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex *x,
                                                rocblas_int incx,
                                                double *result) {
  *result = cblas_dzasum(n, x, incx);
}

// amax
template <>
void cblas_iamax<float>(rocblas_int n, const float *x, rocblas_int incx,
                        rocblas_int *result) {
  *result = (rocblas_int)cblas_isamax(n, x, incx);
}

template <>
void cblas_iamax<double>(rocblas_int n, const double *x, rocblas_int incx,
                         rocblas_int *result) {
  *result = (rocblas_int)cblas_idamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_float_complex>(rocblas_int n,
                                        const rocblas_float_complex *x,
                                        rocblas_int incx, rocblas_int *result) {
  *result = (rocblas_int)cblas_icamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_double_complex>(rocblas_int n,
                                         const rocblas_double_complex *x,
                                         rocblas_int incx,
                                         rocblas_int *result) {
  *result = (rocblas_int)cblas_izamax(n, x, incx);
}
/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

template <>
void cblas_gemv<float>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                       float alpha, float *A, rocblas_int lda, float *x,
                       rocblas_int incx, float beta, float *y,
                       rocblas_int incy) {
  cblas_sgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x,
              incx, beta, y, incy);
}

template <>
void cblas_gemv<double>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                        double alpha, double *A, rocblas_int lda, double *x,
                        rocblas_int incx, double beta, double *y,
                        rocblas_int incy) {
  cblas_dgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x,
              incx, beta, y, incy);
}

template <>
void cblas_gemv<rocblas_float_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, rocblas_float_complex *A, rocblas_int lda,
    rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex beta,
    rocblas_float_complex *y, rocblas_int incy) {
  cblas_cgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x,
              incx, &beta, y, incy);
}

template <>
void cblas_gemv<rocblas_double_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, rocblas_double_complex *A, rocblas_int lda,
    rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex beta,
    rocblas_double_complex *y, rocblas_int incy) {
  cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x,
              incx, &beta, y, incy);
}

template <>
void cblas_symv<float>(rocblas_fill uplo, rocblas_int n, float alpha, float *A,
                       rocblas_int lda, float *x, rocblas_int incx, float beta,
                       float *y, rocblas_int incy) {
  cblas_ssymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta,
              y, incy);
}

template <>
void cblas_symv<double>(rocblas_fill uplo, rocblas_int n, double alpha,
                        double *A, rocblas_int lda, double *x, rocblas_int incx,
                        double beta, double *y, rocblas_int incy) {
  cblas_dsymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta,
              y, incy);
}

template <>
void cblas_hemv<rocblas_float_complex>(
    rocblas_fill uplo, rocblas_int n, rocblas_float_complex alpha,
    rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *x,
    rocblas_int incx, rocblas_float_complex beta, rocblas_float_complex *y,
    rocblas_int incy) {
  cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx,
              &beta, y, incy);
}

template <>
void cblas_hemv<rocblas_double_complex>(
    rocblas_fill uplo, rocblas_int n, rocblas_double_complex alpha,
    rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *x,
    rocblas_int incx, rocblas_double_complex beta, rocblas_double_complex *y,
    rocblas_int incy) {
  cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx,
              &beta, y, incy);
}

template <>
void cblas_ger<float>(rocblas_int m, rocblas_int n, float alpha, float *x,
                      rocblas_int incx, float *y, rocblas_int incy, float *A,
                      rocblas_int lda) {
  cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<double>(rocblas_int m, rocblas_int n, double alpha, double *x,
                       rocblas_int incx, double *y, rocblas_int incy, double *A,
                       rocblas_int lda) {
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr<float>(rocblas_fill uplo, rocblas_int n, float alpha, float *x,
                      rocblas_int incx, float *A, rocblas_int lda) {
  cblas_ssyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void cblas_syr<double>(rocblas_fill uplo, rocblas_int n, double alpha,
                       double *x, rocblas_int incx, double *A,
                       rocblas_int lda) {
  cblas_dsyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */
// gemm
/*template <>
void cblas_gemm<rocblas_half>(rocblas_operation transA,
                              rocblas_operation transB, rocblas_int m,
                              rocblas_int n, rocblas_int k, rocblas_half alpha,
                              rocblas_half *A, rocblas_int lda, rocblas_half *B,
                              rocblas_int ldb, rocblas_half beta,
                              rocblas_half *C, rocblas_int ldc) {
  // cblas does not support rocblas_half, so convert to higher precision float
  // This will give more precise result which is acceptable for testing
  float alpha_float = half_to_float(alpha);
  float beta_float = half_to_float(beta);

  int sizeA = transA == rocblas_operation_none ? k * lda : m * lda;
  int sizeB = transB == rocblas_operation_none ? n * ldb : k * ldb;
  int sizeC = n * ldc;

  std::unique_ptr<float[]> A_float(new float[sizeA]());
  std::unique_ptr<float[]> B_float(new float[sizeB]());
  std::unique_ptr<float[]> C_float(new float[sizeC]());

  for (int i = 0; i < sizeA; i++) {
    A_float[i] = half_to_float(A[i]);
  }
  for (int i = 0; i < sizeB; i++) {
    B_float[i] = half_to_float(B[i]);
  }
  for (int i = 0; i < sizeC; i++) {
    C_float[i] = half_to_float(C[i]);
  }

  // just directly cast, since transA, transB are integers in the enum
  // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
  // );
  cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, alpha_float, const_cast<const float *>(A_float.get()),
              lda, const_cast<const float *>(B_float.get()), ldb, beta_float,
              static_cast<float *>(C_float.get()), ldc);

  for (int i = 0; i < sizeC; i++) {
    C[i] = float_to_half(C_float[i]);
  }
}*/

template <>
void cblas_gemm<float>(rocblas_operation transA, rocblas_operation transB,
                       rocblas_int m, rocblas_int n, rocblas_int k, float alpha,
                       float *A, rocblas_int lda, float *B, rocblas_int ldb,
                       float beta, float *C, rocblas_int ldc) {
  // just directly cast, since transA, transB are integers in the enum
  // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
  // );
  cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<double>(rocblas_operation transA, rocblas_operation transB,
                        rocblas_int m, rocblas_int n, rocblas_int k,
                        double alpha, double *A, rocblas_int lda, double *B,
                        rocblas_int ldb, double beta, double *C,
                        rocblas_int ldc) {
  cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_float_complex>(
    rocblas_operation transA, rocblas_operation transB, rocblas_int m,
    rocblas_int n, rocblas_int k, rocblas_float_complex alpha,
    rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *B,
    rocblas_int ldb, rocblas_float_complex beta, rocblas_float_complex *C,
    rocblas_int ldc) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_cgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_double_complex>(
    rocblas_operation transA, rocblas_operation transB, rocblas_int m,
    rocblas_int n, rocblas_int k, rocblas_double_complex alpha,
    rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *B,
    rocblas_int ldb, rocblas_double_complex beta, rocblas_double_complex *C,
    rocblas_int ldc) {
  cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// trsm
template <>
void cblas_trsm<float>(rocblas_side side, rocblas_fill uplo,
                       rocblas_operation transA, rocblas_diagonal diag,
                       rocblas_int m, rocblas_int n, float alpha,
                       const float *A, rocblas_int lda, float *B,
                       rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_strsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trsm<double>(rocblas_side side, rocblas_fill uplo,
                        rocblas_operation transA, rocblas_diagonal diag,
                        rocblas_int m, rocblas_int n, double alpha,
                        const double *A, rocblas_int lda, double *B,
                        rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_dtrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trsm<rocblas_float_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ctrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

template <>
void cblas_trsm<rocblas_double_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

// potf2
template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, float *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  spotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, double *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  dpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, rocblas_float_complex *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  cpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, rocblas_double_complex *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  zpotf2_(&uploC, &n, A, &lda, info);
}

// potrf
template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, float *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  spotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, double *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  dpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, rocblas_float_complex *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  cpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, rocblas_double_complex *A,
                        rocblas_int lda, rocblas_int *info) {
  char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
  zpotrf_(&uploC, &n, A, &lda, info);
}

// getf2
template <>
void cblas_getf2(rocblas_int m, rocblas_int n, float *A, rocblas_int lda,
                        rocblas_int *ipiv, rocblas_int *info) {
  sgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m, rocblas_int n, double *A,
                        rocblas_int lda, rocblas_int *ipiv, rocblas_int *info) {
  dgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
                        rocblas_int *ipiv, rocblas_int *info) {
  cgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m, rocblas_int n, rocblas_double_complex *A,
                        rocblas_int lda, rocblas_int *ipiv, rocblas_int *info) {
  zgetf2_(&m, &n, A, &lda, ipiv, info);
}


// trtri
template <>
rocblas_int cblas_trtri<float>(char uplo, char diag, rocblas_int n, float *A,
                               rocblas_int lda) {
  // just directly cast, since transA, transB are integers in the enum
  // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
  // );
  rocblas_int info;
  strtri_(&uplo, &diag, &n, A, &lda, &info);
  return info;
}

template <>
rocblas_int cblas_trtri<double>(char uplo, char diag, rocblas_int n, double *A,
                                rocblas_int lda) {
  // just directly cast, since transA, transB are integers in the enum
  // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
  // );
  rocblas_int info;
  dtrtri_(&uplo, &diag, &n, A, &lda, &info);
  return info;
}

// trmm
template <>
void cblas_trmm<float>(rocblas_side side, rocblas_fill uplo,
                       rocblas_operation transA, rocblas_diagonal diag,
                       rocblas_int m, rocblas_int n, float alpha,
                       const float *A, rocblas_int lda, float *B,
                       rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_strmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trmm<double>(rocblas_side side, rocblas_fill uplo,
                        rocblas_operation transA, rocblas_diagonal diag,
                        rocblas_int m, rocblas_int n, double alpha,
                        const double *A, rocblas_int lda, double *B,
                        rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_dtrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trmm<rocblas_float_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ctrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

template <>
void cblas_trmm<rocblas_double_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ztrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

// getrf
template <>
void cblas_getrf<float>(rocblas_int m, rocblas_int n, float *A,
                               rocblas_int lda, rocblas_int *ipiv, rocblas_int *info) {
  sgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<double>(rocblas_int m, rocblas_int n, double *A,
                                rocblas_int lda, rocblas_int *ipiv, rocblas_int *info) {
  dgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_float_complex>(rocblas_int m, rocblas_int n,
                                               rocblas_float_complex *A,
                                               rocblas_int lda,
                                               rocblas_int *ipiv, rocblas_int *info) {
  cgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_double_complex>(rocblas_int m, rocblas_int n,
                                                rocblas_double_complex *A,
                                                rocblas_int lda,
                                                rocblas_int *ipiv, rocblas_int *info) {
  zgetrf_(&m, &n, A, &lda, ipiv, info);
}

// getrs
template <>
rocblas_int cblas_getrs<float>(char trans, rocblas_int n, rocblas_int nrhs,
                               float *A, rocblas_int lda, rocblas_int *ipiv,
                               float *B, rocblas_int ldb) {
  rocblas_int info;
  sgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
  return info;
}

template <>
rocblas_int cblas_getrs<double>(char trans, rocblas_int n, rocblas_int nrhs,
                                double *A, rocblas_int lda, rocblas_int *ipiv,
                                double *B, rocblas_int ldb) {
  rocblas_int info;
  dgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
  return info;
}

template <>
rocblas_int
cblas_getrs<rocblas_float_complex>(char trans, rocblas_int n, rocblas_int nrhs,
                                   rocblas_float_complex *A, rocblas_int lda,
                                   rocblas_int *ipiv, rocblas_float_complex *B,
                                   rocblas_int ldb) {
  rocblas_int info;
  cgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
  return info;
}

template <>
rocblas_int cblas_getrs<rocblas_double_complex>(
    char trans, rocblas_int n, rocblas_int nrhs, rocblas_double_complex *A,
    rocblas_int lda, rocblas_int *ipiv, rocblas_double_complex *B,
    rocblas_int ldb) {
  rocblas_int info;
  zgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
  return info;
}

// geqrf
template <>
void cblas_geqrf<float>(rocblas_int m, rocblas_int n, float *A,
                        rocblas_int lda, float *ipiv, float *work, rocblas_int lwork) {
  int info;
  sgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<double>(rocblas_int m, rocblas_int n, double *A,
                         rocblas_int lda, double *ipiv, double *work, rocblas_int lwork) {
  int info;
  dgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A,
                         rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, rocblas_int lwork) {
  int info;
  cgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A,
                         rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, rocblas_int lwork) {
  int info;
  zgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geqr2
template <>
void cblas_geqr2<float>(rocblas_int m, rocblas_int n, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  sgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<double>(rocblas_int m, rocblas_int n, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  dgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  cgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  zgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gelqf
template <>
void cblas_gelqf<float>(rocblas_int m, rocblas_int n, float *A,
                        rocblas_int lda, float *ipiv, float *work, rocblas_int lwork) {
  int info;
  sgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<double>(rocblas_int m, rocblas_int n, double *A,
                         rocblas_int lda, double *ipiv, double *work, rocblas_int lwork) {
  int info;
  dgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A,
                        rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work, rocblas_int lwork) {
  int info;
  cgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A,
                         rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work, rocblas_int lwork) {
  int info;
  zgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gelq2
template <>
void cblas_gelq2<float>(rocblas_int m, rocblas_int n, float *A,
                               rocblas_int lda, float *ipiv, float *work) {
  int info;
  sgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<double>(rocblas_int m, rocblas_int n, double *A,
                               rocblas_int lda, double *ipiv, double *work) {
  int info;
  dgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A,
                               rocblas_int lda, rocblas_float_complex *ipiv, rocblas_float_complex *work) {
  int info;
  cgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A,
                               rocblas_int lda, rocblas_double_complex *ipiv, rocblas_double_complex *work) {
  int info;
  zgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}


//orgbr & ungbr
template <>
void cblas_orgbr_ungbr<float>(char storev, rocblas_int m, rocblas_int n, rocblas_int k, float *A, rocblas_int lda, float *Ipiv, float *work, rocblas_int size_w)
{
    int info;
    char vect;
    if (storev == 'C')
        vect = 'Q';
    else
        vect = 'P';
    sorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<double>(char storev,rocblas_int m, rocblas_int n, rocblas_int k, double *A, rocblas_int lda, double *Ipiv, double *work, rocblas_int size_w)
{
    int info;
    char vect;
    if (storev == 'C')
        vect = 'Q';
    else
        vect = 'P';
    dorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_float_complex>(char storev, rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *Ipiv, rocblas_float_complex *work, rocblas_int size_w)
{
    int info;
    char vect;
    if (storev == 'C')
        vect = 'Q';
    else
        vect = 'P';
    cungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_double_complex>(char storev,rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *Ipiv, rocblas_double_complex *work, rocblas_int size_w)
{
    int info;
    char vect;
    if (storev == 'C')
        vect = 'Q';
    else
        vect = 'P';
    zungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

//gebd2
template <>
void cblas_gebd2<float,float>(rocblas_int m, rocblas_int n, float *A, rocblas_int lda, float *D, float *E, float *tauq, float *taup, float *work)
{
    int info;
    sgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<double,double>(rocblas_int m, rocblas_int n, double *A, rocblas_int lda, double *D, double *E, double *tauq, double *taup, double *work)
{
    int info;
    dgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<float,rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A, rocblas_int lda, float *D, float *E, rocblas_float_complex *tauq, rocblas_float_complex *taup, rocblas_float_complex *work)
{
    int info;
    cgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<double,rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A, rocblas_int lda, double *D, double *E, rocblas_double_complex *tauq, rocblas_double_complex *taup, rocblas_double_complex *work)
{
    int info;
    zgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

//gebrd
template <>
void cblas_gebrd<float>(rocblas_int m, rocblas_int n, float *A, rocblas_int lda, float *D, float *E, float *tauq, float *taup, float *work, rocblas_int size_w)
{
    int info;
    sgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<double>(rocblas_int m, rocblas_int n, double *A, rocblas_int lda, double *D, double *E, double *tauq, double *taup, double *work, rocblas_int size_w)
{
    int info;
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *D, rocblas_float_complex *E, rocblas_float_complex *tauq, rocblas_float_complex *taup, rocblas_float_complex *work, rocblas_int size_w)
{
    int info;
    cgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *D, rocblas_double_complex *E, rocblas_double_complex *tauq, rocblas_double_complex *taup, rocblas_double_complex *work, rocblas_int size_w)
{
    int info;
    zgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}
