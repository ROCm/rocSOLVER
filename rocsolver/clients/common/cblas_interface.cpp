/* ************************************************************************
 * Copyright (c) 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include "cblas_interface.h"
#include "cblas.h"
#include "rocblas.h"
//#include "rocsolver.h"
//#include "utility.h"
//#include <memory>
//#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is
 * only used for testing not part of the GPU library
 */

#ifdef __cplusplus
extern "C" {
#endif

void strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri_(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri_(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* m, rocblas_float_complex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* m, rocblas_double_complex* A, int* lda, int* info);

void spotf2_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotf2_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotf2_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotf2_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetf2_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetf2_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetf2_(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetf2_(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void sgetrs_(char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs_(char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs_(char* trans,
             int* n,
             int* nrhs,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* B,
             int* ldb,
             int* info);
void zgetrs_(char* trans,
             int* n,
             int* nrhs,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* B,
             int* ldb,
             int* info);

void sgetri_(int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void cgetri_(int* n,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgetri_(int* n,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void slarfg_(int* n, float* alpha, float* x, int* incx, float* tau);
void dlarfg_(int* n, double* alpha, double* x, int* incx, double* tau);
void clarfg_(int* n,
             rocblas_float_complex* alpha,
             rocblas_float_complex* x,
             int* incx,
             rocblas_float_complex* tau);
void zlarfg_(int* n,
             rocblas_double_complex* alpha,
             rocblas_double_complex* x,
             int* incx,
             rocblas_double_complex* tau);

void slarf_(char* side, int* m, int* n, float* x, int* incx, float* alpha, float* A, int* lda, float* work);
void dlarf_(char* side,
            int* m,
            int* n,
            double* x,
            int* incx,
            double* alpha,
            double* A,
            int* lda,
            double* work);
void clarf_(char* side,
            int* m,
            int* n,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* work);
void zlarf_(char* side,
            int* m,
            int* n,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* work);

void slarft_(char* direct, char* storev, int* n, int* k, float* V, int* ldv, float* tau, float* T, int* ldt);
void dlarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             double* V,
             int* ldv,
             double* tau,
             double* T,
             int* ldt);
void clarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* tau,
             rocblas_float_complex* T,
             int* ldt);
void zlarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* tau,
             rocblas_double_complex* T,
             int* ldt);

void sbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             float* D,
             float* E,
             float* V,
             int* ldv,
             float* U,
             int* ldu,
             float* C,
             int* ldc,
             float* W,
             int* info);
void dbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             double* D,
             double* E,
             double* V,
             int* ldv,
             double* U,
             int* ldu,
             double* C,
             int* ldc,
             double* W,
             int* info);
void cbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             float* D,
             float* E,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* U,
             int* ldu,
             rocblas_float_complex* C,
             int* ldc,
             float* W,
             int* info);
void zbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             double* D,
             double* E,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* U,
             int* ldu,
             rocblas_double_complex* C,
             int* ldc,
             double* W,
             int* info);

void slarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             float* V,
             int* ldv,
             float* T,
             int* ldt,
             float* A,
             int* lda,
             float* W,
             int* ldw);
void dlarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             double* V,
             int* ldv,
             double* T,
             int* ldt,
             double* A,
             int* lda,
             double* W,
             int* ldw);
void clarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* T,
             int* ldt,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* W,
             int* ldw);
void zlarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* T,
             int* ldt,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* W,
             int* ldw);

void slabrd_(int* m,
             int* n,
             int* nb,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* X,
             int* ldx,
             float* Y,
             int* ldy);
void dlabrd_(int* m,
             int* n,
             int* nb,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* X,
             int* ldx,
             double* Y,
             int* ldy);
void clabrd_(int* m,
             int* n,
             int* nb,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* X,
             int* ldx,
             rocblas_float_complex* Y,
             int* ldy);
void zlabrd_(int* m,
             int* n,
             int* nb,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* X,
             int* ldx,
             rocblas_double_complex* Y,
             int* ldy);

void sgeqr2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeqr2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeqr2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgeqr2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sgeqrf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqrf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqrf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgeqrf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgeql2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeql2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeql2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgeql2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sgeqlf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqlf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqlf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgeqlf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgelq2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgelq2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgelq2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgelq2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sgelqf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgelqf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgelqf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgelqf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgerq2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgerq2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgerq2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgerq2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sgerqf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgerqf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgerqf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgerqf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void clacgv_(int* n, rocblas_float_complex* x, int* incx);
void zlacgv_(int* n, rocblas_double_complex* x, int* incx);

void slaswp_(int* n, float* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void dlaswp_(int* n, double* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void claswp_(int* n, rocblas_float_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void zlaswp_(int* n, rocblas_double_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);

void sorg2r_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2r_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2r_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zung2r_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sorgqr_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgqr_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungqr_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zungqr_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorgl2_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorgl2_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cungl2_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zungl2_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sorglq_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorglq_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cunglq_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zunglq_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorg2l_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2l_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2l_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zung2l_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);
void sorgql_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgql_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungql_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zungql_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorgbr_(char* vect,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* Ipiv,
             float* work,
             int* size_w,
             int* info);
void dorgbr_(char* vect,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* Ipiv,
             double* work,
             int* size_w,
             int* info);
void cungbr_(char* vect,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* Ipiv,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zungbr_(char* vect,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* Ipiv,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void sorgtr_(char* uplo, int* n, float* A, int* lda, float* Ipiv, float* work, int* size_w, int* info);
void dorgtr_(char* uplo, int* n, double* A, int* lda, double* Ipiv, double* work, int* size_w, int* info);
void cungtr_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* Ipiv,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zungtr_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* Ipiv,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void sorm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);
void sormqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sorml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);
void sormlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sorm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);
void sormql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sormbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sormtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sgebd2_(int* m,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* work,
             int* info);
void dgebd2_(int* m,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* work,
             int* info);
void cgebd2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* work,
             int* info);
void zgebd2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* work,
             int* info);

void sgebrd_(int* m,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* work,
             int* size_w,
             int* info);
void dgebrd_(int* m,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* work,
             int* size_w,
             int* info);
void cgebrd_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zgebrd_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void ssytrd_(char* uplo,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tau,
             float* work,
             int* size_w,
             int* info);
void dsytrd_(char* uplo,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tau,
             double* work,
             int* size_w,
             int* info);
void chetrd_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tau,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zhetrd_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tau,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void sgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             float* A,
             int* lda,
             float* S,
             float* U,
             int* ldu,
             float* V,
             int* ldv,
             float* E,
             int* lwork,
             int* info);
void dgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             double* A,
             int* lda,
             double* S,
             double* U,
             int* ldu,
             double* V,
             int* ldv,
             double* E,
             int* lwork,
             int* info);
void cgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* S,
             rocblas_float_complex* U,
             int* ldu,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* work,
             int* lwork,
             float* E,
             int* info);
void zgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* S,
             rocblas_double_complex* U,
             int* ldu,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* work,
             int* lwork,
             double* E,
             int* info);

#ifdef __cplusplus
}
#endif

// lacgv

template <>
void cblas_lacgv<rocblas_float_complex>(rocblas_int n, rocblas_float_complex* x, rocblas_int incx)
{
    clacgv_(&n, x, &incx);
}

template <>
void cblas_lacgv<rocblas_double_complex>(rocblas_int n, rocblas_double_complex* x, rocblas_int incx)
{
    zlacgv_(&n, x, &incx);
}

// laswp

template <>
void cblas_laswp<float>(rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int k1,
                        rocblas_int k2,
                        rocblas_int* ipiv,
                        rocblas_int inc)
{
    slaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<double>(rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int k1,
                         rocblas_int k2,
                         rocblas_int* ipiv,
                         rocblas_int inc)
{
    dlaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int k1,
                                        rocblas_int k2,
                                        rocblas_int* ipiv,
                                        rocblas_int inc)
{
    claswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int k1,
                                         rocblas_int k2,
                                         rocblas_int* ipiv,
                                         rocblas_int inc)
{
    zlaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

// larfg

template <>
void cblas_larfg<float>(rocblas_int n, float* alpha, float* x, rocblas_int incx, float* tau)
{
    slarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<double>(rocblas_int n, double* alpha, double* x, rocblas_int incx, double* tau)
{
    dlarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* alpha,
                                        rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_float_complex* tau)
{
    clarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* alpha,
                                         rocblas_double_complex* x,
                                         rocblas_int incx,
                                         rocblas_double_complex* tau)
{
    zlarfg_(&n, alpha, x, &incx, tau);
}

// larf

template <>
void cblas_larf<float>(rocblas_side sideR,
                       rocblas_int m,
                       rocblas_int n,
                       float* x,
                       rocblas_int incx,
                       float* alpha,
                       float* A,
                       rocblas_int lda,
                       float* work)
{
    char side = rocblas2char_side(sideR);
    slarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<double>(rocblas_side sideR,
                        rocblas_int m,
                        rocblas_int n,
                        double* x,
                        rocblas_int incx,
                        double* alpha,
                        double* A,
                        rocblas_int lda,
                        double* work)
{
    char side = rocblas2char_side(sideR);
    dlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_float_complex>(rocblas_side sideR,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex* alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* work)
{
    char side = rocblas2char_side(sideR);
    clarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_double_complex>(rocblas_side sideR,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* work)
{
    char side = rocblas2char_side(sideR);
    zlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

// larft

template <>
void cblas_larft<float>(rocblas_direct directR,
                        rocblas_storev storevR,
                        rocblas_int n,
                        rocblas_int k,
                        float* V,
                        rocblas_int ldv,
                        float* tau,
                        float* T,
                        rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<double>(rocblas_direct directR,
                         rocblas_storev storevR,
                         rocblas_int n,
                         rocblas_int k,
                         double* V,
                         rocblas_int ldv,
                         double* tau,
                         double* T,
                         rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_float_complex>(rocblas_direct directR,
                                        rocblas_storev storevR,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_float_complex* V,
                                        rocblas_int ldv,
                                        rocblas_float_complex* tau,
                                        rocblas_float_complex* T,
                                        rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_double_complex>(rocblas_direct directR,
                                         rocblas_storev storevR,
                                         rocblas_int n,
                                         rocblas_int k,
                                         rocblas_double_complex* V,
                                         rocblas_int ldv,
                                         rocblas_double_complex* tau,
                                         rocblas_double_complex* T,
                                         rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

// larfb

template <>
void cblas_larfb<float>(rocblas_side sideR,
                        rocblas_operation transR,
                        rocblas_direct directR,
                        rocblas_storev storevR,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int k,
                        float* V,
                        rocblas_int ldv,
                        float* T,
                        rocblas_int ldt,
                        float* A,
                        rocblas_int lda,
                        float* W,
                        rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<double>(rocblas_side sideR,
                         rocblas_operation transR,
                         rocblas_direct directR,
                         rocblas_storev storevR,
                         rocblas_int m,
                         rocblas_int n,
                         rocblas_int k,
                         double* V,
                         rocblas_int ldv,
                         double* T,
                         rocblas_int ldt,
                         double* A,
                         rocblas_int lda,
                         double* W,
                         rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_float_complex>(rocblas_side sideR,
                                        rocblas_operation transR,
                                        rocblas_direct directR,
                                        rocblas_storev storevR,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_float_complex* V,
                                        rocblas_int ldv,
                                        rocblas_float_complex* T,
                                        rocblas_int ldt,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* W,
                                        rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_double_complex>(rocblas_side sideR,
                                         rocblas_operation transR,
                                         rocblas_direct directR,
                                         rocblas_storev storevR,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         rocblas_double_complex* V,
                                         rocblas_int ldv,
                                         rocblas_double_complex* T,
                                         rocblas_int ldt,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* W,
                                         rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

// bdsqr
template <>
void cblas_bdsqr(rocblas_fill uplo,
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
                 float* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    sbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
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
                 double* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
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
                 float* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    cbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
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
                 double* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    zbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

// gesvd
template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 float* A,
                 rocblas_int lda,
                 float* S,
                 float* U,
                 rocblas_int ldu,
                 float* V,
                 rocblas_int ldv,
                 float* work,
                 rocblas_int lwork,
                 float* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    sgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 double* A,
                 rocblas_int lda,
                 double* S,
                 double* U,
                 rocblas_int ldu,
                 double* V,
                 rocblas_int ldv,
                 double* work,
                 rocblas_int lwork,
                 double* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    dgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 float* S,
                 rocblas_float_complex* U,
                 rocblas_int ldu,
                 rocblas_float_complex* V,
                 rocblas_int ldv,
                 rocblas_float_complex* work,
                 rocblas_int lwork,
                 float* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    cgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 double* S,
                 rocblas_double_complex* U,
                 rocblas_int ldu,
                 rocblas_double_complex* V,
                 rocblas_int ldv,
                 rocblas_double_complex* work,
                 rocblas_int lwork,
                 double* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    zgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

// labrd
template <>
void cblas_labrd<float, float>(rocblas_int m,
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
    int info;
    slabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<double, double>(rocblas_int m,
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
    int info;
    dlabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<float, rocblas_float_complex>(rocblas_int m,
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
    int info;
    clabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<double, rocblas_double_complex>(rocblas_int m,
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
    int info;
    zlabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

// orgqr & ungqr
template <>
void cblas_orgqr_ungqr<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2r & ung2r
template <>
void cblas_org2r_ung2r<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orglq & unglq
template <>
void cblas_orglq_unglq<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// orgl2 & ungl2
template <>
void cblas_orgl2_ungl2<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgql & ungql
template <>
void cblas_orgql_ungql<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorgql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorgql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cungql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zungql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2l & ung2l
template <>
void cblas_org2l_ung2l<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorg2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorg2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cung2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zung2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgbr & ungbr
template <>
void cblas_orgbr_ungbr<float>(rocblas_storev storev,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* Ipiv,
                              float* work,
                              rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    sorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<double>(rocblas_storev storev,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* Ipiv,
                               double* work,
                               rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    dorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_float_complex>(rocblas_storev storev,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* Ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    cungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_double_complex>(rocblas_storev storev,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* Ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    zungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

// orgtr & ungtr
template <>
void cblas_orgtr_ungtr<float>(rocblas_fill uplo,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* Ipiv,
                              float* work,
                              rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    sorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<double>(rocblas_fill uplo,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* Ipiv,
                               double* work,
                               rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<rocblas_float_complex>(rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* Ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    cungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<rocblas_double_complex>(rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* Ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

// ormqr & unmqr
template <>
void cblas_ormqr_unmqr<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2r & unm2r
template <>
void cblas_orm2r_unm2r<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormlq & unmlq
template <>
void cblas_ormlq_unmlq<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orml2 & unml2
template <>
void cblas_orml2_unml2<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormql & unmql
template <>
void cblas_ormql_unmql<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2l & unm2l
template <>
void cblas_orm2l_unm2l<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormbr & unmbr
template <>
void cblas_ormbr_unmbr<float>(rocblas_storev storev,
                              rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    sormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<double>(rocblas_storev storev,
                               rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    dormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_float_complex>(rocblas_storev storev,
                                              rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    cunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_double_complex>(rocblas_storev storev,
                                               rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    zunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// ormtr & unmtr
template <>
void cblas_ormtr_unmtr<float>(rocblas_side side,
                              rocblas_fill uplo,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    sormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<double>(rocblas_side side,
                               rocblas_fill uplo,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    dormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<rocblas_float_complex>(rocblas_side side,
                                              rocblas_fill uplo,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    cunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<rocblas_double_complex>(rocblas_side side,
                                               rocblas_fill uplo,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    zunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// scal
/*template <>
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

// gemv
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

// gemm
template <>
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
}
*/

template <>
void cblas_gemm<float>(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int m,
                       rocblas_int n,
                       rocblas_int k,
                       float alpha,
                       float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb,
                       float beta,
                       float* C,
                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
    // );
    cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<double>(rocblas_operation transA,
                        rocblas_operation transB,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int k,
                        double alpha,
                        double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb,
                        double beta,
                        double* C,
                        rocblas_int ldc)
{
    cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_float_complex>(rocblas_operation transA,
                                       rocblas_operation transB,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int k,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb,
                                       rocblas_float_complex beta,
                                       rocblas_float_complex* C,
                                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A,
                lda, B, ldb, &beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_double_complex>(rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb,
                                        rocblas_double_complex beta,
                                        rocblas_double_complex* C,
                                        rocblas_int ldc)
{
    cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A,
                lda, B, ldb, &beta, C, ldc);
}

/*
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
*/

// potf2
template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    spotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    cpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    zpotf2_(&uploC, &n, A, &lda, info);
}

// potrf
template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    spotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    cpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    zpotrf_(&uploC, &n, A, &lda, info);
}

// getf2
template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 float* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    sgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 double* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    dgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    cgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    zgetf2_(&m, &n, A, &lda, ipiv, info);
}

/*
// trtri
template <>
rocblas_int cblas_trtri<float>(rocblas_fill uplo, rocblas_diagonal diag,
rocblas_int n, float *A, rocblas_int lda) { rocblas_int info; char uploC =
rocblas2char_fill(uplo); char diagC = rocblas2char_diagonal(diag);
  strtri_(&uploC, &diagC, &n, A, &lda, &info);
  return info;
}

template <>
rocblas_int cblas_trtri<double>(rocblas_fill uplo, rocblas_diagonal diag,
rocblas_int n, double *A, rocblas_int lda) { rocblas_int info; char uploC =
rocblas2char_fill(uplo); char diagC = rocblas2char_diagonal(diag);
  dtrtri_(&uploC, &diagC, &n, A, &lda, &info);
  return info;
}

template <>
rocblas_int cblas_trtri<rocblas_float_complex>(rocblas_fill uplo,
rocblas_diagonal diag, rocblas_int n, rocblas_float_complex *A, rocblas_int lda)
{ rocblas_int info; char uploC = rocblas2char_fill(uplo); char diagC =
rocblas2char_diagonal(diag); ctrtri_(&uploC, &diagC, &n, A, &lda, &info); return
info;
}

template <>
rocblas_int cblas_trtri<rocblas_double_complex>(rocblas_fill uplo,
rocblas_diagonal diag, rocblas_int n, rocblas_double_complex *A, rocblas_int
lda) { rocblas_int info; char uploC = rocblas2char_fill(uplo); char diagC =
rocblas2char_diagonal(diag); ztrtri_(&uploC, &diagC, &n, A, &lda, &info); return
info;
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
*/

// getrf
template <>
void cblas_getrf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        rocblas_int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         rocblas_int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}

// getrs
template <>
void cblas_getrs<float>(rocblas_operation trans,
                        rocblas_int n,
                        rocblas_int nrhs,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        float* B,
                        rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    sgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<double>(rocblas_operation trans,
                         rocblas_int n,
                         rocblas_int nrhs,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         double* B,
                         rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    dgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<rocblas_float_complex>(rocblas_operation trans,
                                        rocblas_int n,
                                        rocblas_int nrhs,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_float_complex* B,
                                        rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    cgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<rocblas_double_complex>(rocblas_operation trans,
                                         rocblas_int n,
                                         rocblas_int nrhs,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_double_complex* B,
                                         rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    zgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

// getri
template <>
void cblas_getri<float>(rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        float* work,
                        rocblas_int* lwork,
                        rocblas_int* info)
{
    sgetri_(&n, A, &lda, ipiv, work, lwork, info);
}

template <>
void cblas_getri<double>(rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         double* work,
                         rocblas_int* lwork,
                         rocblas_int* info)
{
    dgetri_(&n, A, &lda, ipiv, work, lwork, info);
}

template <>
void cblas_getri<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int* lwork,
                                        rocblas_int* info)
{
    cgetri_(&n, A, &lda, ipiv, work, lwork, info);
}

template <>
void cblas_getri<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int* lwork,
                                         rocblas_int* info)
{
    zgetri_(&n, A, &lda, ipiv, work, lwork, info);
}

// geqrf
template <>
void cblas_geqrf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geqr2
template <>
void cblas_geqr2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

// geqlf
template <>
void cblas_geqlf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geql2
template <>
void cblas_geql2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gelqf
template <>
void cblas_gelqf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gelq2
template <>
void cblas_gelq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gerqf
template <>
void cblas_gerqf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gerq2
template <>
void cblas_gerq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gebd2
template <>
void cblas_gebd2<float, float>(rocblas_int m,
                               rocblas_int n,
                               float* A,
                               rocblas_int lda,
                               float* D,
                               float* E,
                               float* tauq,
                               float* taup,
                               float* work)
{
    int info;
    sgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<double, double>(rocblas_int m,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double* D,
                                 double* E,
                                 double* tauq,
                                 double* taup,
                                 double* work)
{
    int info;
    dgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<float, rocblas_float_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* tauq,
                                               rocblas_float_complex* taup,
                                               rocblas_float_complex* work)
{
    int info;
    cgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<double, rocblas_double_complex>(rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* work)
{
    int info;
    zgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

// gebrd
template <>
void cblas_gebrd<float, float>(rocblas_int m,
                               rocblas_int n,
                               float* A,
                               rocblas_int lda,
                               float* D,
                               float* E,
                               float* tauq,
                               float* taup,
                               float* work,
                               rocblas_int size_w)
{
    int info;
    sgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<double, double>(rocblas_int m,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double* D,
                                 double* E,
                                 double* tauq,
                                 double* taup,
                                 double* work,
                                 rocblas_int size_w)
{
    int info;
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<float, rocblas_float_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* tauq,
                                               rocblas_float_complex* taup,
                                               rocblas_float_complex* work,
                                               rocblas_int size_w)
{
    int info;
    cgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<double, rocblas_double_complex>(rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* work,
                                                 rocblas_int size_w)
{
    int info;
    zgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

// sytrd & hetrd
template <>
void cblas_sytrd_hetrd<float, float>(rocblas_fill uplo,
                                     rocblas_int n,
                                     float* A,
                                     rocblas_int lda,
                                     float* D,
                                     float* E,
                                     float* tau,
                                     float* work,
                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    ssytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<double, double>(rocblas_fill uplo,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       double* D,
                                       double* E,
                                       double* tau,
                                       double* work,
                                       rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dsytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<float, rocblas_float_complex>(rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     float* D,
                                                     float* E,
                                                     rocblas_float_complex* tau,
                                                     rocblas_float_complex* work,
                                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    chetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<double, rocblas_double_complex>(rocblas_fill uplo,
                                                       rocblas_int n,
                                                       rocblas_double_complex* A,
                                                       rocblas_int lda,
                                                       double* D,
                                                       double* E,
                                                       rocblas_double_complex* tau,
                                                       rocblas_double_complex* work,
                                                       rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zhetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}
