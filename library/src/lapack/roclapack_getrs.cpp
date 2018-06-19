/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_sgetrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                 rocblas_int nrhs, const float *A, rocblas_int lda,
                 const rocblas_int *ipiv, float *B, rocblas_int ldb) {
  return rocsolver_getrs_template<float>(handle, trans, n, nrhs, A, lda, ipiv,
                                         B, ldb);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dgetrs(rocblas_handle handle, rocblas_operation trans, rocblas_int n,
                 rocblas_int nrhs, const double *A, rocblas_int lda,
                 const rocblas_int *ipiv, double *B, rocblas_int ldb) {
  return rocsolver_getrs_template<double>(handle, trans, n, nrhs, A, lda, ipiv,
                                          B, ldb);
}
