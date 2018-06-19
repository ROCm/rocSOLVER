/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_sgetf2(rocblas_handle handle, rocblas_int m, rocblas_int n, float *A,
                 rocblas_int lda, rocblas_int *ipiv) {
  return rocsolver_getf2_template<float>(handle, m, n, A, lda, ipiv);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dgetf2(rocblas_handle handle, rocblas_int m, rocblas_int n, double *A,
                 rocblas_int lda, rocblas_int *ipiv) {
  return rocsolver_getf2_template<double>(handle, m, n, A, lda, ipiv);
}