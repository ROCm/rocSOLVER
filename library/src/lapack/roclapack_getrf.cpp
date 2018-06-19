/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_sgetrf(rocsolver_handle handle, rocsolver_int m, rocsolver_int n,
                 float *A, rocsolver_int lda, rocsolver_int *ipiv) {
  return rocsolver_getrf_template<float>(handle, m, n, A, lda, ipiv);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dgetrf(rocsolver_handle handle, rocsolver_int m, rocsolver_int n,
                 double *A, rocsolver_int lda, rocsolver_int *ipiv) {
  return rocsolver_getrf_template<double>(handle, m, n, A, lda, ipiv);
}