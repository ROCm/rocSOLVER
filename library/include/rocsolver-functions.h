/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCLAPACK_FUNCTIONS_H
#define _ROCLAPACK_FUNCTIONS_H

#include "rocsolver-types.h"
#include <rocblas.h>

/*!\file
 * \brief rocsolver_netlib.h provides Basic Linear Algebra Subprograms of Level
 * 1, 2 and 3, using HIP optimized for AMD HCC-based GPU hardware. This library
 * can also run on CUDA-based NVIDIA GPUs. This file exposes C89 BLAS interface
 */

/*
 * ===========================================================================
 *   READEME: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief LAPACK API

\details
potf2 computes the Cholesky factorization of a real symmetric
positive definite matrix A.

    A = U' * U ,  if UPLO = 'U', or
    A = L  * L',  if UPLO = 'L',
where U is an upper triangular matrix and L is lower triangular.

This is the unblocked version of the algorithm, calling Level 2 BLAS.

@param[in]
handle    rocsolver_handle.
          handle to the rocsolver library context queue.
@param[in]
uplo      rocsolver_fill.
          specifies whether the upper or lower
@param[in]
n         the matrix dimensions
@param[inout]
A         pointer storing matrix A on the GPU.
@param[in]
lda       rocsolver_int
          specifies the leading dimension of A.


********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_spotf2(rocsolver_handle handle,
                                                   rocsolver_fill uplo,
                                                   rocsolver_int n, float *A,
                                                   rocsolver_int lda);

/*! \brief LAPACK API

    \details
    potf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

        A = U' * U ,  if UPLO = 'U', or
        A = L  * L',  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    uplo      rocsolver_fill.
              specifies whether the upper or lower
    @param[in]
    n         the matrix dimensions
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A.


    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_dpotf2(rocsolver_handle handle,
                                                   rocsolver_fill uplo,
                                                   rocsolver_int n, double *A,
                                                   rocsolver_int lda);

/*! \brief LAPACK API

    \details
    potf2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of the matrix A. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of the matrix A. n >= 0.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).

    This implementation will even upon encountering a singularity continue
    and only in the end return an error code.

    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetf2(rocsolver_handle handle,
                                                   rocsolver_int m,
                                                   rocsolver_int n, float *A,
                                                   rocsolver_int lda,
                                                   rocsolver_int *ipiv);

/*! \brief LAPACK API

    \details
    potf2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of the matrix A. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of the matrix A. n >= 0.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).

    This implementation will even upon encountering a singularity continue
    and only in the end return an error code.

    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetf2(rocsolver_handle handle,
                                                   rocsolver_int m,
                                                   rocsolver_int n, double *A,
                                                   rocsolver_int lda,
                                                   rocsolver_int *ipiv);

/*! \brief LAPACK API

    \details
    getrf computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of the matrix A. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of the matrix A. n >= 0.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).

    This implementation will even upon encountering a singularity continue
    and only in the end return an error code.

    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetrf(rocsolver_handle handle,
                                                   rocsolver_int m,
                                                   rocsolver_int n, float *A,
                                                   rocsolver_int lda,
                                                   rocsolver_int *ipiv);

/*! \brief LAPACK API

    \details
    potrf computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of the matrix A. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of the matrix A. n >= 0.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).

    This implementation will even upon encountering a singularity continue
    and only in the end return an error code.

    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrf(rocsolver_handle handle,
                                                   rocsolver_int m,
                                                   rocsolver_int n, double *A,
                                                   rocsolver_int lda,
                                                   rocsolver_int *ipiv);

/*! \brief LAPACK API

  \details
  getrs solves a system of linear equations
     A * X = B,  A**T * X = B,  or  A**H * X = B
  with a general N-by-N matrix A using the LU factorization computed
  by getrf.

  @param[in]
  trans
           Specifies the form of the system of equations:
           = 'N':  A * X = B     (No transpose)
           = 'T':  A**T * X = B  (Transpose)
           = 'C':  A**H * X = B  (Conjugate transpose)

  @param[in]
  n
           The order of the matrix A.  N >= 0.

  @param[in]
  nrhs
           The number of right hand sides, i.e., the number of columns
           of the matrix B.  nrhs >= 0.

  @param[in]
  A
           pointer storing matrix A on the GPU.

  @param[in]
  lda
           The leading dimension of the array A.  lda >= max(1,n).

  @param[in]
  ipiv
           The pivot indices from getrf; for 1<=i<=n, row i of the
           matrix was interchanged with row ipiv(i). Assumes one-based indices!

  @param[in,out]
  B
           pointer storing matrix B on the GPU., dimension (ldb,nrhs)
           On entry, the right hand side matrix B.
           On exit, the solution matrix X.

  @param[in]
  ldb
           The leading dimension of the array B.  ldb >= max(1,n).

   ********************************************************************/
ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetrs(
    rocsolver_handle handle, rocsolver_operation trans, rocsolver_int n,
    rocsolver_int nrhs, const float *A, rocsolver_int lda,
    const rocsolver_int *ipiv, float *B, rocsolver_int ldb);

/*! \brief LAPACK API

  \details
  getrs solves a system of linear equations
     A * X = B,  A**T * X = B,  or  A**H * X = B
  with a general N-by-N matrix A using the LU factorization computed
  by getrf.

  @param[in]
  trans
           Specifies the form of the system of equations:
           = 'N':  A * X = B     (No transpose)
           = 'T':  A**T * X = B  (Transpose)
           = 'C':  A**H * X = B  (Conjugate transpose)

  @param[in]
  n
           The order of the matrix A.  N >= 0.

  @param[in]
  nrhs
           The number of right hand sides, i.e., the number of columns
           of the matrix B.  nrhs >= 0.

  @param[in]
  A
           pointer storing matrix A on the GPU.

  @param[in]
  lda
           The leading dimension of the array A.  lda >= max(1,n).

  @param[in]
  ipiv
           The pivot indices from getrf; for 1<=i<=n, row i of the
           matrix was interchanged with row ipiv(i). Assumes one-based indices!

  @param[in,out]
  B
           pointer storing matrix B on the GPU., dimension (ldb,nrhs)
           On entry, the right hand side matrix B.
           On exit, the solution matrix X.

  @param[in]
  ldb
           The leading dimension of the array B.  ldb >= max(1,n).

   ********************************************************************/
ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrs(
    rocsolver_handle handle, rocsolver_operation trans, rocsolver_int n,
    rocsolver_int nrhs, const double *A, rocsolver_int lda,
    const rocsolver_int *ipiv, double *B, rocsolver_int ldb);
#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */
