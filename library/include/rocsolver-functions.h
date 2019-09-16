/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCLAPACK_FUNCTIONS_H
#define _ROCLAPACK_FUNCTIONS_H

#include "rocsolver-types.h"
#include <rocblas.h>

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

/*
 * ===========================================================================
 *      Auxiliary functions
 * ===========================================================================
 */

/*! \brief AUXILIARY API

    \details
    LASWP performs a series of row interchanges on the matrix A.
    Interchanges row I with row IPIV[k1 + (I - k1) * abs(inc)], for
    each of rows K1 through K2 of A. k1 and k2 are base-1 indices.

    @param[in]
    handle          rocblas_handle
    @param[in]
    N               rocblas_int
                    The number of columns of the matrix A.
    @param[inout]
    A               matrix, dimension (M,N). 
                    On entry, the matrix of column dimension N to which the row
                    interchanges will be applied.
                    On exit, the permuted matrix.
    @param[in]
    lda             rocblas_int
                    The leading dimension of the array A. (lda >= M)
    @param[in]
    k1              rocblas_int
                    The first element of IPIV for which a row interchange will
                    be done.
    @param[in]
    k2              rocblas_int
                    (K2-K1+1) is the number of elements of IPIV for which a row
                    interchange will be done.
    @param[in]
    ipiv            rocblas_int array, dimension at least k1 + (k2 - k1) * abs(incx).
                    The vector of pivot indices.  Only the elements in positions
                    k1 through (k1 + (k2 - k1) * abs(incx)) of IPIV are accessed.
    @param[in]
    incx            rocblas_int
                    The increment between successive values of IPIV.  If IPIV
                    is negative, the pivots are applied in reverse order.
    *************************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_slaswp(rocsolver_handle handle, 
                                                   const rocsolver_int n,
                                                   float *A, 
                                                   const rocsolver_int lda, 
                                                   const rocsolver_int k1, 
                                                   const rocsolver_int k2, 
                                                   const rocsolver_int *ipiv, 
                                                   const rocblas_int incx);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dlaswp(rocsolver_handle handle, 
                                                   const rocsolver_int n,
                                                   double *A, 
                                                   const rocsolver_int lda, 
                                                   const rocsolver_int k1, 
                                                   const rocsolver_int k2, 
                                                   const rocsolver_int *ipiv, 
                                                   const rocblas_int incx);

/*! \brief AUXILIARY API

    \details
    LARFG generates an orthogonal Householder reflector H of order n, such that 

    H * [alpha] = [beta]
        [  x  ]   [  0 ]

    where x is an n-1 vector and alpha and beta are scalars. Matrix H can be 
    generated as
    
    H = I - tau * [1] * [1 v']
                  [v]  

    with v an n-1 vector and tau a scalar. 

    @param[in]
    handle          rocblas_handle
    @param[in]
    n               rocblas_int
                    The order (size) of reflector H
    @param[inout]
    alpha           pointer to scalar on the GPU.
                    on input it points to scalar alpha, 
                    on output it is overwritten with beta
    @param[inout]      
    x               pointer to a vector on the GPU (size at leadst n-1).
                    on input it is the vector x, 
                    on output it is overwritten with vector v
    @param[in]
    incx            rocblas_int
                    The increment between consecutive elements of x
    @param[out]
    tau             pointer to scalar tau

    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfg(rocsolver_handle handle, 
                                                 const rocsolver_int n, 
                                                 float *alpha,
                                                 float *x, 
                                                 const rocblas_int incx, 
                                                 float *tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfg(rocsolver_handle handle, 
                                                 const rocsolver_int n, 
                                                 double *alpha,
                                                 double *x, 
                                                 const rocblas_int incx, 
                                                 double *tau);


/*! \brief AUXILIARY API

    \details
    LARF applies a Householder reflector H of order m (or n), to a general m by n matrix A
    from the left (or the right). H is given by 

    H = I - alpha*x*x'

    where alpha is a scalar and x a Householder vector, but it is never actually computed.

    @param[in]
    handle          rocblas_handle
    @param[in]
    side            rocblas_side
                    If side = rocblas_side_left, then compute H*A
                    If side = rocblas_side_right, then compute A*H
    @param[in]
    m               rocblas_int
                    Number of rows of A
    @param[in]
    n               rocblas_int
                    Number of columns of A
    @param[in]
    x               Pointer to a vector on the GPU
                    Size is at least (1 + (m-1)*abs(incx)) if left side
                    Size is at least (1 + (n-1)*abs(incx)) if right side
    @param[in]
    incx            rocblas_int
                    Increment between to consecutive elements of x (x != 0)
    @param[in]
    alpha           Pointer to scalar in the GPU
                    If alpha = 0, then H = I (A will remain the same, x is never used)
    @param[inout]
    A               Pointer to a matrix in the GPU
                    On input, the matrix A. On output it is overwritten with
                    H*A (or A*H)
    @param[in]
    lda             rocblas_int
                    Leading dimension of A (lda >= m)
                        
    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarf(rocsolver_handle handle, 
                                                const rocsolver_side side, 
                                                const rocsolver_int m,
                                                const rocsolver_int n, 
                                                float* x, 
                                                const rocsolver_int incx, 
                                                const float* alpha,
                                                float* A, 
                                                const rocsolver_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarf(rocsolver_handle handle, 
                                                const rocsolver_side side, 
                                                const rocsolver_int m,
                                                const rocsolver_int n, 
                                                double* x, 
                                                const rocsolver_int incx, 
                                                const double* alpha,
                                                double* A, 
                                                const rocsolver_int lda);



/*
 * ===========================================================================
 *      LAPACK functions
 * ===========================================================================
 */

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

ROCSOLVER_EXPORT rocsolver_status rocsolver_dpotf2(rocsolver_handle handle,
                                                   rocsolver_fill uplo,
                                                   rocsolver_int n, double *A,
                                                   rocsolver_int lda);

/*! \brief LAPACK API

    \details
    getf2 computes an LU factorization of a general m-by-n matrix A
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
              On entry, the M-by-N matrix to be factored.
              On exit, the factors L and U from the factorization
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).
              For 1 <= i <= min(M,N), the row i of the
              matrix was interchanged with row IPIV(i).
              Matrix P of the factorization can be derived from ipiv
              (All row indices are base-1).
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetf2(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *A,
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   rocsolver_int *info);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetf2(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *A,
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   rocsolver_int *info);


/*! \brief LAPACK API

    \details
    getf2_batched computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    The factorization of matrix A_i in the batch has the form
       A_i = P_i * L_i * U_i
    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of all matrices A_i in the batch. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of all matrices A_i in the batch. n >= 0.
    @param[inout]
    A         Array of pointers storing the different matrices A_i on the GPU.
              On entry, the M-by-N matrix A_i to be factored.
              On exit, the factors L_i and U_i from the factorization
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of matrices A_i. lda >= max(1,m).
    @param[out]
    ipiv      pointer to the first vector of pivots ipiv_0 (corresponding to A_0) on the GPU. 
              For each instance A_i in the batch and for 1 <= j <= min(M,N), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i
              (All row indices are base-1).
    @param[in]
    strideP   stride from the start of one vector ipiv_i to the next one ipiv_(i+1)
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
    @param[in]
    batch_count  number of matrices in the batch
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetf2_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *const A[],
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetf2_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *const A[],
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);
/*! \brief LAPACK API

    \details
    getf2_strided_batched computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    The factorization of matrix A_i in the batch has the form
       A_i = P_i * L_i * U_i
    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of all matrices A_i in the batch. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of all matrices A_i in the batch. n >= 0.
    @param[inout]
    A         Pointer to the first matrix A_0 on the GPU.
              On entry, the M-by-N matrix A_i to be factored.
              On exit, the factors L_i and U_i from the factorization
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of matrices A_i. lda >= max(1,m).
    @param[in]
    strideA   stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[out]
    ipiv      pointer to the first vector of pivots ipiv_0 (corresponding to A_0) on the GPU. 
              For each instance A_i in the batch and for 1 <= j <= min(M,N), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i
              (All row indices are base-1).
    @param[in]
    strideP   stride from the start of one vector ipiv_i to the next one ipiv_(i+1)
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
    @param[in]
    batch_count  number of matrices in the batch
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetf2_strided_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *A,
                                                   const rocsolver_int lda,
                                                   const rocsolver_int strideA,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetf2_strided_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *A,
                                                   const rocsolver_int lda,
                                                   const rocsolver_int strideA,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);


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
              On entry, the M-by-N matrix to be factored.
              On exit, the factors L and U from the factorization
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of A. lda >= max(1,m).
    @param[out]
    ipiv      pointer storing pivots on the GPU. Dimension (min(m,n)).
              For 1 <= i <= min(M,N), the row i of the
              matrix was interchanged with row IPIV(i).
              Matrix P of the factorization can be derived from ipiv
              (All row indices are base-1).
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetrf(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *A,
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   rocsolver_int *info);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrf(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *A,
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   rocsolver_int *info);

/*! \brief LAPACK API

    \details
    getrf_batched computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    The factorization of matrix A_i in the batch has the form
       A_i = P_i * L_i * U_i
    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of all matrices A_i in the batch. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of all matrices A_i in the batch. n >= 0.
    @param[inout]
    A         Array of pointers storing the different matrices A_i on the GPU.
              On entry, the M-by-N matrix A_i to be factored.
              On exit, the factors L_i and U_i from the factorization
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of matrices A_i. lda >= max(1,m).
    @param[out]
    ipiv      pointer to the first vector of pivots ipiv_0 (corresponding to A_0) on the GPU. 
              For each instance A_i in the batch and for 1 <= j <= min(M,N), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i
              (All row indices are base-1).
    @param[in]
    strideP   stride from the start of one vector ipiv_i to the next one ipiv_(i+1)
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
    @param[in]
    batch_count  number of matrices in the batch
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetrf_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *const A[],
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrf_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *const A[],
                                                   const rocsolver_int lda,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);

/*! \brief LAPACK API

    \details
    getrf_strided_batched computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    The factorization of matrix A_i in the batch has the form
       A_i = P_i * L_i * U_i
    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    @param[in]
    handle    rocsolver_handle.
              handle to the rocsolver library context queue.
    @param[in]
    m         rocsolver_int
              the number of rows of all matrices A_i in the batch. m >= 0.
    @param[in]
    n         rocsolver_int
              the number of colums of all matrices A_i in the batch. n >= 0.
    @param[inout]
    A         Pointer to the first matrix A_0 on the GPU.
              On entry, the M-by-N matrix A_i to be factored.
              On exit, the factors L_i and U_i from the factorization
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocsolver_int
              specifies the leading dimension of matrices A_i. lda >= max(1,m).
    @param[in]
    strideA   stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[out]
    ipiv      pointer to the first vector of pivots ipiv_0 (corresponding to A_0) on the GPU. 
              For each instance A_i in the batch and for 1 <= j <= min(M,N), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i
              (All row indices are base-1).
    @param[in]
    strideP   stride from the start of one vector ipiv_i to the next one ipiv_(i+1)
    @param[out]
    info      pointer storing and integer on the GPU.
              If info = 0, succesful exit. 
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.
    @param[in]
    batch_count  number of matrices in the batch
            
    ********************************************************************/

ROCSOLVER_EXPORT rocsolver_status rocsolver_sgetrf_strided_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   float *A,
                                                   const rocsolver_int lda,
                                                   const rocsolver_int strideA,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrf_strided_batched(rocsolver_handle handle,
                                                   const rocsolver_int m,
                                                   const rocsolver_int n, 
                                                   double *A,
                                                   const rocsolver_int lda,
                                                   const rocsolver_int strideA,
                                                   rocsolver_int *ipiv,
                                                   const rocsolver_int strideP,
                                                   rocsolver_int *info,
                                                   const rocsolver_int batch_count);


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

ROCSOLVER_EXPORT rocsolver_status rocsolver_dgetrs(
    rocsolver_handle handle, rocsolver_operation trans, rocsolver_int n,
    rocsolver_int nrhs, const double *A, rocsolver_int lda,
    const rocsolver_int *ipiv, double *B, rocsolver_int ldb);

#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */
