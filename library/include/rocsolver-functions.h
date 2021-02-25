/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCLAPACK_FUNCTIONS_H
#define _ROCLAPACK_FUNCTIONS_H

#include "rocsolver-extra-types.h"
#include <rocblas.h>

/*! \file
    \brief rocsolver_functions.h provides Lapack functionality for the ROCm platform.
 *********************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *      Build information (library version)
 * ===========================================================================
 */

/*! \brief GET_VERSION_STRING Queries the library version.

    \details
    @param[out]
    buf         A buffer that the version string will be written into.
    @param[in]
    len         The size of the given buffer in bytes.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_get_version_string(char* buf, size_t len);

/*
 * ===========================================================================
 *      Multi-level logging
 * ===========================================================================
 */

/*! \brief LOG_BEGIN begins a rocSOLVER multi-level logging session.

    \details
    Initializes the rocSOLVER logging environment with default values (no
    logging and one level depth). Default mode can be overridden by using the
    environment variables ROCSOLVER_LAYER and ROCSOLVER_LEVELS.

    This function also sets the streams where the log results will be outputted.
    The default is STDERR for all the modes. This default can also be overridden
    using the environment variable ROCSOLVER_LOG_PATH, or specifically
    ROCSOLVER_LOG_TRACE_PATH, ROCSOLVER_LOG_BENCH_PATH, and/or ROCSOLVER_LOG_PROFILE_PATH.
******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_begin(void);

/*! \brief LOG_END ends the multi-level rocSOLVER logging session.

    \details
    If applicable, this function also prints the profile logging results
    before cleaning the logging environment.
*****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_end(void);

/*! \brief LOG_SET_LAYER_MODE sets the logging mode for the rocSOLVER multi-level
    logging environment.

    \details
    @param[in]
    layer_mode      rocblas_layer_mode_flags.\n
                    Specifies the logging mode.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_set_layer_mode(const rocblas_layer_mode_flags layer_mode);

/*! \brief LOG_SET_MAX_LEVELS sets the maximum trace log depth for the rocSOLVER
    multi-level logging environment.

    \details
    @param[in]
    max_levels      rocblas_int. max_levels >= 1.\n
                    Specifies the maximum depth at which nested function calls
                    will appear in the trace and profile logs.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_set_max_levels(const rocblas_int max_levels);

/*! \brief LOG_RESTORE_DEFAULTS restores the default values of the rocSOLVER
    multi-level logging environment.

    \details
    This function sets the logging mode and maximum trace log depth to their
    default values (no logging and one level depth).
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_restore_defaults(void);

/*! \brief LOG_WRITE_PROFILE prints the profile logging results.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_write_profile(void);

/*! \brief LOG_FLUSH_PROFILE prints the profile logging results and clears the
    profile record.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_flush_profile(void);

/*
 * ===========================================================================
 *      Auxiliary functions
 * ===========================================================================
 */

/*! @{
    \brief LACGV conjugates the complex vector x.

    \details
    It conjugates the n entries of a complex vector x with increment incx.

    @param[in]
    handle          rocblas_handle
    @param[in]
    n               rocblas_int. n >= 0.\n
                    The number of entries of the vector x.
    @param[inout]
    x               pointer to type. Array on the GPU of size at least n.\n
                    On input it is the vector x,
                    on output it is overwritten with vector conjg(x).
    @param[in]
    incx            rocblas_int. incx != 0.\n
                    The increment between consecutive elements of x.
                    If incx is negative, the elements of x are indexed in
                    reverse order.
    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_clacgv(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* x,
                                                 const rocblas_int incx);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlacgv(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* x,
                                                 const rocblas_int incx);
//! @}

/*! @{
    \brief LASWP performs a series of row interchanges on the matrix A.

    \details
    It interchanges row I with row IPIV[k1 + (I - k1) * abs(inx)], for
    each of rows K1 through K2 of A. k1 and k2 are 1-based indices.

    @param[in]
    handle          rocblas_handle
    @param[in]
    n               rocblas_int. n >= 0.\n
                    The number of columns of the matrix A.
    @param[inout]
    A               pointer to type. Array on the GPU of dimension lda*n. \n
                    On entry, the matrix of column dimension n to which the row
                    interchanges will be applied. On exit, the permuted matrix.
    @param[in]
    lda             rocblas_int. lda > 0.\n
                    The leading dimension of the array A.
    @param[in]
    k1              rocblas_int. k1 > 0.\n
                    The first element of IPIV for which a row interchange will
                    be done. This is a 1-based index.
    @param[in]
    k2              rocblas_int. k2 > k1 > 0.\n
                    (K2-K1+1) is the number of elements of IPIV for which a row
                    interchange will be done. This is a 1-based index.
    @param[in]
    ipiv            pointer to rocblas_int. Array on the GPU of dimension at least k1 + (k2 - k1) * abs(incx).\n
                    The vector of pivot indices.  Only the elements in positions
                    k1 through (k1 + (k2 - k1) * abs(incx)) of IPIV are accessed.
                    Elements of ipiv are considered 1-based.
    @param[in]
    incx            rocblas_int. incx != 0.\n
                    The increment between successive values of IPIV.  If IPIV
                    is negative, the pivots are applied in reverse order.
    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slaswp(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int k1,
                                                 const rocblas_int k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int incx);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlaswp(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int k1,
                                                 const rocblas_int k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int incx);

ROCSOLVER_EXPORT rocblas_status rocsolver_claswp(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int k1,
                                                 const rocblas_int k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int incx);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlaswp(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int k1,
                                                 const rocblas_int k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int incx);
//! @}

/*! @{
    \brief LARFG generates an orthogonal Householder reflector H of order n.

    \details
    Householder reflector H is such that

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
    n               rocblas_int. n >= 0.\n
                    The order (size) of reflector H.
    @param[inout]
    alpha           pointer to type. A scalar on the GPU.\n
                    On input the scalar alpha,
                    on output it is overwritten with beta.
    @param[inout]
    x               pointer to type. Array on the GPU of size at least n-1.\n
                    On input it is the vector x,
                    on output it is overwritten with vector v.
    @param[in]
    incx            rocblas_int. incx > 0.\n
                    The increment between consecutive elements of x.
    @param[out]
    tau             pointer to type. A scalar on the GPU.\n
                    The scalar tau.

    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* alpha,
                                                 float* x,
                                                 const rocblas_int incx,
                                                 float* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* alpha,
                                                 double* x,
                                                 const rocblas_int incx,
                                                 double* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_clarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* alpha,
                                                 rocblas_float_complex* x,
                                                 const rocblas_int incx,
                                                 rocblas_float_complex* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarfg(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* alpha,
                                                 rocblas_double_complex* x,
                                                 const rocblas_int incx,
                                                 rocblas_double_complex* tau);
//! @}

/*! @{
    \brief LARFT Generates the triangular factor T of a block reflector H of
    order n.

    \details
    The block reflector H is defined as the product of k Householder matrices as

        H = H(1) * H(2) * ... * H(k)  (forward direction), or
        H = H(k) * ... * H(2) * H(1)  (backward direction)

    depending on the value of direct.

    The triangular matrix T is upper triangular in forward direction and lower triangular in backward direction.
    If storev is column-wise, then

        H = I - V * T * V'

    where the i-th column of matrix V contains the Householder vector associated to H(i). If storev is row-wise, then

        H = I - V' * T * V

    where the i-th row of matrix V contains the Householder vector associated to H(i).

    @param[in]
    handle              rocblas_handle.
    @param[in]
    direct              #rocblas_direct.\n
                        Specifies the direction in which the Householder matrices are applied.
    @param[in]
    storev              #rocblas_storev.\n
                        Specifies how the Householder vectors are stored in matrix V.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        The order (size) of the block reflector.
    @param[in]
    k                   rocblas_int. k >= 1.\n
                        The number of Householder matrices.
    @param[in]
    V                   pointer to type. Array on the GPU of size ldv*k if column-wise, or ldv*n if row-wise.\n
                        The matrix of Householder vectors.
    @param[in]
    ldv                 rocblas_int. ldv >= n if column-wise, or ldv >= k if row-wise.\n
                        Leading dimension of V.
    @param[in]
    tau                 pointer to type. Array of k scalars on the GPU.\n
                        The vector of all the scalars associated to the Householder matrices.
    @param[out]
    T                   pointer to type. Array on the GPU of dimension ldt*k.\n
                        The triangular factor. T is upper triangular is forward operation, otherwise it is lower triangular.
                        The rest of the array is not used.
    @param[in]
    ldt                 rocblas_int. ldt >= k.\n
                        The leading dimension of T.

    **************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* tau,
                                                 float* T,
                                                 const rocblas_int ldt);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* tau,
                                                 double* T,
                                                 const rocblas_int ldt);

ROCSOLVER_EXPORT rocblas_status rocsolver_clarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_float_complex* tau,
                                                 rocblas_float_complex* T,
                                                 const rocblas_int ldt);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_double_complex* tau,
                                                 rocblas_double_complex* T,
                                                 const rocblas_int ldt);
//! @}

/*! @{
    \brief LARF applies a Householder reflector H to a general matrix A.

    \details
    The Householder reflector H, of order m (or n), is to be applied to a m-by-n matrix A
    from the left (or the right). H is given by

        H = I - alpha * x * x'

    where alpha is a scalar and x a Householder vector. H is never actually computed.

    @param[in]
    handle          rocblas_handle.
    @param[in]
    side            rocblas_side.\n
                    If side = rocblas_side_left, then compute H*A
                    If side = rocblas_side_right, then compute A*H
    @param[in]
    m               rocblas_int. m >= 0.\n
                    Number of rows of A.
    @param[in]
    n               rocblas_int. n >= 0.\n
                    Number of columns of A.
    @param[in]
    x               pointer to type. Array on the GPU of
                    size at least (1 + (m-1)*abs(incx)) if left side, or
                    at least (1 + (n-1)*abs(incx)) if right side.\n
                    The Householder vector x.
    @param[in]
    incx            rocblas_int. incx != 0.\n
                    Increment between to consecutive elements of x.
                    If incx < 0, the elements of x are used in reverse order.
    @param[in]
    alpha           pointer to type. A scalar on the GPU.\n
                    If alpha = 0, then H = I (A will remain the same, x is never used)
    @param[inout]
    A               pointer to type. Array on the GPU of size lda*n.\n
                    On input, the matrix A. On output it is overwritten with
                    H*A (or A*H).
    @param[in]
    lda             rocblas_int. lda >= m.\n
                    Leading dimension of A.

    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarf(rocblas_handle handle,
                                                const rocblas_side side,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                float* x,
                                                const rocblas_int incx,
                                                const float* alpha,
                                                float* A,
                                                const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarf(rocblas_handle handle,
                                                const rocblas_side side,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                double* x,
                                                const rocblas_int incx,
                                                const double* alpha,
                                                double* A,
                                                const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_clarf(rocblas_handle handle,
                                                const rocblas_side side,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_float_complex* x,
                                                const rocblas_int incx,
                                                const rocblas_float_complex* alpha,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarf(rocblas_handle handle,
                                                const rocblas_side side,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* x,
                                                const rocblas_int incx,
                                                const rocblas_double_complex* alpha,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda);
//! @}

/*! @{
    \brief LARFB applies a block reflector H to a general m-by-n matrix A.

    \details
    The block reflector H is applied in one of the following forms, depending on
    the values of side and trans:

        H  * A  (No transpose from the left)
        H' * A  (Transpose or conjugate transpose from the left)
        A * H   (No transpose from the right), and
        A * H'  (Transpose or conjugate transpose from the right)

    The block reflector H is defined as the product of k Householder matrices as

        H = H(1) * H(2) * ... * H(k)  (forward direction), or
        H = H(k) * ... * H(2) * H(1)  (backward direction)

    depending on the value of direct. H is never stored. It is calculated as

        H = I - V * T * V'

    where the i-th column of matrix V contains the Householder vector associated with H(i), if storev is column-wise; or

        H = I - V' * T * V

    where the i-th row of matrix V contains the Householder vector associated with H(i), if storev is row-wise.
    T is the associated triangular factor as computed by LARFT.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply H.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the block reflector or its transpose/conjugate transpose is to be applied.
    @param[in]
    direct              #rocblas_direct.\n
                        Specifies the direction in which the Householder matrices were to be applied to generate H.
    @param[in]
    storev              #rocblas_storev.\n
                        Specifies how the Householder vectors are stored in matrix V.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix A.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix A.
    @param[in]
    k                   rocblas_int. k >= 1.\n
                        The number of Householder matrices.
    @param[in]
    V                   pointer to type. Array on the GPU of size ldv*k if column-wise, ldv*n if row-wise and applying from the right,
                        or ldv*m if row-wise and applying from the left.\n
                        The matrix of Householder vectors.
    @param[in]
    ldv                 rocblas_int. ldv >= k if row-wise, ldv >= m if column-wise and applying from the left, or ldv >= n if
                        column-wise and applying from the right.\n
                        Leading dimension of V.
    @param[in]
    T                   pointer to type. Array on the GPU of dimension ldt*k.\n
                        The triangular factor of the block reflector.
    @param[in]
    ldt                 rocblas_int. ldt >= k.\n
                        The leading dimension of T.
    @param[inout]
    A                   pointer to type. Array on the GPU of size lda*n.\n
                        On input, the matrix A. On output it is overwritten with
                        H*A, A*H, H'*A, or A*H'.
    @param[in]
    lda                 rocblas_int. lda >= m.\n
                        Leading dimension of A.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* T,
                                                 const rocblas_int ldt,
                                                 float* A,
                                                 const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* T,
                                                 const rocblas_int ldt,
                                                 double* A,
                                                 const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_clarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_float_complex* T,
                                                 const rocblas_int ldt,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_double_complex* T,
                                                 const rocblas_int ldt,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda);
//! @}

/*! @{
    \brief LABRD computes the bidiagonal form of the first k rows and columns of
    a general m-by-n matrix A, as well as the matrices X and Y needed to reduce
    the remaining part of A.

    \details
    The bidiagonal form is given by:

        B = Q' * A * P

    where B is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

        Q = H(1) * H(2) * ... *  H(k)  and P = G(1) * G(2) * ... * G(k-1), if m >= n, or
        Q = H(1) * H(2) * ... * H(k-1) and P = G(1) * G(2) * ... *  G(k),  if m < n

    Each Householder matrix H(i) and G(i) is given by

        H(i) = I - tauq[i-1] * v(i) * v(i)', and
        G(i) = I - taup[i-1] * u(i) * u(i)'

    If m >= n, the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1;
    while the first i elements of the Householder vector u(i) are zero, and u(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u(i) are zero, and u(i)[i] = 1.

    The unreduced part of the matrix A can be updated using a block update:

        A = A - V * Y' - X * U'

    where V is an m-by-k matrix and U is an n-by-k formed using the vectors v and u.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[in]
    k         rocblas_int. min(m,n) >= k >= 0.\n
              The number of leading rows and columns of the matrix A to be reduced.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension k.\n
              The diagonal elements of B.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension k.\n
              The off-diagonal elements of B.
    @param[out]
    tauq      pointer to type. Array on the GPU of dimension k.\n
              The scalar factors of the Householder matrices H(i).
    @param[out]
    taup      pointer to type. Array on the GPU of dimension k.\n
              The scalar factors of the Householder matrices G(i).
    @param[out]
    X         pointer to type. Array on the GPU of dimension ldx*k.\n
              The m-by-k matrix needed to reduce the unreduced part of A.
    @param[in]
    ldx       rocblas_int. ldx >= m.\n
              specifies the leading dimension of X.
    @param[out]
    Y         pointer to type. Array on the GPU of dimension ldy*k.\n
              The n-by-k matrix needed to reduce the unreduced part of A.
    @param[in]
    ldy       rocblas_int. ldy >= n.\n
              specifies the leading dimension of Y.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slabrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 float* tauq,
                                                 float* taup,
                                                 float* X,
                                                 const rocblas_int ldx,
                                                 float* Y,
                                                 const rocblas_int ldy);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlabrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 double* tauq,
                                                 double* taup,
                                                 double* X,
                                                 const rocblas_int ldx,
                                                 double* Y,
                                                 const rocblas_int ldy);

ROCSOLVER_EXPORT rocblas_status rocsolver_clabrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* tauq,
                                                 rocblas_float_complex* taup,
                                                 rocblas_float_complex* X,
                                                 const rocblas_int ldx,
                                                 rocblas_float_complex* Y,
                                                 const rocblas_int ldy);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlabrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* X,
                                                 const rocblas_int ldx,
                                                 rocblas_double_complex* Y,
                                                 const rocblas_int ldy);
//! @}

/*! @{
    \brief LATRD computes the tridiagonal form of k rows and columns of
    a symmetric/hermitian matrix A, as well as the matrix W needed to update
    the remaining part of A.

    \details
    The reduced form is given by:

        T = Q' * A * Q

    If uplo is lower, the first k rows and columns of T form a tridiagonal block, if uplo is upper, then the last
    k rows and columns of T form the tridiagonal block. Q is an orthogonal/unitary matrix represented as the
    product of Householder matrices

        Q = H(1) * H(2) * ... *  H(k)  if uplo indicates lower, or
        Q = H(n-1) * H(n-2) * ... * H(n-k) if uplo is upper.

    Each Householder matrix H(i) is given by

        H(i) = I - tau[i] * v(i) * v(i)'

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1. If uplo is upper,
    the last n-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    The unreduced part of the matrix A can be updated using a rank update of the form:

        A = A - V * W' - W * V'

    where V is an n-by-k matrix formed by the vectors v(i).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrix A is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[in]
    k         rocblas_int. 0 <= k <= n.\n
              The number of rows and columns of the matrix A to be reduced.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the n-by-n matrix to be reduced.
              On exit, if uplo is lower, the first k columns have been reduced to tridiagonal form
              (given in the diagonal elements of A and the array E), the elements below the diagonal
              contain the vectors v(i) stored as columns.
              If uplo is upper, the last k columns have been reduced to tridiagonal form
              (given in the diagonal elements of A and the array E), the elements above the diagonal
              contain the vectors v(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension n-1.\n
              If upper (lower), the last (first) k elements of E are the off-diagonal elements of the
              computed tridiagonal block.
    @param[out]
    tau       pointer to type. Array on the GPU of dimension n-1.\n
              If upper (lower), the last (first) k elements of tau are the scalar factors of the Householder
              matrices H(i).
    @param[out]
    W         pointer to type. Array on the GPU of dimension ldw*k.\n
              The n-by-k matrix needed to update the unreduced part of A.
    @param[in]
    ldw       rocblas_int. ldw >= n.\n
              specifies the leading dimension of W.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slatrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* E,
                                                 float* tau,
                                                 float* W,
                                                 const rocblas_int ldw);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlatrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* E,
                                                 double* tau,
                                                 double* W,
                                                 const rocblas_int ldw);

ROCSOLVER_EXPORT rocblas_status rocsolver_clatrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* E,
                                                 rocblas_float_complex* tau,
                                                 rocblas_float_complex* W,
                                                 const rocblas_int ldw);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlatrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* E,
                                                 rocblas_double_complex* tau,
                                                 rocblas_double_complex* W,
                                                 const rocblas_int ldw);
//! @}

/*! @{
    \brief ORG2R generates a m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GEQRF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k columns of matrix A of GEQRF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQRF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorg2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorg2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNG2R generates a m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GEQRF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k columns of matrix A of GEQRF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQRF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cung2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zung2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGQR generates a m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GEQRF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k columns of matrix A of GEQRF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQRF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGQR generates a m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GEQRF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k columns of matrix A of GEQRF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQRF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cungqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zungqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGL2 generates a m-by-n Matrix Q with orthonormal rows.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GELQF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. 0 <= m <= n.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= m.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th row has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k rows of matrix A of GELQF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GELQF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgl2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgl2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGL2 generates a m-by-n complex Matrix Q with orthonormal rows.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

        Q = H(k)**H * H(k-1)**H * ... * H(1)**H

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GELQF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. 0 <= m <= n.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= m.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th row has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k rows of matrix A of GELQF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GELQF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cungl2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zungl2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGLQ generates a m-by-n Matrix Q with orthonormal rows.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GELQF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. 0 <= m <= n.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= m.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th row has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k rows of matrix A of GELQF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GELQF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorglq(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorglq(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGLQ generates a m-by-n complex Matrix Q with orthonormal rows.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

        Q = H(k)**H * H(k-1)**H * ... * H(1)**H

    Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vector v(i) and scalar ipiv_i as returned by GELQF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. 0 <= m <= n.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= m.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th row has Householder vector v(i), for i = 1,2,...,k
                as returned in the first k rows of matrix A of GELQF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GELQF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunglq(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunglq(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORG2L generates a m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vector v(i) and scalar ipiv_i as returned by GEQLF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (n-k+i)-th column has Householder vector v(i), for
                i = 1,2,...,k as returned in the last k columns of matrix A of GEQLF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQLF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorg2l(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorg2l(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNG2L generates a m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vector v(i) and scalar ipiv_i as returned by GEQLF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (n-k+i)-th column has Householder vector v(i), for
                i = 1,2,...,k as returned in the last k columns of matrix A of GEQLF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQLF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cung2l(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zung2l(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGQL generates a m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the last n column of the product of k Householder
    reflectors of order m

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vector v(i) and scalar ipiv_i as returned by GEQLF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (n-k+i)-th column has Householder vector v(i), for
                i = 1,2,...,k as returned in the last k columns of matrix A of GEQLF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQLF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgql(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgql(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGQL generates a m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

        Q = H(k) * H(k-1) * ... * H(1)

    Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vector v(i) and scalar ipiv_i as returned by GEQLF.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
    @param[in]
    n           rocblas_int. 0 <= n <= m.\n
                The number of columns of the matrix Q.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of Householder reflectors.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (n-k+i)-th column has Householder vector v(i), for
                i = 1,2,...,k as returned in the last k columns of matrix A of GEQLF.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The scalar factors of the Householder matrices H(i) as returned by GEQLF.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cungql(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zungql(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGBR generates a m-by-n Matrix Q with orthonormal rows or columns.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns. If m >= k, Q is defined as the first
    n columns of the product of k Householder reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    If m < k, Q is defined as the product of Householder reflectors of order m

        Q = H(1) * H(2) * ... * H(m-1)

    On the other hand, if storev is row-wise, then the matrix Q has orthonormal rows. If n > k, Q is defined as the
    first m rows of the product of k Householder reflectors of order n

        Q = H(k) * H(k-1) * ... * H(1)

    If n <= k, Q is defined as the product of Householder reflectors of order n

        Q = H(n-1) * H(n-2) * ... * H(1)

    The Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vectors v(i) and scalars ipiv_i as returned by GEBRD in its arguments A and tauq or taup.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies whether to work column-wise or row-wise.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
                If row-wise, then min(n,k) <= m <= n.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
                If column-wise, then min(m,k) <= n <= m.
    @param[in]
    k           rocblas_int. k >= 0.\n
                The number of columns (if storev is colum-wise) or rows (if row-wise) of the
                original matrix reduced by GEBRD.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column (or row) has the Householder vector v(i)
                as returned by GEBRD.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension min(m,k) if column-wise, or min(n,k) if row-wise.\n
                The scalar factors of the Householder matrices H(i) as returned by GEBRD.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGBR generates a m-by-n complex Matrix Q with orthonormal rows or
    columns.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns. If m >= k, Q is defined as the first
    n columns of the product of k Householder reflectors of order m

        Q = H(1) * H(2) * ... * H(k)

    If m < k, Q is defined as the product of Householder reflectors of order m

        Q = H(1) * H(2) * ... * H(m-1)

    On the other hand, if storev is row-wise, then the matrix Q has orthonormal rows. If n > k, Q is defined as the
    first m rows of the product of k Householder reflectors of order n

        Q = H(k) * H(k-1) * ... * H(1)

    If n <= k, Q is defined as the product of Householder reflectors of order n

        Q = H(n-1) * H(n-2) * ... * H(1)

    The Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vectors v(i) and scalars ipiv_i as returned by GEBRD in its arguments A and tauq or taup.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies whether to work column-wise or row-wise.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix Q.
                If row-wise, then min(n,k) <= m <= n.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix Q.
                If column-wise, then min(m,k) <= n <= m.
    @param[in]
    k           rocblas_int. k >= 0.\n
                The number of columns (if storev is colum-wise) or rows (if row-wise) of the
                original matrix reduced by GEBRD.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the i-th column (or row) has the Householder vector v(i)
                as returned by GEBRD.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension min(m,k) if column-wise, or min(n,k) if row-wise.\n
                The scalar factors of the Householder matrices H(i) as returned by GEBRD.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cungbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zungbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORGTR generates a n-by-n orthogonal Matrix Q.

    \details
    Q is defined as the product of n-1 Householder reflectors of order n. If
    uplo indicates upper, then Q has the form

        Q = H(n-1) * H(n-2) * ... * H(1)

    On the other hand, if uplo indicates lower, then Q has the form

        Q = H(1) * H(2) * ... * H(n-1)

    The Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vectors v(i) and scalars ipiv_i as returned by
    SYTRD in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the SYTRD factorization was upper or lower
                triangular. If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix Q.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (i+1)-th column (if uplo indicates upper) or i-th
                column (if uplo indicates lower) has the Householder vector v(i) as returned
                by SYTRD. On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension n-1.\n
                The scalar factors of the Householder
                matrices H(i) as returned by SYTRD.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgtr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgtr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);
//! @}

/*! @{
    \brief UNGTR generates a n-by-n unitary Matrix Q.

    \details
    Q is defined as the product of n-1 Householder reflectors of order n. If
    uplo indicates upper, then Q has the form

        Q = H(n-1) * H(n-2) * ... * H(1)

    On the other hand, if uplo indicates lower, then Q has the form

        Q = H(1) * H(2) * ... * H(n-1)

    The Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vectors v(i) and scalars ipiv_i as returned by
    HETRD in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the HETRD factorization was upper or lower
                triangular. If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix Q.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the (i+1)-th column (if uplo indicates upper) or i-th
                column (if uplo indicates lower) has the Householder vector v(i) as returned
                by HETRD. On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension n-1.\n
                The scalar factors of the Householder
                matrices H(i) as returned by HETRD.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cungtr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zungtr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief ORM2R applies a matrix Q with orthonormal columns to a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors as

        Q = H(1) * H(2) * ... * H(k)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization GEQRF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i) associated with H(i) as returned by GEQRF
                        in the first k columns of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GEQRF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorm2r(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorm2r(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNM2R applies a complex matrix Q with orthonormal columns to a
    general m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(1) * H(2) * ... * H(k)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization GEQRF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i) associated with H(i) as returned by GEQRF
                        in the first k columns of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GEQRF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunm2r(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunm2r(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORMQR applies a matrix Q with orthonormal columns to a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors as

        Q = H(1) * H(2) * ... * H(k)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization GEQRF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i) associated with H(i) as returned by GEQRF
                        in the first k columns of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GEQRF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sormqr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dormqr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNMQR applies a complex matrix Q with orthonormal columns to a
    general m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(1) * H(2) * ... * H(k)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization GEQRF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i) associated with H(i) as returned by GEQRF
                        in the first k columns of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GEQRF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmqr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmqr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORML2 applies a matrix Q with orthonormal rows to a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization GELQF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                        The i-th row has the Householder vector v(i) associated with H(i) as returned by GELQF
                        in the first k rows of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= k. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GELQF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNML2 applies a complex matrix Q with orthonormal rows to a general
    m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(k)**H * H(k-1)**H * ... * H(1)**H

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization GELQF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                        The i-th row has the Householder vector v(i) associated with H(i) as returned by GELQF
                        in the first k rows of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= k. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GELQF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORMLQ applies a matrix Q with orthonormal rows to a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization GELQF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                        The i-th row has the Householder vector v(i) associated with H(i) as returned by GELQF
                        in the first k rows of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= k. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GELQF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sormlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dormlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNMLQ applies a complex matrix Q with orthonormal rows to a general
    m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(k)**H * H(k-1)**H * ... * H(1)**H

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization GELQF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                        The i-th row has the Householder vector v(i) associated with H(i) as returned by GELQF
                        in the first k rows of its argument A.
    @param[in]
    lda                 rocblas_int. lda >= k. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by GELQF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORM2L applies a matrix Q with orthonormal columns to a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors
    as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization GEQLF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be
                        applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i)
                        associated with H(i) as returned by GEQLF in the last k columns of its
                        argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        GEQLF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sorm2l(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dorm2l(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNM2L applies a complex matrix Q with orthonormal columns to a
    general m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization GEQLF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate
                        transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i)
                        associated with H(i) as returned by GEQLF in the last k columns of its
                        argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        GEQLF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunm2l(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunm2l(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORMQL applies a matrix Q with orthonormal columns to a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    Q is an orthogonal matrix defined as the product of k Householder reflectors
    as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization GEQLF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be
                        applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i)
                        associated with H(i) as returned by GEQLF in the last k columns of its
                        argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        GEQLF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sormql(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dormql(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNMQL applies a complex matrix Q with orthonormal columns to a
    general m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    Q is a unitary matrix defined as the product of k Householder reflectors as

        Q = H(k) * H(k-1) * ... * H(1)

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization GEQLF.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate
                        transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                        The number of Householder reflectors that form Q.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*k.\n
                        The i-th column has the Householder vector v(i)
                        associated with H(i) as returned by GEQLF in the last k columns of its
                        argument A.
    @param[in]
    lda                 rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least k.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        GEQLF.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmql(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmql(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORMBR applies a matrix Q with orthonormal rows or columns to a
    general m-by-n matrix C.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns.
    If storev is row-wise, then the matrix Q has orthonormal rows.
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    The order nq of orthogonal matrix Q is nq = m if applying from the left, or nq = n if applying from the right.

    When storev is column-wise, if nq >= k, then Q is defined as the product of k Householder reflectors of order nq

        Q = H(1) * H(2) * ... * H(k),

    and if nq < k, then Q is defined as the product

        Q = H(1) * H(2) * ... * H(nq-1).

    When storev is row-wise, if nq > k, then Q is defined as the product of k Householder reflectors of order nq

        Q = H(1) * H(2) * ... * H(k),

    and if n <= k, Q is defined as the product

        Q = H(1) * H(2) * ... * H(nq-1)

    The Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vectors v(i) and scalars ipiv_i as returned by GEBRD in its arguments A and tauq or taup.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    storev              #rocblas_storev.\n
                        Specifies whether to work column-wise or row-wise.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0.\n
                        The number of columns (if storev is colum-wise) or rows (if row-wise) of the
                        original matrix reduced by GEBRD.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*min(nq,k) if column-wise, or lda*nq if row-wise.\n
                        The i-th column (or row) has the Householder vector v(i) associated with H(i) as returned by GEBRD.
    @param[in]
    lda                 rocblas_int. lda >= nq if column-wise, or lda >= min(nq,k) if row-wise. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least min(nq,k).\n
                        The scalar factors of the Householder matrices H(i) as returned by GEBRD.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sormbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dormbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNMBR applies a complex matrix Q with orthonormal rows or columns to
    a general m-by-n matrix C.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns.
    If storev is row-wise, then the matrix Q has orthonormal rows.
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    The order nq of unitary matrix Q is nq = m if applying from the left, or nq = n if applying from the right.

    When storev is column-wise, if nq >= k, then Q is defined as the product of k Householder reflectors of order nq

        Q = H(1) * H(2) * ... * H(k),

    and if nq < k, then Q is defined as the product

        Q = H(1) * H(2) * ... * H(nq-1).

    When storev is row-wise, if nq > k, then Q is defined as the product of k Householder reflectors of order nq

        Q = H(1) * H(2) * ... * H(k),

    and if n <= k, Q is defined as the product

        Q = H(1) * H(2) * ... * H(nq-1)

    The Householder matrices H(i) are never stored, they are computed from its corresponding
    Householder vectors v(i) and scalars ipiv_i as returned by GEBRD in its arguments A and tauq or taup.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    storev              #rocblas_storev.\n
                        Specifies whether to work column-wise or row-wise.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    k                   rocblas_int. k >= 0.\n
                        The number of columns (if storev is colum-wise) or rows (if row-wise) of the
                        original matrix reduced by GEBRD.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*min(nq,k) if column-wise, or lda*nq if row-wise.\n
                        The i-th column (or row) has the Householder vector v(i) associated with H(i) as returned by GEBRD.
    @param[in]
    lda                 rocblas_int. lda >= nq if column-wise, or lda >= min(nq,k) if row-wise. \n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least min(nq,k).\n
                        The scalar factors of the Householder matrices H(i) as returned by GEBRD.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmbr(rocblas_handle handle,
                                                 const rocblas_storev storev,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief ORMTR applies an orthogonal matrix Q to a general m-by-n matrix C.

    \details
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Transpose from the right)

    The order nq of orthogonal matrix Q is nq = m if applying from the left, or
    nq = n if applying from the right.

    Q is defined as the product of nq-1 Householder reflectors of order nq. If
    uplo indicates upper, then Q has the form

        Q = H(nq-1) * H(nq-2) * ... * H(1).

    On the other hand, if uplo indicates lower, then Q has the form

        Q = H(1) * H(2) * ... * H(nq-1)

    The Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vectors v(i) and scalars ipiv_i as returned by
    SYTRD in its arguments A and tau.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    uplo                rocblas_fill.\n
                        Specifies whether the SYTRD factorization was upper or
                        lower triangular. If uplo indicates lower (or upper), then the upper (or
                        lower) part of A is not used.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its transpose is to be
                        applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*nq.\n
                        On entry, the (i+1)-th column (if uplo indicates upper)
                        or i-th column (if uplo indicates lower) has the Householder vector v(i) as
                        returned by SYTRD.
    @param[in]
    lda                 rocblas_int. lda >= nq.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least nq-1.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        SYTRD.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sormtr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_fill uplo,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv,
                                                 float* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_dormtr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_fill uplo,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv,
                                                 double* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief UNMTR applies a unitary matrix Q to a general m-by-n matrix C.

    \details
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

        Q  * C  (No transpose from the left)
        Q' * C  (Conjugate transpose from the left)
        C * Q   (No transpose from the right), and
        C * Q'  (Conjugate transpose from the right)

    The order nq of unitary matrix Q is nq = m if applying from the left, or
    nq = n if applying from the right.

    Q is defined as the product of nq-1 Householder reflectors of order nq. If
    uplo indicates upper, then Q has the form

        Q = H(nq-1) * H(nq-2) * ... * H(1).

    On the other hand, if uplo indicates lower, then Q has the form

        Q = H(1) * H(2) * ... * H(nq-1)

    The Householder matrices H(i) are never stored, they are computed from its
    corresponding Householder vectors v(i) and scalars ipiv_i as returned by
    HETRD in its arguments A and tau.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply Q.
    @param[in]
    uplo                rocblas_fill.\n
                        Specifies whether the SYTRD factorization was upper or
                        lower triangular. If uplo indicates lower (or upper), then the upper (or
                        lower) part of A is not used.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the matrix Q or its conjugate
                        transpose is to be applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix C.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix C.
    @param[in]
    A                   pointer to type. Array on the GPU of size lda*nq.\n
                        On entry, the (i+1)-th column (if uplo indicates upper)
                        or i-th column (if uplo indicates lower) has the Householder vector v(i) as
                        returned by HETRD.
    @param[in]
    lda                 rocblas_int. lda >= nq.\n
                        Leading dimension of A.
    @param[in]
    ipiv                pointer to type. Array on the GPU of dimension at least nq-1.\n
                        The scalar factors of the Householder matrices H(i) as returned by
                        HETRD.
    @param[inout]
    C                   pointer to type. Array on the GPU of size ldc*n.\n
                        On input, the matrix C. On output it is overwritten with
                        Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc                 rocblas_int. ldc >= m.\n
                        Leading dimension of C.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmtr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_fill uplo,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc);

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmtr(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_fill uplo,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc);
//! @}

/*! @{
    \brief BDSQR computes the singular value decomposition (SVD) of a
    n-by-n bidiagonal matrix B.

    \details
    The SVD of B has the form:

        B = Ub * S * Vb'

    where S is the n-by-n diagonal matrix of singular values of B, the columns of Ub are the left
    singular vectors of B, and the columns of Vb are its right singular vectors.

    The computation of the singular vectors is optional; this function accepts input matrices
    U (of size nu-by-n) and V (of size n-by-nv) that are overwritten with U*Ub and Vb'*V. If nu = 0
    no left vectors are computed; if nv = 0 no right vectors are computed.

    Optionally, this function can also compute Ub'*C for a given n-by-nc input matrix C.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether B is upper or lower bidiagonal.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix B.
    @param[in]
    nv          rocblas_int. nv >= 0.\n
                The number of columns of matrix V.
    @param[in]
    nu          rocblas_int. nu >= 0.\n
                The number of rows of matrix U.
    @param[in]
    nc          rocblas_int. nu >= 0.\n
                The number of columns of matrix C.
    @param[inout]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On entry, the diagonal elements of B. On exit, if info = 0,
                the singular values of B in decreasing order; if info > 0,
                the diagonal elements of a bidiagonal matrix
                orthogonally equivalent to B.
    @param[inout]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                On entry, the off-diagonal elements of B. On exit, if info > 0,
                the off-diagonal elements of a bidiagonal matrix
                orthogonally equivalent to B (if info = 0 this matrix converges to zero).
    @param[inout]
    V           pointer to type. Array on the GPU of dimension ldv*nv.\n
                On entry, the matrix V. On exit, it is overwritten with Vb'*V.
                (Not referenced if nv = 0).
    @param[in]
    ldv         rocblas_int. ldv >= n if nv > 0, or ldv >=1 if nv = 0.\n
                Specifies the leading dimension of V.
    @param[inout]
    U           pointer to type. Array on the GPU of dimension ldu*n.\n
                On entry, the matrix U. On exit, it is overwritten with U*Ub.
                (Not referenced if nu = 0).
    @param[in]
    ldu         rocblas_int. ldu >= nu.\n
                Specifies the leading dimension of U.
    @param[inout]
    C           pointer to type. Array on the GPU of dimension ldc*nc.\n
                On entry, the matrix C. On exit, it is overwritten with Ub'*C.
                (Not referenced if nc = 0).
    @param[in]
    ldc         rocblas_int. ldc >= n if nc > 0, or ldc >=1 if nc = 0.\n
                Specifies the leading dimension of C.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, i elements of E have not converged to zero.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 float* D,
                                                 float* E,
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* U,
                                                 const rocblas_int ldu,
                                                 float* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 double* D,
                                                 double* E,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* U,
                                                 const rocblas_int ldu,
                                                 double* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_float_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_double_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief STERF computes the eigenvalues of a symmetric tridiagonal matrix.

    \details
    The eigenvalues of the symmetric tridiagonal matrix are computed by the
    Pal-Walker-Kahan variant of the QL/QR algorithm, and returned in
    increasing order.

    The matrix is not represented explicitly, but rather as the array of
    diagonal elements D and the array of symmetric off-diagonal elements E
    as returned by, e.g., SYTRD or HETRD.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the tridiagonal matrix.
    @param[inout]
    D         pointer to real type. Array on the GPU of dimension n.\n
              On entry, the diagonal elements of the matrix.
              On exit, if info = 0, the eigenvalues in increasing order.
              If info > 0, the diagonal elements of a tridiagonal matrix
              that is similar to the original matrix (i.e. has the same
              eigenvalues).
    @param[inout]
    E         pointer to real type. Array on the GPU of dimension n-1.\n
              On entry, the off-diagonal elements of the matrix.
              On exit, if info = 0, this array converges to zero.
              If info > 0, the off-diagonal elements of a tridiagonal matrix
              that is similar to the original matrix (i.e. has the same
              eigenvalues).
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, STERF did not converge. i elements of E did not
              converge to zero.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssterf(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsterf(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief STEQR computes the eigenvalues and (optionally) eigenvectors of
    a symmetric tridiagonal matrix.

    \details
    The eigenvalues of the symmetric tridiagonal matrix are computed by the
    implicit QL/QR algorithm, and returned in increasing order.

    The matrix is not represented explicitly, but rather as the array of
    diagonal elements D and the array of symmetric off-diagonal elements E
    as returned by, e.g., SYTRD or HETRD. If the tridiagonal matrix is the
    reduced form of a full symmetric/Hermitian matrix as returned by, e.g.,
    SYTRD or HETRD, then the eigenvectors of the original matrix can also
    be computed, depending on the value of compC.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    compC     #rocblas_evect.\n
              Specifies how the eigenvectors are computed.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the tridiagonal matrix.
    @param[inout]
    D         pointer to real type. Array on the GPU of dimension n.\n
              On entry, the diagonal elements of the matrix.
              On exit, if info = 0, the eigenvalues in increasing order.
              If info > 0, the diagonal elements of a tridiagonal matrix
              that is similar to the original matrix (i.e. has the same
              eigenvalues).
    @param[inout]
    E         pointer to real type. Array on the GPU of dimension n-1.\n
              On entry, the off-diagonal elements of the matrix.
              On exit, if info = 0, this array converges to zero.
              If info > 0, the off-diagonal elements of a tridiagonal matrix
              that is similar to the original matrix (i.e. has the same
              eigenvalues).
    @param[inout]
    C         pointer to type. Array on the GPU of dimension ldc*n.\n
              On entry, if compC is original, the orthogonal/unitary matrix
              used for the reduction to tridiagonal form as returned by, e.g.,
              ORGTR or UNGTR.
              On exit, it is overwritten with the eigenvectors of the original
              symmetric/Hermitian matrix (if compC is original), or the
              eigenvectors of the tridiagonal matrix (if compC is tridiagonal).
              (Not referenced if compC is none).
    @param[in]
    ldc       rocblas_int. ldc >= n if compc is original or tridiagonal.\n
              Specifies the leading dimension of C.
              (Not referenced if compC is none).
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, STEQR did not converge. i elements of E did not
              converge to zero.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssteqr(rocblas_handle handle,
                                                 const rocblas_evect compC,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 float* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsteqr(rocblas_handle handle,
                                                 const rocblas_evect compC,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 double* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_csteqr(rocblas_handle handle,
                                                 const rocblas_evect compC,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsteqr(rocblas_handle handle,
                                                 const rocblas_evect compC,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);
//! @}

/*
 * ===========================================================================
 *      LAPACK functions
 * ===========================================================================
 */

/*! @{
    \brief GETF2_NPVT computes the LU factorization of a general m-by-n matrix A
    without partial pivoting.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization has the form

        A = L * U

    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETF2 routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      float* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      double* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);
//! @}

/*! @{
    \brief GETF2_NPVT_BATCHED computes the LU factorization of a batch of
    general m-by-n matrices without partial pivoting.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = L_i * U_i

    where L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETF2 routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              float* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              double* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              rocblas_float_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              rocblas_double_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETF2_NPVT_STRIDED_BATCHED computes the LU factorization of a batch
    of general m-by-n matrices without partial pivoting.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = L_i * U_i

    where L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETF2 routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      float* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      double* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      rocblas_float_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      rocblas_double_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRF_NPVT computes the LU factorization of a general m-by-n matrix A
    without partial pivoting.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization has the form

        A = L * U

    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETRF routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      float* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      double* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_npvt(rocblas_handle handle,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);
//! @}

/*! @{
    \brief GETRF_NPVT_BATCHED computes the LU factorization of a batch of
    general m-by-n matrices without partial pivoting.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = L_i * U_i

    where L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETRF routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              float* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              double* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              rocblas_float_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              rocblas_double_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRF_NPVT_STRIDED_BATCHED computes the LU factorization of a batch
    of general m-by-n matrices without partial pivoting.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = L_i * U_i

    where L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API GETRF routines instead.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero element in the diagonal. The factorization from
              this point might be incomplete.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      float* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      double* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      rocblas_float_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int m,
                                                                      const rocblas_int n,
                                                                      rocblas_double_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETF2 computes the LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization has the form

        A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
              The vector of pivot indices. Elements of ipiv are 1-based indices.
              For 1 <= i <= min(m,n), the row i of the
              matrix was interchanged with row ipiv[i].
              Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief GETF2_BATCHED computes the LU factorization of a batch of general
    m-by-n matrices using partial pivoting with row interchanges.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).


    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i[j].
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETF2_STRIDED_BATCHED computes the LU factorization of a batch of
    general m-by-n matrices using partial pivoting with row interchanges.

    \details
    (This is the unblocked Level-2-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i[j].
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRF computes the LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization has the form

        A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
              The vector of pivot indices. Elements of ipiv are 1-based indices.
              For 1 <= i <= min(m,n), the row i of the
              matrix was interchanged with row ipiv[i].
              Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief GETRF_BATCHED computes the LU factorization of a batch of general
    m-by-n matrices using partial pivoting with row interchanges.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRF_STRIDED_BATCHED computes the LU factorization of a batch of
    general m-by-n matrices using partial pivoting with row interchanges.

    \details
    (This is the blocked Level-3-BLAS version of the algorithm. An optimized internal implementation without rocBLAS calls
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details see the
    section "tuning rocSOLVER performance" on the User's guide).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQR2 computes a QR factorization of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

        A =  Q * [ R ]
                 [ 0 ]

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(1) * H(2) * ... * H(k), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R; the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqr2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqr2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GEQR2_BATCHED computes the QR factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [ R_j ]
                     [  0  ]

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqr2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqr2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQR2_STRIDED_BATCHED computes the QR factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [ R_j ]
                     [  0  ]

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqr2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqr2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQL2 computes a QL factorization of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

        A =  Q * [ 0 ]
                 [ L ]

    where L is lower triangular (lower trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(k) * ... * H(2) * H(1), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the last m-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L; the elements above the sub/superdiagonal are the i - 1
              elements of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeql2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeql2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeql2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeql2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GEQL2_BATCHED computes the QL factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [  0  ]
                     [ L_j ]

    where L_j is lower triangular (lower trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * ... * H_j(2) * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the last m-i elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L_j; the elements above the sub/superdiagonal are the i - 1
              elements of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeql2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeql2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeql2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeql2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQL2_STRIDED_BATCHED computes the QL factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [  0  ]
                     [ L_j ]

    where L_j is lower triangular (lower trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * ... * H_j(2) * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the last m-i elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L_j; the elements above the sub/superdiagonal are the i - 1
              elements of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeql2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeql2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeql2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeql2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELQ2 computes a LQ factorization of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

        A = [ L 0 ] * Q

    where L is lower triangular (lower trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(k) * H(k-1) * ... * H(1), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i)' * v(i)

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and delow the diagonal contain the
              factor L; the elements above the diagonal are the n - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GELQ2_BATCHED computes the LQ factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j = [ L_j 0 ] * Q_j

    where L_j is lower triangular (lower trapezoidal if m > n), and Q_j is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * H_j(k-1) * ... * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i)' * v_j(i)

    where the first i-1 elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the diagonal contain the
              factor L_j. The elements above the diagonal are the n - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELQ2_STRIDED_BATCHED computes the LQ factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j = [ L_j 0 ] * Q_j

    where L_j is lower triangular (lower trapezoidal if m > n), and Q_j is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * H_j(k-1) * ... * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i)' * v_j(i)

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the diagonal contain the
              factor L_j. The elements above the diagonal are the n - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQRF computes a QR factorization of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

        A =  Q * [ R ]
                 [ 0 ]

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(1) * H(2) * ... * H(k), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R; the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqrf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GEQRF_BATCHED computes the QR factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [ R_j ]
                     [  0  ]

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqrf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQRF_STRIDED_BATCHED computes the QR factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [ R_j ]
                     [  0  ]

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQLF computes a QL factorization of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

        A =  Q * [ 0 ]
                 [ L ]

    where L is lower triangular (lower trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(k) * ... * H(2) * H(1), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the last m-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L; the elements above the sub/superdiagonal are the i - 1
              elements of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqlf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqlf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqlf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqlf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GEQLF_BATCHED computes the QL factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [  0  ]
                     [ L_j ]

    where L_j is lower triangular (lower trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * ... * H_j(2) * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the last m-i elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L_j; the elements above the sub/superdiagonal are the i - 1
              elements of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqlf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqlf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqlf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqlf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEQLF_STRIDED_BATCHED computes the QL factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * [  0  ]
                     [ L_j ]

    where L_j is lower triangular (lower trapezoidal if m < n), and Q_j is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * ... * H_j(2) * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the last m-i elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the (m-n)th subdiagonal (when
              m >= n) or the (n-m)th superdiagonal (when n > m) contain the
              factor L_j; the elements above the sub/superdiagonal are the i - 1
              elements of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqlf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqlf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqlf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqlf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELQF computes a LQ factorization of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

        A = [ L 0 ] * Q

    where L is lower triangular (lower trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q = H(k) * H(k-1) * ... * H(1), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i)' * v(i)

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and below the diagonal contain the
              factor L; the elements above the diagonal are the n - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GELQF_BATCHED computes the LQ factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j = [ L_j 0 ] * Q_j

    where L_j is lower triangular (lower trapezoidal if m > n), and Q_j is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * H_j(k-1) * ... * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i)' * v_j(i)

    where the first i-1 elements of Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the diagonal contain the
              factor L_j. The elements above the diagonal are the n - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELQF_STRIDED_BATCHED computes the LQ factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j = [ L_j 0 ] * Q_j

    where L_j is lower triangular (lower trapezoidal if m > n), and Q_j is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

        Q_j = H_j(k) * H_j(k-1) * ... * H_j(1), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i)' * v_j(i)

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and below the diagonal contain the
              factor L_j. The elements above the diagonal are the n - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEBD2 computes the bidiagonal form of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The bidiagonal form is given by:

        B = Q' * A * P

    where B is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

        Q = H(1) * H(2) * ... *  H(n)  and P = G(1) * G(2) * ... * G(n-1), if m >= n, or
        Q = H(1) * H(2) * ... * H(m-1) and P = G(1) * G(2) * ... *  G(m),  if m < n

    Each Householder matrix H(i) and G(i) is given by

        H(i) = I - tauq[i-1] * v(i) * v(i)', and
        G(i) = I - taup[i-1] * u(i) * u(i)'

    If m >= n, the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1;
    while the first i elements of the Householder vector u(i) are zero, and u(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u(i) are zero, and u(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension min(m,n).\n
              The diagonal elements of B.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
              The off-diagonal elements of B.
    @param[out]
    tauq      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).
    @param[out]
    taup      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices G(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebd2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 float* tauq,
                                                 float* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebd2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 double* tauq,
                                                 double* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebd2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* tauq,
                                                 rocblas_float_complex* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebd2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup);
//! @}

/*! @{
    \brief GEBD2_BATCHED computes the bidiagonal form of a batch of general
    m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The bidiagonal form is given by:

        B_j = Q_j' * A_j * P_j

    where B_j is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q_j and
    P_j are orthogonal/unitary matrices represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n)  and P_j = G_j(1) * G_j(2) * ... * G_j(n-1), if m >= n, or
        Q_j = H_j(1) * H_j(2) * ... * H_j(m-1) and P_j = G_j(1) * G_j(2) * ... *  G_j(m),  if m < n

    Each Householder matrix H_j(i) and G_j(i), for j = 1,2,...,batch_count, is given by

        H_j(i) = I - tauq_j[i-1] * v_j(i) * v_j(i)', and
        G_j(i) = I - taup_j[i-1] * u_j(i) * u_j(i)'

    If m >= n, the first i-1 elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1;
    while the first i elements of the Householder vector u_j(i) are zero, and u_j(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u_j(i) are zero, and u_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B_j.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u_j(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v_j(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u_j(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of B_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of B_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq      pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
              Contains the vectors tauq_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideQ   rocblas_stride.\n
              Stride from the start of one vector tauq_j to the next one tauq_(j+1).
              There is no restriction for the value
              of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors taup_j of scalar factors of the
              Householder matrices G_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector taup_j to the next one taup_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebd2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         float* tauq,
                                                         const rocblas_stride strideQ,
                                                         float* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebd2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         double* tauq,
                                                         const rocblas_stride strideQ,
                                                         double* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebd2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_float_complex* tauq,
                                                         const rocblas_stride strideQ,
                                                         rocblas_float_complex* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebd2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_double_complex* tauq,
                                                         const rocblas_stride strideQ,
                                                         rocblas_double_complex* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEBD2_STRIDED_BATCHED computes the bidiagonal form of a batch of
    general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The bidiagonal form is given by:

        B_j = Q_j' * A_j * P_j

    where B_j is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q_j and
    P_j are orthogonal/unitary matrices represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n)  and P_j = G_j(1) * G_j(2) * ... * G_j(n-1), if m >= n, or
        Q_j = H_j(1) * H_j(2) * ... * H_j(m-1) and P_j = G_j(1) * G_j(2) * ... *  G_j(m),  if m < n

    Each Householder matrix H_j(i) and G_j(i), for j = 1,2,...,batch_count, is given by

        H_j(i) = I - tauq_j[i-1] * v_j(i) * v_j(i)', and
        G_j(i) = I - taup_j[i-1] * u_j(i) * u_j(i)'

    If m >= n, the first i-1 elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1;
    while the first i elements of the Householder vector u_j(i) are zero, and u_j(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u_j(i) are zero, and u_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B_j.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u_j(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v_j(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u_j(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of B_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of B_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq      pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
              Contains the vectors tauq_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideQ   rocblas_stride.\n
              Stride from the start of one vector tauq_j to the next one tauq_(j+1).
              There is no restriction for the value
              of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors taup_j of scalar factors of the
              Householder matrices G_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector taup_j to the next one taup_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 float* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 float* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 double* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 double* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_float_complex* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 rocblas_float_complex* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_double_complex* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 rocblas_double_complex* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEBRD computes the bidiagonal form of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The bidiagonal form is given by:

        B = Q' * A * P

    where B is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

        Q = H(1) * H(2) * ... *  H(n)  and P = G(1) * G(2) * ... * G(n-1), if m >= n, or
        Q = H(1) * H(2) * ... * H(m-1) and P = G(1) * G(2) * ... *  G(m),  if m < n

    Each Householder matrix H(i) and G(i) is given by

        H(i) = I - tauq[i-1] * v(i) * v(i)', and
        G(i) = I - taup[i-1] * u(i) * u(i)'

    If m >= n, the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1;
    while the first i elements of the Householder vector u(i) are zero, and u(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u(i) are zero, and u(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension min(m,n).\n
              The diagonal elements of B.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
              The off-diagonal elements of B.
    @param[out]
    tauq      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).
    @param[out]
    taup      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices G(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 float* tauq,
                                                 float* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 double* tauq,
                                                 double* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* tauq,
                                                 rocblas_float_complex* taup);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebrd(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup);
//! @}

/*! @{
    \brief GEBRD_BATCHED computes the bidiagonal form of a batch of general
    m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The bidiagonal form is given by:

        B_j = Q_j' * A_j * P_j

    where B_j is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q_j and
    P_j are orthogonal/unitary matrices represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n)  and P_j = G_j(1) * G_j(2) * ... * G_j(n-1), if m >= n, or
        Q_j = H_j(1) * H_j(2) * ... * H_j(m-1) and P_j = G_j(1) * G_j(2) * ... *  G_j(m),  if m < n

    Each Householder matrix H_j(i) and G_j(i), for j = 1,2,...,batch_count, is given by

        H_j(i) = I - tauq_j[i-1] * v_j(i) * v_j(i)', and
        G_j(i) = I - taup_j[i-1] * u_j(i) * u_j(i)'

    If m >= n, the first i-1 elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1;
    while the first i elements of the Householder vector u_j(i) are zero, and u_j(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u_j(i) are zero, and u_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B_j.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u_j(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v_j(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u_j(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of B_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of B_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq      pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
              Contains the vectors tauq_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideQ   rocblas_stride.\n
              Stride from the start of one vector tauq_j to the next one tauq_(j+1).
              There is no restriction for the value
              of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors taup_j of scalar factors of the
              Householder matrices G_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector taup_j to the next one taup_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebrd_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         float* tauq,
                                                         const rocblas_stride strideQ,
                                                         float* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebrd_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         double* tauq,
                                                         const rocblas_stride strideQ,
                                                         double* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebrd_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_float_complex* tauq,
                                                         const rocblas_stride strideQ,
                                                         rocblas_float_complex* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebrd_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_double_complex* tauq,
                                                         const rocblas_stride strideQ,
                                                         rocblas_double_complex* taup,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GEBRD_STRIDED_BATCHED computes the bidiagonal form of a batch of
    general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The bidiagonal form is given by:

        B_j = Q_j' * A_j * P_j

    where B_j is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q_j and
    P_j are orthogonal/unitary matrices represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n)  and P_j = G_j(1) * G_j(2) * ... * G_j(n-1), if m >= n, or
        Q_j = H_j(1) * H_j(2) * ... * H_j(m-1) and P_j = G_j(1) * G_j(2) * ... *  G_j(m),  if m < n

    Each Householder matrix H_j(i) and G_j(i), for j = 1,2,...,batch_count, is given by

        H_j(i) = I - tauq_j[i-1] * v_j(i) * v_j(i)', and
        G_j(i) = I - taup_j[i-1] * u_j(i) * u_j(i)'

    If m >= n, the first i-1 elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1;
    while the first i elements of the Householder vector u_j(i) are zero, and u_j(i)[i+1] = 1.
    If m < n, the first i elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1;
    while the first i-1 elements of the Householder vector u_j(i) are zero, and u_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on the diagonal and superdiagonal (if m >= n), or
              subdiagonal (if m < n) contain the bidiagonal form B_j.
              If m >= n, the elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,n, and the elements above the
              superdiagonal are the n - i - 1 elements of vector u_j(i) for i = 1,2,...,n-1.
              If m < n, the elements below the subdiagonal are the m - i - 1
              elements of vector v_j(i) for i = 1,2,...,m-1, and the elements above the
              diagonal are the n - i elements of vector u_j(i) for i = 1,2,...,m.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of B_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of B_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq      pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
              Contains the vectors tauq_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideQ   rocblas_stride.\n
              Stride from the start of one vector tauq_j to the next one tauq_(j+1).
              There is no restriction for the value
              of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors taup_j of scalar factors of the
              Householder matrices G_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector taup_j to the next one taup_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgebrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 float* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 float* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgebrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 double* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 double* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgebrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_float_complex* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 rocblas_float_complex* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgebrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_double_complex* tauq,
                                                                 const rocblas_stride strideQ,
                                                                 rocblas_double_complex* taup,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRS solves a system of n linear equations on n variables using the
    LU factorization computed by GETRF.

    \details
    It solves one of the following systems:

        A  * X = B (no transpose),
        A' * X = B (transpose),  or
        A* * X = B (conjugate transpose)

    depending on the value of trans.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of the matrix B.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                The factors L and U of the factorization A = P*L*U returned by GETRF.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by GETRF.
    @param[in,out]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of B.

   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrs(rocblas_handle handle,
                                                 const rocblas_operation trans,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int* ipiv,
                                                 float* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrs(rocblas_handle handle,
                                                 const rocblas_operation trans,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int* ipiv,
                                                 double* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrs(rocblas_handle handle,
                                                 const rocblas_operation trans,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int* ipiv,
                                                 rocblas_float_complex* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrs(rocblas_handle handle,
                                                 const rocblas_operation trans,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_int* ipiv,
                                                 rocblas_double_complex* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief GETRS_BATCHED solves a batch of systems of n linear equations on n
    variables using the LU factorization computed by GETRF_BATCHED.

    \details
    For each instance j in the batch, it solves one of the following systems:

        A_j  * X_j = B_j (no transpose),
        A_j' * X_j = B_j (transpose),  or
        A_j* * X_j = B_j (conjugate transpose)

    depending on the value of trans.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations of each instance in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by GETRF_BATCHED.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of pivot indices returned by GETRF_BATCHED.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[in,out]
    B           Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.

   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrs_batched(rocblas_handle handle,
                                                         const rocblas_operation trans,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         const rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         float* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrs_batched(rocblas_handle handle,
                                                         const rocblas_operation trans,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         const rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         double* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrs_batched(rocblas_handle handle,
                                                         const rocblas_operation trans,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         const rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_float_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrs_batched(rocblas_handle handle,
                                                         const rocblas_operation trans,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         const rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_double_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRS_STRIDED_BATCHED solves a batch of systems of n linear equations
    on n variables using the LU factorization computed by GETRF_STRIDED_BATCHED.

    \details
    For each instance j in the batch, it solves one of the following systems:

        A_j  * X_j = B_j (no transpose),
        A_j' * X_j = B_j (transpose),  or
        A_j* * X_j = B_j (conjugate transpose)

    depending on the value of trans.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations of each instance in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by GETRF_STRIDED_BATCHED.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j and the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of pivot indices returned by GETRF_STRIDED_BATCHED.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[in,out]
    B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j and the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.

   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_operation trans,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 const rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 float* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_operation trans,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 const rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 double* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_operation trans,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 const rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_float_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_operation trans,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 const rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_double_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI inverts a general n-by-n matrix A using the LU factorization
    computed by GETRF.

    \details
    The inverse is computed by solving the linear system

        inv(A) * L = inv(U)

    where L is the lower triangular factor of A with unit diagonal elements, and U is the
    upper triangular factor.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the factors L and U of the factorization A = P*L*U returned by GETRF.
              On exit, the inverse of A if info = 0; otherwise undefined.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[in]
    ipiv      pointer to rocblas_int. Array on the GPU of dimension n.\n
              The pivot indices returned by GETRF.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief GETRI_BATCHED inverts a batch of general n-by-n matrices using
    the LU factorization computed by GETRF_BATCHED.

    \details
    The inverse is computed by solving the linear system

        inv(A_j) * L_j = inv(U_j)

    where L_j is the lower triangular factor of A_j with unit diagonal elements, and U_j is the
    upper triangular factor.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the factors L_j and U_j of the factorization A = P_j*L_j*U_j returned by
              GETRF_BATCHED.
              On exit, the inverses of A_j if info_j = 0; otherwise undefined.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              The pivot indices returned by GETRF_BATCHED.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(i+j).
              There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_j = 0, successful exit for inversion of A_j.
              If info_j = i > 0, U_j is singular. U_j(i,i) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_batched(rocblas_handle handle,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_batched(rocblas_handle handle,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_batched(rocblas_handle handle,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_batched(rocblas_handle handle,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI_STRIDED_BATCHED inverts a batch of general n-by-n matrices
    using the LU factorization computed by GETRF_STRIDED_BATCHED.

    \details
    The inverse is computed by solving the linear system

        inv(A_j) * L_j = inv(U_j)

    where L_j is the lower triangular factor of A_j with unit diagonal elements, and U_j is the
    upper triangular factor.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by
              GETRF_STRIDED_BATCHED.
              On exit, the inverses of A_j if info_j = 0; otherwise undefined.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[in]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              The pivot indices returned by GETRF_STRIDED_BATCHED.
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_j = 0, successful exit for inversion of A_j.
              If info_j = i > 0, U_j is singular. U_j(i,i) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
    matrix A, and a corresponding matrix B, using the QR factorization computed by GEQRF (or the LQ
    factorization computed by GELQF).

    \details
    The problem solved by this function is either of the form

        A  * X = B (no transpose), or
        A' * X = B (transpose/conjugate transpose)

    depending on the value of trans.

    If m >= n (or n < m in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X is found minimizing

        || B - A  * X || (no transpose), or
        || B - A' * X || (transpose/conjugate transpose)

    If n < m (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X is chosen minimizing || X ||

    \note
    The current implementation only supports the overdetermined, no transpose case.
    \p rocblas_status_not_implemented will be returned if m < n, or if trans is
    \p rocblas_operation_transpose or \p rocblas_operation_conjugate_transpose.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    trans     rocblas_operation.\n
              Specifies the form of the system of equations.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of matrix A.
    @param[in]
    nrhs      rocblas_int. nrhs >= 0.\n
              The number of columns of matrices B and X;
              i.e., the columns on the right hand side.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A.
              On exit, the QR (or LQ) factorization of A as returned by GEQRF (or GELQF).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrix A.
    @param[inout]
    B         pointer to type. Array on the GPU of dimension ldb*nrhs.\n
              On entry, the matrix B is m-by-nrhs if non-transposed, or n-by-nrhs if transposed.
              On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
              the overdetermined cases) stored as columns.
    @param[in]
    ldb       rocblas_int. ldb >= max(m,n).\n
              Specifies the leading dimension of matrix B.
    @param[out]
    info      pointer to rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = j > 0, the solution could not be computed because input matrix A is
              rank deficient; the j-th diagonal element of its triangular factor is zero.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgels(rocblas_handle handle,
                                                rocblas_operation trans,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                float* A,
                                                const rocblas_int lda,
                                                float* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgels(rocblas_handle handle,
                                                rocblas_operation trans,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                double* A,
                                                const rocblas_int lda,
                                                double* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgels(rocblas_handle handle,
                                                rocblas_operation trans,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgels(rocblas_handle handle,
                                                rocblas_operation trans,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);
///@}

/*! @{
    \brief GELS_BATCHED solves batches of overdetermined (or underdetermined) linear systems
    defined by the array of m-by-n matrices A, and an array of corresponding matrices B, using the
    QR factorizations computed by GEQRF (or the LQ factorizations computed by GELQF).

    \details
    The problem solved by this function is either of the form

        A_i  * X_i = B_i (no transpose), or
        A_i' * X_i = B_i (transpose/conjugate transpose)

    depending on the value of trans.

    If m >= n (or n < m in the case of transpose/conjugate transpose), the systems are
    overdetermined and least-squares solutions approximating X_i are found minimizing

        || B_i - A_i  * X_i || (no transpose), or
        || B_i - A_i' * X_i || (transpose/conjugate transpose)

    If n < m (or m >= n in the case of transpose/conjugate transpose), the system is
    underdetermined and a unique solution for X_i is chosen minimizing || X_i ||

    \note
    The current implementation only supports the overdetermined, no transpose case.
    \p rocblas_status_not_implemented will be returned if m < n, or if trans is
    \p rocblas_operation_transpose or \p rocblas_operation_conjugate_transpose.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    trans     rocblas_operation.\n
              Specifies the form of the system of equations.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[in]
    nrhs      rocblas_int. nrhs >= 0.\n
              The number of columns of all matrices B_i and X_i in the batch;
              i.e., the columns on the right hand side.
    @param[inout]
    A         array of pointer to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i.
              On exit, the QR (or LQ) factorizations of A_i as returned by GEQRF (or GELQF).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[inout]
    B         array of pointer to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
              On entry, the matrices B_i are m-by-nrhs if non-transposed, or n-by-nrhs if transposed.
              On exit, when info = 0, each B_i is overwritten by the solution vectors (and the residuals in
              the overdetermined cases) stored as columns.
    @param[in]
    ldb       rocblas_int. ldb >= max(m,n).\n
              Specifies the leading dimension of matrices B_i.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for solution of A_i.
              If info_i = j > 0, the solution of A_i could not be computed because input
              matrix A_i is rank deficient; the j-th diagonal element of its triangular factor is zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
              Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgels_batched(rocblas_handle handle,
                                                        rocblas_operation trans,
                                                        const rocblas_int m,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgels_batched(rocblas_handle handle,
                                                        rocblas_operation trans,
                                                        const rocblas_int m,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgels_batched(rocblas_handle handle,
                                                        rocblas_operation trans,
                                                        const rocblas_int m,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_float_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgels_batched(rocblas_handle handle,
                                                        rocblas_operation trans,
                                                        const rocblas_int m,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_double_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
///@}

/*! @{
    \brief GELS_STRIDED_BATCHED solves batches of overdetermined (or underdetermined) linear
    systems defined by the array of m-by-n matrices A, and an array of corresponding matrices B,
    using the QR factorizations computed by GEQRF (or the LQ factorizations computed by GELQF).

    \details
    The problem solved by this function is either of the form

        A_i  * X_i = B_i (no transpose), or
        A_i' * X_i = B_i (transpose/conjugate transpose)

    depending on the value of trans.

    If m >= n (or n < m in the case of transpose/conjugate transpose), the systems are
    overdetermined and least-squares solutions approximating X_i are found minimizing

        || B_i - A_i  * X_i || (no transpose), or
        || B_i - A_i' * X_i || (transpose/conjugate transpose)

    If n < m (or m >= n in the case of transpose/conjugate transpose), the system is
    underdetermined and a unique solution for X_i is chosen minimizing || X_i ||

    \note
    The current implementation only supports the overdetermined, no transpose case.
    \p rocblas_status_not_implemented will be returned if m < n, or if trans is
    \p rocblas_operation_transpose or \p rocblas_operation_conjugate_transpose.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    trans     rocblas_operation.\n
              Specifies the form of the system of equations.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of columns of all matrices A_i in the batch.
    @param[in]
    nrhs      rocblas_int. nrhs >= 0.\n
              The number of columns of all matrices B_i and X_i in the batch;
              i.e., the columns on the right hand side.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_i.
              On exit, the QR (or LQ) factorizations of A_i as returned by GEQRF (or GELQF).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[inout]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              On entry, the matrices B_i are m-by-nrhs if non-transposed, or n-by-nrhs if transposed.
              On exit, when info = 0, each B_i is overwritten by the solution vectors (and the residuals in
              the overdetermined cases) stored as columns.
    @param[in]
    ldb       rocblas_int. ldb >= max(m,n).\n
              Specifies the leading dimension of matrices B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i and the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit for solution of A_i.
              If info_i = j > 0, the solution of A_i could not be computed because input
              matrix A_i is rank deficient; the j-th diagonal element of its triangular factor is zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
              Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgels_strided_batched(rocblas_handle handle,
                                                                rocblas_operation trans,
                                                                const rocblas_int m,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                float* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                float* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgels_strided_batched(rocblas_handle handle,
                                                                rocblas_operation trans,
                                                                const rocblas_int m,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                double* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                double* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgels_strided_batched(rocblas_handle handle,
                                                                rocblas_operation trans,
                                                                const rocblas_int m,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                rocblas_float_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_float_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgels_strided_batched(rocblas_handle handle,
                                                                rocblas_operation trans,
                                                                const rocblas_int m,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                rocblas_double_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_double_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
///@}

/*! @{
    \brief POTF2 computes the Cholesky factorization of a real symmetric/complex
    Hermitian positive definite matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form:

        A = U' * U, or
        A = L  * L'

    depending on the value of uplo. U is an upper triangular matrix and L is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A to be factored. On exit, the lower or upper triangular factor.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful factorization of matrix A.
              If info = i > 0, the leading minor of order i of A is not positive definite.
              The factorization stopped at this point.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief POTF2_BATCHED computes the Cholesky factorization of a
    batch of real symmetric/complex Hermitian positive definite matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_i in the batch has the form:

        A_i = U_i' * U_i, or
        A_i = L_i  * L_i'

    depending on the value of uplo. U_i is an upper triangular matrix and L_i is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The dimension of matrix A_i.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_i.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful factorization of matrix A_i.
              If info_i = j > 0, the leading minor of order j of A_i is not positive definite.
              The i-th factorization stopped at this point.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief POTF2_STRIDED_BATCHED computes the Cholesky factorization of a
    batch of real symmetric/complex Hermitian positive definite matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_i in the batch has the form:

        A_i = U_i' * U_i, or
        A_i = L_i  * L_i'

    depending on the value of uplo. U_i is an upper triangular matrix and L_i is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The dimension of matrix A_i.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful factorization of matrix A_i.
              If info_i = j > 0, the leading minor of order j of A_i is not positive definite.
              The i-th factorization stopped at this point.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief POTRF computes the Cholesky factorization of a real symmetric/complex
    Hermitian positive definite matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form:

        A = U' * U, or
        A = L  * L'

    depending on the value of uplo. U is an upper triangular matrix and L is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A to be factored. On exit, the lower or upper triangular factor.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful factorization of matrix A.
              If info = i > 0, the leading minor of order i of A is not positive definite.
              The factorization stopped at this point.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief POTRF_BATCHED computes the Cholesky factorization of a
    batch of real symmetric/complex Hermitian positive definite matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_i in the batch has the form:

        A_i = U_i' * U_i, or
        A_i = L_i  * L_i'

    depending on the value of uplo. U_i is an upper triangular matrix and L_i is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The dimension of matrix A_i.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_i.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful factorization of matrix A_i.
              If info_i = j > 0, the leading minor of order j of A_i is not positive definite.
              The i-th factorization stopped at this point.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief POTRF_STRIDED_BATCHED computes the Cholesky factorization of a
    batch of real symmetric/complex Hermitian positive definite matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_i in the batch has the form:

        A_i = U_i' * U_i, or
        A_i = L_i  * L_i'

    depending on the value of uplo. U_i is an upper triangular matrix and L_i is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the factorization is upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The dimension of matrix A_i.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful factorization of matrix A_i.
              If info_i = j > 0, the leading minor of order j of A_i is not positive definite.
              The i-th factorization stopped at this point.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GESVD computes the Singular Values and optionally the Singular
    Vectors of a general m-by-n matrix A (Singular Value Decomposition).

    \details
    The SVD of matrix A is given by:

        A = U * S * V'

    where the m-by-n matrix S is zero except, possibly, for its min(m,n)
    diagonal elements, which are the singular values of A. U and V are orthogonal
    (unitary) matrices. The first min(m,n) columns of U and V are the left and
    right singular vectors of A, respectively.

    The computation of the singular vectors is optional and it is controlled by
    the function arguments left_svect and right_svect as described below. When
    computed, this function returns the transpose (or transpose conjugate) of the
    right singular vectors, i.e. the rows of V'.

    left_svect and right_svect are #rocblas_svect enums that can take the
    following values:

    - rocblas_svect_all: the entire matrix U (or V') is computed,
    - rocblas_svect_singular: only the singular vectors (first min(m,n)
      columns of U or rows of V') are computed,
    - rocblas_svect_overwrite: the first
      columns (or rows) of A are overwritten with the singular vectors, or
    - rocblas_svect_none: no columns (or rows) of U (or V') are computed, i.e.
      no singular vectors.

    left_svect and right_svect cannot both be set to overwrite. When neither is
    set to overwrite, the contents of A are destroyed by the time the function
    returns.

   \note
    When m >> n (or n >> m) the algorithm could be sped up by compressing
    the matrix A via a QR (or LQ) factorization, and working with the triangular
    factor afterwards (thin-SVD). If the singular vectors are also requested, its
    computation could be sped up as well via executing some intermediate
    operations out-of-place, and relying more on matrix multiplications (GEMMs);
    this will require, however, a larger memory workspace. The parameter fast_alg
    controls whether the fast algorithm is executed or not. For more details see
    the sections "Tuning rocSOLVER performance" and "Memory model" on the User's
    guide.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    left_svect  #rocblas_svect.\n
                Specifies how the left singular vectors are computed.
    @param[in]
    right_svect #rocblas_svect.\n
                Specifies how the right singular vectors are computed.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry the matrix A.
                On exit, if left_svect (or right_svect) is equal to overwrite,
                the first columns (or rows) contain the left (or right) singular vectors;
                otherwise, contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                The leading dimension of A.
    @param[out]
    S           pointer to real type. Array on the GPU of dimension min(m,n). \n
                The singular values of A in decreasing order.
    @param[out]
    U           pointer to type. Array on the GPU of dimension ldu*min(m,n) if
                left_svect is set to singular, or ldu*m when left_svect is equal to all.\n
                The matrix of left singular vectors stored as columns. Not
                referenced if left_svect is set to overwrite or none.
    @param[in]
    ldu         rocblas_int. ldu >= m if left_svect is all or singular; ldu >= 1 otherwise.\n
                The leading dimension of U.
    @param[out]
    V           pointer to type. Array on the GPU of dimension ldv*n. \n
                The matrix of right singular vectors stored as rows (transposed / conjugate-transposed).
                Not referenced if right_svect is set to overwrite or none.
    @param[in]
    ldv         rocblas_int. ldv >= n if right_svect is all; ldv >= min(m,n) if right_svect is
                set to singular; or ldv >= 1 otherwise.\n The leading dimension of V.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
                This array is used to work internally with the bidiagonal matrix
                B associated to A (using BDSQR). On exit, if info > 0, it contains the
                unconverged off-diagonal elements of B (or properly speaking, a bidiagonal
                matrix orthogonally equivalent to B). The diagonal elements of this matrix
                are in S; those that converged correspond to a subset of the singular values
                of A (not necessarily ordered).
    @param[in]
    fast_alg    #rocblas_workmode. \n
                If set to rocblas_outofplace, the function will execute the
                fast thin-SVD version of the algorithm when possible.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                                 const rocblas_svect left_svect,
                                                 const rocblas_svect right_svect,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* S,
                                                 float* U,
                                                 const rocblas_int ldu,
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* E,
                                                 const rocblas_workmode fast_alg,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesvd(rocblas_handle handle,
                                                 const rocblas_svect left_svect,
                                                 const rocblas_svect right_svect,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* S,
                                                 double* U,
                                                 const rocblas_int ldu,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* E,
                                                 const rocblas_workmode fast_alg,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesvd(rocblas_handle handle,
                                                 const rocblas_svect left_svect,
                                                 const rocblas_svect right_svect,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* S,
                                                 rocblas_float_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 float* E,
                                                 const rocblas_workmode fast_alg,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesvd(rocblas_handle handle,
                                                 const rocblas_svect left_svect,
                                                 const rocblas_svect right_svect,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* S,
                                                 rocblas_double_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 double* E,
                                                 const rocblas_workmode fast_alg,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief GESVD_BATCHED computes the Singular Values and optionally the
    Singular Vectors of a batch of general m-by-n matrix A (Singular Value
    Decomposition).

    \details
    The SVD of matrix A_j is given by:

        A_j = U_j * S_j * V_j'

    where the m-by-n matrix S_j is zero except, possibly, for its min(m,n)
    diagonal elements, which are the singular values of A_j. U_j and V_j are
    orthogonal (unitary) matrices. The first min(m,n) columns of U_j and V_j are
    the left and right singular vectors of A_j, respectively.

    The computation of the singular vectors is optional and it is controlled by
    the function arguments left_svect and right_svect as described below. When
    computed, this function returns the transpose (or transpose conjugate) of the
    right singular vectors, i.e. the rows of V_j'.

    left_svect and right_svect are #rocblas_svect enums that can take the
    following values:

    - rocblas_svect_all: the entire matrix U_j (or V_j') is computed,
    - rocblas_svect_singular: only the singular vectors (first min(m,n)
      columns of U_j or rows of V_j') are computed,
    - rocblas_svect_overwrite: the
      first columns (or rows) of A_j are overwritten with the singular vectors, or
    - rocblas_svect_none: no columns (or rows) of U_j (or V_j') are computed,
      i.e. no singular vectors.

    left_svect and right_svect cannot both be set to overwrite. When neither is
    set to overwrite, the contents of A_j are destroyed by the time the function
    returns.

    \note
    When m >> n (or n >> m) the algorithm could be sped up by compressing
    the matrix A_j via a QR (or LQ) factorization, and working with the
    triangular factor afterwards (thin-SVD). If the singular vectors are also
    requested, its computation could be sped up as well via executing some
    intermediate operations out-of-place, and relying more on matrix
    multiplications (GEMMs); this will require, however, a larger memory
    workspace. The parameter fast_alg controls whether the fast algorithm is
    executed or not. For more details see the sections
    "Tuning rocSOLVER performance" and "Memory model" on the User's guide.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    left_svect  #rocblas_svect.\n
                Specifies how the left singular vectors are computed.
    @param[in]
    right_svect #rocblas_svect.\n
                Specifies how the right singular vectors are computed.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on
                the GPU of dimension lda*n.\n
                On entry the matrices A_j.
                On exit, if left_svect (or right_svect) is equal to overwrite,
                the first columns (or rows) of A_j contain the left (or right)
                corresponding singular vectors; otherwise, contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                The leading dimension of A_j.
    @param[out]
    S           pointer to real type. Array on the GPU (the size depends on the value of strideS).\n
                The singular values of A_j in decreasing order.
    @param[in]
    strideS     rocblas_stride.\n
                Stride from the start of one vector S_j to the next one S_(j+1).
                There is no restriction for the value of strideS.
                Normal use case is strideS >= min(m,n).
    @param[out]
    U           pointer to type. Array on the GPU (the side depends on the value of strideU). \n
                The matrices U_j of left singular vectors stored as columns.
                Not referenced if left_svect is set to overwrite or none.
    @param[in]
    ldu         rocblas_int. ldu >= m if left_svect is all or singular; ldu >= 1 otherwise.\n
                The leading dimension of U_j.
    @param[in]
    strideU     rocblas_stride.\n
                Stride from the start of one matrix U_j to the next one U_(j+1).
                There is no restriction for the value of strideU.
                Normal use case is strideU >= ldu*min(m,n) if left_svect is set to singular,
                or strideU >= ldu*m when left_svect is equal to all.
    @param[out]
    V           pointer to type. Array on the GPU (the size depends on the value of strideV). \n
                The matrices V_j of right singular vectors stored as rows (transposed / conjugate-transposed).
                Not referenced if right_svect is set to overwrite or none.
    @param[in]
    ldv         rocblas_int. ldv >= n if right_svect is all; ldv >= min(m,n) if
                right_svect is set to singular; or ldv >= 1 otherwise.\n
                The leading dimension of V.
    @param[in]
    strideV     rocblas_stride.\n
                Stride from the start of one matrix V_j to the next one V_(j+1).
                There is no restriction for the value of strideV.
                Normal use case is strideV >= ldv*n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the bidiagonal matrix B_j associated to A_j (using BDSQR).
                On exit, if info > 0, it contains the unconverged off-diagonal elements of B_j (or properly speaking,
                a bidiagonal matrix orthogonally equivalent to B_j). The diagonal elements of this matrix are in S_j;
                those that converged correspond to a subset of the singular values of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[in]
    fast_alg    #rocblas_workmode. \n
                If set to rocblas_outofplace, the function will execute the fast thin-SVD version
                of the algorithm when possible.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesvd_batched(rocblas_handle handle,
                                                         const rocblas_svect left_svect,
                                                         const rocblas_svect right_svect,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* S,
                                                         const rocblas_stride strideS,
                                                         float* U,
                                                         const rocblas_int ldu,
                                                         const rocblas_stride strideU,
                                                         float* V,
                                                         const rocblas_int ldv,
                                                         const rocblas_stride strideV,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         const rocblas_workmode fast_alg,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesvd_batched(rocblas_handle handle,
                                                         const rocblas_svect left_svect,
                                                         const rocblas_svect right_svect,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* S,
                                                         const rocblas_stride strideS,
                                                         double* U,
                                                         const rocblas_int ldu,
                                                         const rocblas_stride strideU,
                                                         double* V,
                                                         const rocblas_int ldv,
                                                         const rocblas_stride strideV,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         const rocblas_workmode fast_alg,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesvd_batched(rocblas_handle handle,
                                                         const rocblas_svect left_svect,
                                                         const rocblas_svect right_svect,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         float* S,
                                                         const rocblas_stride strideS,
                                                         rocblas_float_complex* U,
                                                         const rocblas_int ldu,
                                                         const rocblas_stride strideU,
                                                         rocblas_float_complex* V,
                                                         const rocblas_int ldv,
                                                         const rocblas_stride strideV,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         const rocblas_workmode fast_alg,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesvd_batched(rocblas_handle handle,
                                                         const rocblas_svect left_svect,
                                                         const rocblas_svect right_svect,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         double* S,
                                                         const rocblas_stride strideS,
                                                         rocblas_double_complex* U,
                                                         const rocblas_int ldu,
                                                         const rocblas_stride strideU,
                                                         rocblas_double_complex* V,
                                                         const rocblas_int ldv,
                                                         const rocblas_stride strideV,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         const rocblas_workmode fast_alg,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GESVD_STRIDED_BATCHED computes the Singular Values and optionally the
    Singular Vectors of a batch of general m-by-n matrix A (Singular Value
    Decomposition).

    \details
    The SVD of matrix A_j is given by:

        A_j = U_j * S_j * V_j'

    where the m-by-n matrix S_j is zero except, possibly, for its min(m,n)
    diagonal elements, which are the singular values of A_j. U_j and V_j are
    orthogonal (unitary) matrices. The first min(m,n) columns of U_j and V_j are
    the left and right singular vectors of A_j, respectively.

    The computation of the singular vectors is optional and it is controlled by
    the function arguments left_svect and right_svect as described below. When
    computed, this function returns the transpose (or transpose conjugate) of the
    right singular vectors, i.e. the rows of V_j'.

    left_svect and right_svect are #rocblas_svect enums that can take the
    following values:

    - rocblas_svect_all: the entire matrix U_j (or V_j') is computed,
    - rocblas_svect_singular: only the singular vectors (first min(m,n) columns
      of U_j or rows of V_j') are computed,
    - rocblas_svect_overwrite: the first columns (or rows) of
      A_j are overwritten with the singular vectors, or
    - rocblas_svect_none: no columns (or rows) of U_j (or V_j')
      are computed, i.e. no singular vectors.

    left_svect and right_svect cannot both be set to overwrite. When neither is
    set to overwrite, the contents of A_j are destroyed by the time the function
    returns.

    \note
    When m >> n (or n >> m) the algorithm could be sped up by compressing
    the matrix A_j via a QR (or LQ) factorization, and working with the
    triangular factor afterwards (thin-SVD). If the singular vectors are also
    requested, its computation could be sped up as well via executing some
    intermediate operations out-of-place, and relying more on matrix
    multiplications (GEMMs); this will require, however, a larger memory
    workspace. The parameter fast_alg controls whether the fast algorithm is
    executed or not. For more details see the sections
    "Tuning rocSOLVER performance" and "Memory model" on the User's guide.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    left_svect  #rocblas_svect.\n
                Specifies how the left singular vectors are computed.
    @param[in]
    right_svect #rocblas_svect.\n
                Specifies how the right singular vectors are computed.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry the matrices A_j. On exit, if left_svect (or right_svect) is equal to
                overwrite, the first columns (or rows) of A_j contain the left (or right)
                corresponding singular vectors; otherwise, contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                The leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA.
                Normal use case is strideA >= lda*n.
    @param[out]
    S           pointer to real type. Array on the GPU (the size depends on the value of strideS).\n
                The singular values of A_j in decreasing order.
    @param[in]
    strideS     rocblas_stride.\n
                Stride from the start of one vector S_j to the next one S_(j+1).
                There is no restriction for the value of strideS.
                Normal use case is strideS >= min(m,n).
    @param[out]
    U           pointer to type. Array on the GPU (the side depends on the value of strideU). \n
                The matrices U_j of left singular vectors stored as columns.
                Not referenced if left_svect is set to overwrite or none.
    @param[in]
    ldu         rocblas_int. ldu >= m if left_svect is all or singular; ldu >= 1 otherwise.\n
                The leading dimension of U_j.
    @param[in]
    strideU     rocblas_stride.\n
                Stride from the start of one matrix U_j to the next one U_(j+1).
                There is no restriction for the value of strideU.
                Normal use case is strideU >= ldu*min(m,n) if left_svect is set to singular,
                or strideU >= ldu*m when left_svect is equal to all.
    @param[out]
    V           pointer to type. Array on the GPU (the size depends on the value of strideV). \n
                The matrices V_j of right singular vectors stored as rows (transposed / conjugate-transposed).
                Not referenced if right_svect is set to overwrite or none.
    @param[in]
    ldv         rocblas_int. ldv >= n if right_svect is all; ldv >= min(m,n) if right_svect is
                set to singular; or ldv >= 1 otherwise.\n
                The leading dimension of V.
    @param[in]
    strideV     rocblas_stride.\n
                Stride from the start of one matrix V_j to the next one V_(j+1).
                There is no restriction for the value of strideV.
                Normal use case is strideV >= ldv*n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the bidiagonal matrix B_j associated to A_j (using BDSQR).
                On exit, if info > 0, it contains the unconverged off-diagonal elements of B_j (or properly speaking,
                a bidiagonal matrix orthogonally equivalent to B_j). The diagonal elements of this matrix are in S_j;
                those that converged correspond to a subset of the singular values of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE.
                Normal use case is strideE >= min(m,n)-1.
    @param[in]
    fast_alg    #rocblas_workmode. \n
                If set to rocblas_outofplace, the function will execute the fast thin-SVD version
                of the algorithm when possible.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_svect left_svect,
                                                                 const rocblas_svect right_svect,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* S,
                                                                 const rocblas_stride strideS,
                                                                 float* U,
                                                                 const rocblas_int ldu,
                                                                 const rocblas_stride strideU,
                                                                 float* V,
                                                                 const rocblas_int ldv,
                                                                 const rocblas_stride strideV,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 const rocblas_workmode fast_alg,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_svect left_svect,
                                                                 const rocblas_svect right_svect,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* S,
                                                                 const rocblas_stride strideS,
                                                                 double* U,
                                                                 const rocblas_int ldu,
                                                                 const rocblas_stride strideU,
                                                                 double* V,
                                                                 const rocblas_int ldv,
                                                                 const rocblas_stride strideV,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 const rocblas_workmode fast_alg,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_svect left_svect,
                                                                 const rocblas_svect right_svect,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* S,
                                                                 const rocblas_stride strideS,
                                                                 rocblas_float_complex* U,
                                                                 const rocblas_int ldu,
                                                                 const rocblas_stride strideU,
                                                                 rocblas_float_complex* V,
                                                                 const rocblas_int ldv,
                                                                 const rocblas_stride strideV,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 const rocblas_workmode fast_alg,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_svect left_svect,
                                                                 const rocblas_svect right_svect,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* S,
                                                                 const rocblas_stride strideS,
                                                                 rocblas_double_complex* U,
                                                                 const rocblas_int ldu,
                                                                 const rocblas_stride strideU,
                                                                 rocblas_double_complex* V,
                                                                 const rocblas_int ldv,
                                                                 const rocblas_stride strideV,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 const rocblas_workmode fast_alg,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTD2 computes the tridiagonal form of a real symmetric matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form is given by:

        T = Q' * A * Q

    where T is symmetric tridiagonal and Q is an orthogonal matrix represented as the product
    of Householder matrices

        Q = H(1) * H(2) * ... *  H(n-1) if uplo indicates lower, or
        Q = H(n-1) * H(n-2) * ... * H(1) if uplo indicates upper.

    Each Householder matrix H(i) is given by

        H(i) = I - tau[i] * v(i) * v(i)'

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal
              contain the tridiagonal form T; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to type. Array on the GPU of dimension n.\n
              The diagonal elements of T.
    @param[out]
    E         pointer to type. Array on the GPU of dimension n-1.\n
              The off-diagonal elements of T.
    @param[out]
    tau       pointer to type. Array on the GPU of dimension n-1.\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytd2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 float* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytd2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 double* tau);
//! @}

/*! @{
    \brief HETD2 computes the tridiagonal form of a complex hermitian matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form is given by:

        T = Q' * A * Q

    where T is hermitian tridiagonal and Q is an unitary matrix represented as the product
    of Householder matrices

        Q = H(1) * H(2) * ... *  H(n-1) if uplo indicates lower, or
        Q = H(n-1) * H(n-2) * ... * H(1) if uplo indicates upper.

    Each Householder matrix H(i) is given by

        H(i) = I - tau[i] * v(i) * v(i)'

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal
              contain the tridiagonal form T; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension n.\n
              The diagonal elements of T.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension n-1.\n
              The off-diagonal elements of T.
    @param[out]
    tau       pointer to type. Array on the GPU of dimension n-1.\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetd2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetd2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tau);

//! @}

/*! @{
    \brief SYTD2_BATCHED computes the tridiagonal form of a batch of real symmetric matrices A_j.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is symmetric tridiagonal and Q_j is an orthogonal matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytd2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         float* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytd2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         double* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief HETD2_BATCHED computes the tridiagonal form of a batch of complex hermitian matrices A_j.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is hermitian tridiagonal and Q_j is a unitary  matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetd2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_float_complex* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetd2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_double_complex* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTD2_STRIDED_BATCHED computes the tridiagonal form of a batch of real symmetric matrices A_j.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is symmetric tridiagonal and Q_j is an orthogonal matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 float* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 double* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief HETD2_STRIDED_BATCHED computes the tridiagonal form of a batch of complex hermitian matrices A_j.

    \details
    (This is the unblocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is hermitian tridiagonal and Q_j is a unitary  matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_float_complex* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetd2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_double_complex* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTRD computes the tridiagonal form of a real symmetric matrix A.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form is given by:

        T = Q' * A * Q

    where T is symmetric tridiagonal and Q is an orthogonal matrix represented as the product
    of Householder matrices

        Q = H(1) * H(2) * ... *  H(n-1) if uplo indicates lower, or
        Q = H(n-1) * H(n-2) * ... * H(1) if uplo indicates upper.

    Each Householder matrix H(i) is given by

        H(i) = I - tau[i] * v(i) * v(i)'

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal
              contain the tridiagonal form T; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to type. Array on the GPU of dimension n.\n
              The diagonal elements of T.
    @param[out]
    E         pointer to type. Array on the GPU of dimension n-1.\n
              The off-diagonal elements of T.
    @param[out]
    tau       pointer to type. Array on the GPU of dimension n-1.\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 float* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 double* tau);
//! @}

/*! @{
    \brief HETRD computes the tridiagonal form of a complex hermitian matrix A.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form is given by:

        T = Q' * A * Q

    where T is hermitian tridiagonal and Q is an unitary matrix represented as the product
    of Householder matrices

        Q = H(1) * H(2) * ... *  H(n-1) if uplo indicates lower, or
        Q = H(n-1) * H(n-2) * ... * H(1) if uplo indicates upper.

    Each Householder matrix H(i) is given by

        H(i) = I - tau[i] * v(i) * v(i)'

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v(i) are zero, and v(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal
              contain the tridiagonal form T; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension n.\n
              The diagonal elements of T.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension n-1.\n
              The off-diagonal elements of T.
    @param[out]
    tau       pointer to type. Array on the GPU of dimension n-1.\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetrd(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tau);

//! @}

/*! @{
    \brief SYTRD_BATCHED computes the tridiagonal form of a batch of real symmetric matrices A_j.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is symmetric tridiagonal and Q_j is an orthogonal matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrd_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         float* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrd_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         double* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief HETRD_BATCHED computes the tridiagonal form of a batch of complex hermitian matrices A_j.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is hermitian tridiagonal and Q_j is a unitary  matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetrd_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         float* D,
                                                         const rocblas_stride strideD,
                                                         float* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_float_complex* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetrd_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         double* D,
                                                         const rocblas_stride strideD,
                                                         double* E,
                                                         const rocblas_stride strideE,
                                                         rocblas_double_complex* tau,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTRD_STRIDED_BATCHED computes the tridiagonal form of a batch of real symmetric matrices A_j.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is symmetric tridiagonal and Q_j is an orthogonal matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 float* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 double* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief HETRD_STRIDED_BATCHED computes the tridiagonal form of a batch of complex hermitian matrices A_j.

    \details
    (This is the blocked version of the algorithm).

    The tridiagonal form of A_j is given by:

        T_j = Q_j' * A_j * Q_j, for j = 1,2,...,batch_count

    where T_j is hermitian tridiagonal and Q_j is a unitary  matrix represented as the product
    of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... *  H_j(n-1) if uplo indicates lower, or
        Q_j = H_j(n-1) * H_j(n-2) * ... * H_j(1) if uplo indicates upper.

    Each Householder matrix H_j(i) is given by

        H_j(i) = I - tau_j[i] * v_j(i) * v_j(i)'

    where tau_j[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector v_j(i) are zero, and v_j(i)[i+1] = 1. If uplo indicates upper,
    the last n-i elements of the Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
              If uplo indicates lower (or upper), then the upper (or lower)
              part of A is not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of rows and columns of the matrices A_j.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_j to be factored.
              On exit, if upper, then the elements on the diagonal and superdiagonal of A_j
              contain the tridiagonal form T_j; the elements above the superdiagonal contain
              the i-1 non-zero elements of vectors v_j(i) stored as columns.
              If lower, then the elements on the diagonal and subdiagonal
              contain the tridiagonal form T_j; the elements below the subdiagonal contain
              the n-i-1 non-zero elements of vectors v_j(i) stored as columns.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A_j.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              The diagonal elements of T_j.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_j and the next one D_(j+1).
              There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              The off-diagonal elements of T_j.
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_j and the next one E_(j+1).
              There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau       pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors tau_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_stride.\n
              Stride from the start of one vector tau_j to the next one tau_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chetrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* D,
                                                                 const rocblas_stride strideD,
                                                                 float* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_float_complex* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhetrd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* D,
                                                                 const rocblas_stride strideD,
                                                                 double* E,
                                                                 const rocblas_stride strideE,
                                                                 rocblas_double_complex* tau,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGS2 reduces a real symmetric-definite generalized eigenproblem to standard
    form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten as

        inv(U') * A * inv(U), or
        inv(L)  * A * inv(L'),

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten as

        U  * A * U', or
        L' * A * L,

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrix A is stored, and
              whether the factorization applied to B was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A and
              B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A. On exit, the transformed matrix associated with
              the equivalent standard eigenvalue problem.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              The triangular factor of the matrix B, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygs2(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygs2(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief HEGS2 reduces a hermitian-definite generalized eigenproblem to standard form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten as

        inv(U') * A * inv(U), or
        inv(L)  * A * inv(L'),

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten as

        U  * A * U', or
        L' * A * L,

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrix A is stored, and
              whether the factorization applied to B was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A and
              B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A. On exit, the transformed matrix associated with
              the equivalent standard eigenvalue problem.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              The triangular factor of the matrix B, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegs2(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegs2(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief SYGS2_BATCHED reduces a batch of real symmetric-definite generalized eigenproblems
    to standard form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygs2_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygs2_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGS2_BATCHED reduces a batch of hermitian-definite generalized eigenproblems to
    standard form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegs2_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegs2_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGS2_STRIDED_BATCHED reduces a batch of real symmetric-definite generalized
    eigenproblems to standard form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i and the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygs2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygs2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGS2_STRIDED_BATCHED reduces a batch of hermitian-definite generalized
    eigenproblems to standard form.

    \details
    (This is the unblocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i and the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegs2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegs2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGST reduces a real symmetric-definite generalized eigenproblem to standard
    form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten as

        inv(U') * A * inv(U), or
        inv(L)  * A * inv(L'),

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten as

        U  * A * U', or
        L' * A * L,

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrix A is stored, and
              whether the factorization applied to B was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A and
              B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A. On exit, the transformed matrix associated with
              the equivalent standard eigenvalue problem.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              The triangular factor of the matrix B, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygst(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygst(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief HEGST reduces a hermitian-definite generalized eigenproblem to standard form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten as

        inv(U') * A * inv(U), or
        inv(L)  * A * inv(L'),

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten as

        U  * A * U', or
        L' * A * L,

    where B has been factorized as either U' * U or L * L' as returned by POTRF, depending
    on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrix A is stored, and
              whether the factorization applied to B was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A and
              B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the matrix A. On exit, the transformed matrix associated with
              the equivalent standard eigenvalue problem.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              The triangular factor of the matrix B, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegst(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegst(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief SYGST_BATCHED reduces a batch of real symmetric-definite generalized eigenproblems
    to standard form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygst_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygst_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGST_BATCHED reduces a batch of hermitian-definite generalized eigenproblems to
    standard form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegst_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegst_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGST_STRIDED_BATCHED reduces a batch of real symmetric-definite generalized
    eigenproblems to standard form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i and the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygst_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygst_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGST_STRIDED_BATCHED reduces a batch of hermitian-definite generalized
    eigenproblems to standard form.

    \details
    (This is the blocked version of the algorithm).

    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    If the problem is of the 1st form, then A_i is overwritten as

        inv(U_i') * A_i * inv(U_i), or
        inv(L_i)  * A_i * inv(L_i'),

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A_i is overwritten as

        U_i  * A_i * U_i', or
        L_i' * A_i * L_i,

    where B_i has been factorized as either U_i' * U_i or L_i * L_i' as returned by POTRF,
    depending on the value of uplo.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower part of the matrices A_i are stored, and
              whether the factorization applied to B_i was upper or lower triangular.
              If uplo indicates lower (or upper), then the upper (or lower) parts of A_i and
              B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the matrices A_i. On exit, the transformed matrices associated with
              the equivalent standard eigenvalue problems.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              The triangular factors of the matrices B_i, as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i and the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegst_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegst_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYEV computes the eigenvalues and optionally the eigenvectors of a real symmetric
    matrix A.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated to A.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues of A (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit. If info = i > 0, the algorithm did not converge.
                i elements of E did not converge to zero.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyev(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                float* D,
                                                float* E,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyev(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                double* D,
                                                double* E,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief HEEV computes the eigenvalues and optionally the eigenvectors of a Hermitian matrix A.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the Hermitian matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated to A.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues of A (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit. If info = i > 0, the algorithm did not converge.
                i elements of E did not converge to zero.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheev(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                float* D,
                                                float* E,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zheev(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                double* D,
                                                double* E,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief SYEV_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    real symmetric matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrices A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A_j
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to type. Array on the GPU (the side depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated to A_j.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info_j = 0, successful exit for matrix A_j. If info_j = i > 0, the algorithm did not converge.
                i elements of E_j did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyev_batched(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* D,
                                                        const rocblas_stride strideD,
                                                        float* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyev_batched(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* D,
                                                        const rocblas_stride strideD,
                                                        double* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEEV_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    Hermitian matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the Hermitian matrices A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A_j
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the side depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated to A_j.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info_j = 0, successful exit for matrix A_j. If info_j = i > 0, the algorithm did not converge.
                i elements of E_j did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheev_batched(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        float* D,
                                                        const rocblas_stride strideD,
                                                        float* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zheev_batched(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        double* D,
                                                        const rocblas_stride strideD,
                                                        double* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYEV_STRIDED_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    real symmetric matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrices A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A_j
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the side depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to type. Array on the GPU (the side depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated to A_j.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info_j = 0, successful exit for matrix A_j. If info_j = i > 0, the algorithm did not converge.
                i elements of E_j did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyev_strided_batched(rocblas_handle handle,
                                                                const rocblas_evect evect,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                float* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                float* D,
                                                                const rocblas_stride strideD,
                                                                float* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyev_strided_batched(rocblas_handle handle,
                                                                const rocblas_evect evect,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                double* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                double* D,
                                                                const rocblas_stride strideD,
                                                                double* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEEV_STRIDED_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    Hermitian matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed depending
    on the value of evect. The computed eigenvectors are orthonormal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the Hermitian matrices A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower) part of A_j
                is not used.
    @param[in]
    n           rocblas_int. n >= 0\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the side depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the side depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated to A_j.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info_j = 0, successful exit for matrix A_j. If info_j = i > 0, the algorithm did not converge.
                i elements of E_j did not converge to zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheev_strided_batched(rocblas_handle handle,
                                                                const rocblas_evect evect,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                rocblas_float_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                float* D,
                                                                const rocblas_stride strideD,
                                                                float* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zheev_strided_batched(rocblas_handle handle,
                                                                const rocblas_evect evect,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                rocblas_double_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                double* D,
                                                                const rocblas_stride strideD,
                                                                double* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGV computes the eigenvalues and (optionally) eigenvectors of
    a real generalized symmetric-definite eigenproblem.

    \details
    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A and B are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A and B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the symmetric matrix A. On exit, if jobz is original,
              the matrix Z of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z' * B * Z = I;
              2. If itype is bax, as Z' * inv(B) * Z = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrix A (including the diagonal) is destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              On entry, the symmetric positive definite matrix B. On exit, the
              triangular factor of B as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.
    @param[out]
    D         pointer to type. Array on the GPU of dimension n.\n
              On exit, the eigenvalues in increasing order.
    @param[out]
    E         pointer to type. Array on the GPU of dimension n.\n
              This array is used to work internally with the tridiagonal matrix T associated with
              the reduced eigenvalue problem.
              On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
              (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
              of this matrix are in D; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i <= n, i off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info = n + i, the leading minor of order i of B is not
              positive definite.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect jobz,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                float* B,
                                                const rocblas_int ldb,
                                                float* D,
                                                float* E,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect jobz,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                double* B,
                                                const rocblas_int ldb,
                                                double* D,
                                                double* E,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief HEGV computes the eigenvalues and (optionally) eigenvectors of
    a complex generalized hermitian-definite eigenproblem.

    \details
    The problem solved by this function is either of the form

        A * X = lambda * B * X (1st form), or
        A * B * X = lambda * X (2nd form), or
        B * A * X = lambda * X (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblem.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A and B are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A and B are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the hermitian matrix A. On exit, if jobz is original,
              the matrix Z of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z' * B * Z = I;
              2. If itype is bax, as Z' * inv(B) * Z = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrix A (including the diagonal) is destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A.
    @param[out]
    B         pointer to type. Array on the GPU of dimension ldb*n.\n
              On entry, the hermitian positive definite matrix B. On exit, the
              triangular factor of B as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B.
    @param[out]
    D         pointer to real type. Array on the GPU of dimension n.\n
              On exit, the eigenvalues in increasing order.
    @param[out]
    E         pointer to real type. Array on the GPU of dimension n.\n
              This array is used to work internally with the tridiagonal matrix T associated with
              the reduced eigenvalue problem.
              On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
              (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
              of this matrix are in D; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, successful exit.
              If info = i <= n, i off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info = n + i, the leading minor of order i of B is not
              positive definite.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect jobz,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                float* D,
                                                float* E,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect jobz,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                double* D,
                                                double* E,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief SYGV_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of real generalized symmetric-definite eigenproblems.

    \details
    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A_i and B_i are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A_i and B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the symmetric matrices A_i. On exit, if jobz is original,
              the matrix Z_i of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z_i' * B_i * Z_i = I;
              2. If itype is bax, as Z_i' * inv(B_i) * Z_i = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrices A_i (including the diagonal) are destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              On entry, the symmetric positive definite matrices B_i. On exit, the
              triangular factor of B_i as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              On exit, the eigenvalues in increasing order.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_i to the next one D_(i+1).
              There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              This array is used to work internally with the tridiagonal matrix T_i associated with
              the ith reduced eigenvalue problem.
              On exit, if 0 < info_i <= n, it contains the unconverged off-diagonal elements of T_i
              (or properly speaking, a tridiagonal matrix equivalent to T_i). The diagonal elements
              of this matrix are in D_i; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_i to the next one E_(i+1).
              There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit of batch i.
              If info_i = j <= n, j off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info_i = n + j, the leading minor of order j of B_i is not
              positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect jobz,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* const B[],
                                                        const rocblas_int ldb,
                                                        float* D,
                                                        const rocblas_stride strideD,
                                                        float* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect jobz,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* const B[],
                                                        const rocblas_int ldb,
                                                        double* D,
                                                        const rocblas_stride strideD,
                                                        double* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGV_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of complex generalized hermitian-definite eigenproblems.

    \details
    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A_i and B_i are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A_i and B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the hermitian matrices A_i. On exit, if jobz is original,
              the matrix Z_i of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z_i' * B_i * Z_i = I;
              2. If itype is bax, as Z_i' * inv(B_i) * Z_i = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrices A_i (including the diagonal) are destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[out]
    B         array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
              On entry, the hermitian positive definite matrices B_i. On exit, the
              triangular factor of B_i as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              On exit, the eigenvalues in increasing order.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_i to the next one D_(i+1).
              There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              This array is used to work internally with the tridiagonal matrix T_i associated with
              the ith reduced eigenvalue problem.
              On exit, if 0 < info_i <= n, it contains the unconverged off-diagonal elements of T_i
              (or properly speaking, a tridiagonal matrix equivalent to T_i). The diagonal elements
              of this matrix are in D_i; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_i to the next one E_(i+1).
              There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit of batch i.
              If info_i = j <= n, j off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info_i = n + j, the leading minor of order j of B_i is not
              positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect jobz,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_float_complex* const B[],
                                                        const rocblas_int ldb,
                                                        float* D,
                                                        const rocblas_stride strideD,
                                                        float* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect jobz,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_double_complex* const B[],
                                                        const rocblas_int ldb,
                                                        double* D,
                                                        const rocblas_stride strideD,
                                                        double* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYGV_STRIDED_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of real generalized symmetric-definite eigenproblems.

    \details
    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A_i and B_i are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A_i and B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the symmetric matrices A_i. On exit, if jobz is original,
              the matrix Z_i of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z_i' * B_i * Z_i = I;
              2. If itype is bax, as Z_i' * inv(B_i) * Z_i = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrices A_i (including the diagonal) are destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i to the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              On entry, the symmetric positive definite matrices B_i. On exit, the
              triangular factor of B_i as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i to the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D         pointer to type. Array on the GPU (the size depends on the value of strideD).\n
              On exit, the eigenvalues in increasing order.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_i to the next one D_(i+1).
              There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E         pointer to type. Array on the GPU (the size depends on the value of strideE).\n
              This array is used to work internally with the tridiagonal matrix T_i associated with
              the ith reduced eigenvalue problem.
              On exit, if 0 < info_i <= n, it contains the unconverged off-diagonal elements of T_i
              (or properly speaking, a tridiagonal matrix equivalent to T_i). The diagonal elements
              of this matrix are in D_i; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_i to the next one E_(i+1).
              There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit of batch i.
              If info_i = j <= n, j off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info_i = n + j, the leading minor of order j of B_i is not
              positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect jobz,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                float* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                float* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                float* D,
                                                                const rocblas_stride strideD,
                                                                float* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect jobz,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                double* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                double* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                double* D,
                                                                const rocblas_stride strideD,
                                                                double* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
//! @}

/*! @{
    \brief HEGV_STRIDED_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of complex generalized hermitian-definite eigenproblems.

    \details
    The problem solved by this function is either of the form

        A_i * X_i = lambda_i * B_i * X_i (1st form), or
        A_i * B_i * X_i = lambda_i * X_i (2nd form), or
        B_i * A_i * X_i = lambda_i * X_i (3rd form),

    depending on the value of itype.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    itype     #rocblas_eform.\n
              Specifies the form of the generalized eigenproblems.
    @param[in]
    jobz      #rocblas_evect.\n
              Specifies whether the eigenvectors are to be computed.
              If evect is rocblas_evect_original, then the eigenvectors are computed.
              rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo      rocblas_fill.\n
              Specifies whether the upper or lower parts of the matrices
              A_i and B_i are stored. If uplo indicates lower (or upper),
              then the upper (or lower) parts of A_i and B_i are not used.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the hermitian matrices A_i. On exit, if jobz is original,
              the matrix Z_i of eigenvectors, normalized as follows:
              1. If itype is ax or abx, as Z_i' * B_i * Z_i = I;
              2. If itype is bax, as Z_i' * inv(B_i) * Z_i = I.
              Otherwise, if jobz is none, then the upper or lower triangular
              part of the matrices A_i (including the diagonal) are destroyed,
              depending on the value of uplo.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              Specifies the leading dimension of A_i.
    @param[in]
    strideA   rocblas_stride.\n
              Stride from the start of one matrix A_i to the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B         pointer to type. Array on the GPU (the size depends on the value of strideB).\n
              On entry, the hermitian positive definite matrices B_i. On exit, the
              triangular factor of B_i as returned by POTRF.
    @param[in]
    ldb       rocblas_int. ldb >= n.\n
              Specifies the leading dimension of B_i.
    @param[in]
    strideB   rocblas_stride.\n
              Stride from the start of one matrix B_i to the next one B_(i+1).
              There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D         pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
              On exit, the eigenvalues in increasing order.
    @param[in]
    strideD   rocblas_stride.\n
              Stride from the start of one vector D_i to the next one D_(i+1).
              There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E         pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
              This array is used to work internally with the tridiagonal matrix T_i associated with
              the ith reduced eigenvalue problem.
              On exit, if 0 < info_i <= n, it contains the unconverged off-diagonal elements of T_i
              (or properly speaking, a tridiagonal matrix equivalent to T_i). The diagonal elements
              of this matrix are in D_i; those that converged correspond to a subset of the
              eigenvalues (not necessarily ordered).
    @param[in]
    strideE   rocblas_stride.\n
              Stride from the start of one vector E_i to the next one E_(i+1).
              There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, successful exit of batch i.
              If info_i = j <= n, j off-diagonal elements of an intermediate
              tridiagonal form did not converge to zero.
              If info_i = n + j, the leading minor of order j of B_i is not
              positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect jobz,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                rocblas_float_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_float_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                float* D,
                                                                const rocblas_stride strideD,
                                                                float* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect jobz,
                                                                const rocblas_fill uplo,
                                                                const rocblas_int n,
                                                                rocblas_double_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_double_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                double* D,
                                                                const rocblas_stride strideD,
                                                                double* E,
                                                                const rocblas_stride strideE,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
//! @}

#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */
