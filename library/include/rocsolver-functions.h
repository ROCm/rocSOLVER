/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
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

/*! \brief GET_VERSION_STRING_SIZE Queries the minimum buffer size for a
    successful call to \ref rocsolver_get_version_string.

    \details
    @param[out]
    len         pointer to size_t.\n
                The minimum length of buffer to pass to
                \ref rocsolver_get_version_string.
 ******************************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_get_version_string_size(size_t* len);

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
    layer_mode  rocblas_layer_mode_flags.\n
                Specifies the logging mode.
 ******************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_log_set_layer_mode(const rocblas_layer_mode_flags layer_mode);

/*! \brief LOG_SET_MAX_LEVELS sets the maximum trace log depth for the rocSOLVER
    multi-level logging environment.

    \details
    @param[in]
    max_levels  rocblas_int. max_levels >= 1.\n
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
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The dimension of vector x.
    @param[inout]
    x           pointer to type. Array on the GPU of size at least n (size depends on the value of incx).\n
                On entry, the vector x.
                On exit, each entry is overwritten with its conjugate value.
    @param[in]
    incx        rocblas_int. incx != 0.\n
                The distance between two consecutive elements of x.
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
    Row interchanges are done one by one. If \f$\text{ipiv}[k_1 + (j - k_1) \cdot \text{abs}(\text{incx})] = r\f$, then the j-th row of A
    will be interchanged with the r-th row of A, for \f$j = k_1,k_1+1,\dots,k_2\f$. Indices \f$k_1\f$ and \f$k_2\f$ are 1-based indices.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n. \n
                On entry, the matrix to which the row
                interchanges will be applied. On exit, the resulting permuted matrix.
    @param[in]
    lda         rocblas_int. lda > 0.\n
                The leading dimension of the array A.
    @param[in]
    k1          rocblas_int. k1 > 0.\n
                The k_1 index. It is the first element of ipiv for which a row interchange will
                be done. This is a 1-based index.
    @param[in]
    k2          rocblas_int. k2 > k1 > 0.\n
                The k_2 index. k_2 - k_1 + 1 is the number of elements of ipiv for which a row
                interchange will be done. This is a 1-based index.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension at least k_1 + (k_2 - k_1)*abs(incx).\n
                The vector of pivot indices. Only the elements in positions
                k_1 through k_1 + (k_2 - k_1)*abs(incx) of this vector are accessed.
                Elements of ipiv are considered 1-based.
    @param[in]
    incx        rocblas_int. incx != 0.\n
                The distance between successive values of ipiv.  If incx
                is negative, the pivots are applied in reverse order.
    **************************************************************************/

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
    \brief LARFG generates a Householder reflector H of order n.

    \details
    The reflector H is such that

    \f[
        H'\left[\begin{array}{c}
        \text{alpha}\\
        x
        \end{array}\right]=\left[\begin{array}{c}
        \text{beta}\\
        0
        \end{array}\right]
    \f]

    where x is an n-1 vector, and alpha and beta are scalars. Matrix H can be
    generated as

    \f[
        H = I - \text{tau}\left[\begin{array}{c}
        1\\
        v
        \end{array}\right]\left[\begin{array}{cc}
        1 & v'
        \end{array}\right]
    \f]

    where v is an n-1 vector, and tau is a scalar known as the Householder scalar. The vector

    \f[
        \bar{v}=\left[\begin{array}{c}
        1\\
        v
        \end{array}\right]
    \f]

    is the Householder vector associated with the reflection.

    \note
    The matrix H is orthogonal/unitary (i.e. \f$H'H=HH'=I\f$). It is symmetric when real (i.e. \f$H^T=H\f$), but not Hermitian when complex
    (i.e. \f$H^H\neq H\f$ in general).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order (size) of reflector H.
    @param[inout]
    alpha       pointer to type. A scalar on the GPU.\n
                On entry, the scalar alpha.
                On exit, it is overwritten with beta.
    @param[inout]
    x           pointer to type. Array on the GPU of size at least n-1 (size depends on the value of incx).\n
                On entry, the vector x,
                On exit, it is overwritten with vector v.
    @param[in]
    incx        rocblas_int. incx > 0.\n
                The distance between two consecutive elements of x.
    @param[out]
    tau         pointer to type. A scalar on the GPU.\n
                The Householder scalar tau.
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
    \brief LARFT generates the triangular factor T of a block reflector H of
    order n.

    \details
    The block reflector H is defined as the product of k Householder matrices

    \f[
        \begin{array}{cl}
        H = H_1H_2\cdots H_k & \: \text{if direct indicates forward direction, or} \\
        H = H_k\cdots H_2H_1 & \: \text{if direct indicates backward direction}
        \end{array}
    \f]

    The triangular factor T is upper triangular in the forward direction and lower triangular in the backward direction.
    If storev is column-wise, then

    \f[
        H = I - VTV'
    \f]

    where the i-th column of matrix V contains the Householder vector associated with \f$H_i\f$. If storev is row-wise, then

    \f[
        H = I - V'TV
    \f]

    where the i-th row of matrix V contains the Householder vector associated with \f$H_i\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    direct      #rocblas_direct.\n
                Specifies the direction in which the Householder matrices are applied.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies how the Householder vectors are stored in matrix V.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order (size) of the block reflector.
    @param[in]
    k           rocblas_int. k >= 1.\n
                The number of Householder matrices forming H.
    @param[in]
    V           pointer to type. Array on the GPU of size ldv*k if column-wise, or ldv*n if row-wise.\n
                The matrix of Householder vectors.
    @param[in]
    ldv         rocblas_int. ldv >= n if column-wise, or ldv >= k if row-wise.\n
                Leading dimension of V.
    @param[in]
    tau         pointer to type. Array of k scalars on the GPU.\n
                The vector of all the Householder scalars.
    @param[out]
    T           pointer to type. Array on the GPU of dimension ldt*k.\n
                The triangular factor. T is upper triangular if direct indicates forward direction, otherwise it is
                lower triangular. The rest of the array is not used.
    @param[in]
    ldt         rocblas_int. ldt >= k.\n
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
    The Householder reflector H, of order m or n, is to be applied to an m-by-n matrix A
    from the left or the right, depending on the value of side. H is given by

    \f[
        H = I - \text{alpha}\cdot xx'
    \f]

    where alpha is the Householder scalar and x is a Householder vector. H is never actually computed.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Determines whether H is applied from the left or the right.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of A.
    @param[in]
    x           pointer to type. Array on the GPU of size at least 1 + (m-1)*abs(incx) if left side, or
                at least 1 + (n-1)*abs(incx) if right side.\n
                The Householder vector x.
    @param[in]
    incx        rocblas_int. incx != 0.\n
                Distance between two consecutive elements of x.
                If incx < 0, the elements of x are indexed in reverse order.
    @param[in]
    alpha       pointer to type. A scalar on the GPU.\n
                The Householder scalar. If alpha = 0, then H = I (A will remain the same; x is never used)
    @param[inout]
    A           pointer to type. Array on the GPU of size lda*n.\n
                On entry, the matrix A. On exit, it is overwritten with
                H*A (or A*H).
    @param[in]
    lda         rocblas_int. lda >= m.\n
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

    \f[
        \begin{array}{cl}
        HA & \: \text{(No transpose from the left),}\\
        H'A & \:  \text{(Transpose or conjugate transpose from the left),}\\
        AH & \: \text{(No transpose from the right), or}\\
        AH' & \: \text{(Transpose or conjugate transpose from the right).}
        \end{array}
    \f]

    The block reflector H is defined as the product of k Householder matrices as

    \f[
        \begin{array}{cl}
        H = H_1H_2\cdots H_k & \: \text{if direct indicates forward direction, or} \\
        H = H_k\cdots H_2H_1 & \: \text{if direct indicates backward direction}
        \end{array}
    \f]

    H is never stored. It is calculated as

    \f[
        H = I - VTV'
    \f]

    where the i-th column of matrix V contains the Householder vector associated with \f$H_i\f$, if storev is column-wise; or

    \f[
        H = I - V'TV
    \f]

    where the i-th row of matrix V contains the Householder vector associated with \f$H_i\f$, if storev is row-wise.
    T is the associated triangular factor as computed by \ref rocsolver_slarft "LARFT".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply H.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the block reflector or its transpose/conjugate transpose is to be applied.
    @param[in]
    direct      #rocblas_direct.\n
                Specifies the direction in which the Householder matrices are to be applied to generate H.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies how the Householder vectors are stored in matrix V.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix A.
    @param[in]
    k           rocblas_int. k >= 1.\n
                The number of Householder matrices.
    @param[in]
    V           pointer to type. Array on the GPU of size ldv*k if column-wise, ldv*n if row-wise and applying from the right,
                or ldv*m if row-wise and applying from the left.\n
                The matrix of Householder vectors.
    @param[in]
    ldv         rocblas_int. ldv >= k if row-wise, ldv >= m if column-wise and applying from the left, or ldv >= n if
                column-wise and applying from the right.\n
                Leading dimension of V.
    @param[in]
    T           pointer to type. Array on the GPU of dimension ldt*k.\n
                The triangular factor of the block reflector.
    @param[in]
    ldt         rocblas_int. ldt >= k.\n
                The leading dimension of T.
    @param[inout]
    A           pointer to type. Array on the GPU of size lda*n.\n
                On entry, the matrix A. On exit, it is overwritten with
                H*A, A*H, H'*A, or A*H'.
    @param[in]
    lda         rocblas_int. lda >= m.\n
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
    The reduced form is given by:

    \f[
        B = Q'AP
    \f]

    where the leading k-by-k block of B is upper bidiagonal if m >= n, or lower bidiagonal if m < n. Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_k, & \text{and} \\
        P = G_1G_2\cdots G_k.
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ and \f$G_i\f$ is given by

    \f[
        \begin{array}{cl}
        H_i = I - \text{tauq}[i]\cdot v_iv_i', & \text{and} \\
        G_i = I - \text{taup}[i]\cdot u_iu_i'.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i]=1\f$;
    while the first i elements of the Householder vector \f$u_i\f$ are zero, and  \f$u_i[i+1]=1\f$.
    If m < n, the first i elements of the Householder vector  \f$v_i\f$ are zero, and  \f$v_i[i+1]=1\f$;
    while the first i-1 elements of the Householder vector \f$u_i\f$ are zero, and \f$u_i[i]=1\f$.

    The unreduced part of the matrix A can be updated using the block update

    \f[
        A = A - VY' - XU'
    \f]

    where V and U are the m-by-k and n-by-k matrices formed with the vectors \f$v_i\f$ and \f$u_i\f$, respectively.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[in]
    k           rocblas_int. min(m,n) >= k >= 0.\n
                The number of leading rows and columns of matrix A that will be reduced.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be reduced.
                On exit, the first k elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n), contain the bidiagonal form B.
                If m >= n, the elements below the diagonal of the first k columns are the possibly non-zero elements
                of the Householder vectors associated with Q, while the elements above the
                superdiagonal of the first k rows are the n - i - 1 possibly non-zero elements of the Householder vectors related to P.
                If m < n, the elements below the subdiagonal of the first k columns are the m - i - 1 possibly non-zero
                elements of the Householder vectors related to Q, while the elements above the
                diagonal of the first k rows are the n - i possibly non-zero elements of the vectors associated with P.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension k.\n
                The diagonal elements of B.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension k.\n
                The off-diagonal elements of B.
    @param[out]
    tauq        pointer to type. Array on the GPU of dimension k.\n
                The Householder scalars associated with matrix Q.
    @param[out]
    taup        pointer to type. Array on the GPU of dimension k.\n
                The Householder scalars associated with matrix P.
    @param[out]
    X           pointer to type. Array on the GPU of dimension ldx*k.\n
                The m-by-k matrix needed to update the unreduced part of A.
    @param[in]
    ldx         rocblas_int. ldx >= m.\n
                The leading dimension of X.
    @param[out]
    Y           pointer to type. Array on the GPU of dimension ldy*k.\n
                The n-by-k matrix needed to update the unreduced part of A.
    @param[in]
    ldy         rocblas_int. ldy >= n.\n
                The leading dimension of Y.
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

    \f[
        T = Q'AQ
    \f]

    If uplo is lower, the first k rows and columns of T form the tridiagonal block. If uplo is upper, then the last
    k rows and columns of T form the tridiagonal block. Q is an orthogonal/unitary matrix represented as the
    product of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_k & \text{if uplo indicates lower, or}\\
        Q = H_nH_{n-1}\cdots H_{n-k+1} & \text{if uplo is upper}.
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{tau}[i]\cdot v_iv_i'
    \f]

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$. If uplo is upper,
    the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    The unreduced part of the matrix A can be updated using a rank update of the form:

    \f[
        A = A - VW' - WV'
    \f]

    where V is the n-by-k matrix formed by the vectors \f$v_i\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[in]
    k           rocblas_int. 0 <= k <= n.\n
                The number of rows and columns of the matrix A to be reduced.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the n-by-n matrix to be reduced.
                On exit, if uplo is lower, the first k columns have been reduced to tridiagonal form
                (given in the diagonal elements of A and the array E), the elements below the diagonal
                contain the possibly non-zero entries of the Householder vectors associated with Q, stored as columns.
                If uplo is upper, the last k columns have been reduced to tridiagonal form
                (given in the diagonal elements of A and the array E), the elements above the diagonal
                contain the possibly non-zero entries of the Householder vectors associated with Q, stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                If upper (lower), the last (first) k elements of E are the off-diagonal elements of the
                computed tridiagonal block.
    @param[out]
    tau         pointer to type. Array on the GPU of dimension n-1.\n
                If upper (lower), the last (first) k elements of tau are the Householder scalars related to Q.
    @param[out]
    W           pointer to type. Array on the GPU of dimension ldw*k.\n
                The n-by-k matrix needed to update the unreduced part of A.
    @param[in]
    ldw         rocblas_int. ldw >= n.\n
                The leading dimension of W.
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
    \brief LASYF computes a partial factorization of a symmetric matrix \f$A\f$
    using Bunch-Kaufman diagonal pivoting.

    \details
    The partial factorization has the form

    \f[
        A = \left[ \begin{array}{cc}
        I & U_{12} \\
        0 & U_{22}
        \end{array} \right] \left[ \begin{array}{cc}
        A_{11} & 0 \\
        0 & D
        \end{array} \right] \left[ \begin{array}{cc}
        I & 0 \\
        U_{12}^T & U_{22}^T
        \end{array} \right]
    \f]

    or

    \f[
        A = \left[ \begin{array}{cc}
        L_{11} & 0 \\
        L_{21} & I
        \end{array} \right] \left[ \begin{array}{cc}
        D & 0 \\
        0 & A_{22}
        \end{array} \right] \left[ \begin{array}{cc}
        L_{11}^T & L_{21}^T \\
        0 & I
        \end{array} \right]
    \f]

    depending on the value of uplo. The order of the block diagonal matrix \f$D\f$
    is either \f$nb\f$ or \f$nb-1\f$, and is returned in the argument \f$kb\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[in]
    nb          rocblas_int. 2 <= nb <= n.\n
                The number of columns of A to be factored.
    @param[out]
    kb          pointer to a rocblas_int on the GPU.\n
                The number of columns of A that were actually factored (either nb or
                nb-1).
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrix A to be factored.
                On exit, the partially factored matrix.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                If uplo is upper, then only the last kb elements of ipiv will be
                set. For n - kb < k <= n, if ipiv[k] > 0 then rows and columns k
                and ipiv[k] were interchanged and D[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv[k] = ipiv[k-1] < 0, then rows and columns k-1
                and -ipiv[k] were interchanged and D[k-1,k-1] to D[k,k] is a 2-by-2
                diagonal block.
                If uplo is lower, then only the first kb elements of ipiv will be
                set. For 1 <= k <= kb, if ipiv[k] > 0 then rows and columns k
                and ipiv[k] were interchanged and D[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv[k] = ipiv[k+1] < 0, then rows and columns k+1
                and -ipiv[k] were interchanged and D[k,k] to D[k+1,k+1] is a 2-by-2
                diagonal block.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, D is singular. D[i,i] is the first diagonal zero.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slasyf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nb,
                                                 rocblas_int* kb,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlasyf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nb,
                                                 rocblas_int* kb,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_clasyf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nb,
                                                 rocblas_int* kb,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zlasyf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nb,
                                                 rocblas_int* kb,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief ORG2R generates an m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k.
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqrf "GEQRF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQRF", with the Householder vectors in the first k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
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
    \brief UNG2R generates an m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqrf "GEQRF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQRF", with the Householder vectors in the first k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
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
    \brief ORGQR generates an m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqrf "GEQRF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQRF", with the Householder vectors in the first k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
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
    \brief UNGQR generates an m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first n columns of the product of k Householder
    reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqrf "GEQRF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQRF", with the Householder vectors in the first k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
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
    \brief ORGL2 generates an m-by-n Matrix Q with orthonormal rows.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgelqf "GELQF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GELQF", with the Householder vectors in the first k rows.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
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
    \brief UNGL2 generates an m-by-n complex Matrix Q with orthonormal rows.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

    \f[
        Q = H_k^HH_{k-1}^H\cdots H_1^H
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgelqf "GELQF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GELQF", with the Householder vectors in the first k rows.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
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
    \brief ORGLQ generates an m-by-n Matrix Q with orthonormal rows.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgelqf "GELQF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GELQF", with the Householder vectors in the first k rows.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
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
    \brief UNGLQ generates an m-by-n complex Matrix Q with orthonormal rows.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the first m rows of the product of k Householder
    reflectors of order n

    \f[
        Q = H_k^HH_{k-1}^H\cdots H_1^H
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgelqf "GELQF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GELQF", with the Householder vectors in the first k rows.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
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
    \brief ORG2L generates an m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqlf "GEQLF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQLF", with the Householder vectors in the last k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqlf "GEQLF".
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
    \brief UNG2L generates an m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqlf "GEQLF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQLF", with the Householder vectors in the last k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqlf "GEQLF".
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
    \brief ORGQL generates an m-by-n Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the last n column of the product of k Householder
    reflectors of order m

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqlf "GEQLF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQLF", with the Householder vectors in the last k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqlf "GEQLF".
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
    \brief UNGQL generates an m-by-n complex Matrix Q with orthonormal columns.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is defined as the last n columns of the product of k
    Householder reflectors of order m

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgeqlf "GEQLF".

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
                On entry, the matrix A as returned by \ref rocsolver_sgeqrf "GEQLF", with the Householder vectors in the last k columns.
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqlf "GEQLF".
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
    \brief ORGBR generates an m-by-n Matrix Q with orthonormal rows or columns.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns. If m >= k, Q is defined as the first
    n columns of the product of k Householder reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    If m < k, Q is defined as the product of Householder reflectors of order m

    \f[
        Q = H_1H_2\cdots H_{m-1}
    \f]

    On the other hand, if storev is row-wise, then the matrix Q has orthonormal rows. If n > k, Q is defined as the
    first m rows of the product of k Householder reflectors of order n

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    If n <= k, Q is defined as the product of Householder reflectors of order n

    \f[
        Q = H_{n-1}H_{n-2}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgebrd "GEBRD" in its arguments A and tauq or taup.

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
                The number of columns (if storev is column-wise) or rows (if row-wise) of the
                original matrix reduced by \ref rocsolver_sgebrd "GEBRD".
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the Householder vectors as returned by \ref rocsolver_sgebrd "GEBRD".
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension min(m,k) if column-wise, or min(n,k) if row-wise.\n
                The Householder scalars as returned by \ref rocsolver_sgebrd "GEBRD".
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
    \brief UNGBR generates an m-by-n complex Matrix Q with orthonormal rows or
    columns.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns. If m >= k, Q is defined as the first
    n columns of the product of k Householder reflectors of order m

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    If m < k, Q is defined as the product of Householder reflectors of order m

    \f[
        Q = H_1H_2\cdots H_{m-1}
    \f]

    On the other hand, if storev is row-wise, then the matrix Q has orthonormal rows. If n > k, Q is defined as the
    first m rows of the product of k Householder reflectors of order n

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    If n <= k, Q is defined as the product of Householder reflectors of order n

    \f[
        Q = H_{n-1}H_{n-2}\cdots H_1
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by \ref rocsolver_sgebrd "GEBRD" in its arguments A and tauq or taup.

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
                The number of columns (if storev is column-wise) or rows (if row-wise) of the
                original matrix reduced by \ref rocsolver_sgebrd "GEBRD".
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the Householder vectors as returned by \ref rocsolver_sgebrd "GEBRD".
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension min(m,k) if column-wise, or min(n,k) if row-wise.\n
                The Householder scalars as returned by \ref rocsolver_sgebrd "GEBRD".
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
    \brief ORGTR generates an n-by-n orthogonal Matrix Q.

    \details
    Q is defined as the product of n-1 Householder reflectors of order n. If
    uplo indicates upper, then Q has the form

    \f[
        Q = H_{n-1}H_{n-2}\cdots H_1
    \f]

    On the other hand, if uplo indicates lower, then Q has the form

    \f[
        Q = H_1H_2\cdots H_{n-1}
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by
    \ref rocsolver_ssytrd "SYTRD" in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the \ref rocsolver_ssytrd "SYTRD" factorization was upper or lower
                triangular. If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix Q.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the Householder vectors as returned
                by \ref rocsolver_ssytrd "SYTRD". On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars as returned by \ref rocsolver_ssytrd "SYTRD".
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
    \brief UNGTR generates an n-by-n unitary Matrix Q.

    \details
    Q is defined as the product of n-1 Householder reflectors of order n. If
    uplo indicates upper, then Q has the form

    \f[
        Q = H_{n-1}H_{n-2}\cdots H_1
    \f]

    On the other hand, if uplo indicates lower, then Q has the form

    \f[
        Q = H_1H_2\cdots H_{n-1}
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{ipiv}[i]\f$, as returned by
    \ref rocsolver_chetrd "HETRD" in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the \ref rocsolver_chetrd "HETRD" factorization was upper or lower
                triangular. If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix Q.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the Householder vectors as returned
                by \ref rocsolver_chetrd "HETRD". On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars as returned by \ref rocsolver_chetrd "HETRD".
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
    \brief ORM2R multiplies a matrix Q with orthonormal columns by a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2 \cdots H_k
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization \ref rocsolver_sgeqrf "GEQRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqrf "GEQRF"
                in the first k columns of its argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNM2R multiplies a complex matrix Q with orthonormal columns by a
    general m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization \ref rocsolver_sgeqrf "GEQRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqrf "GEQRF"
                in the first k columns of its argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORMQR multiplies a matrix Q with orthonormal columns by a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization \ref rocsolver_sgeqrf "GEQRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqrf "GEQRF"
                in the first k columns of its argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNMQR multiplies a complex matrix Q with orthonormal columns by a
    general m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the QR factorization \ref rocsolver_sgeqrf "GEQRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqrf "GEQRF"
                in the first k columns of its argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, or lda >= n if side is right. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgeqrf "GEQRF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORML2 multiplies a matrix Q with orthonormal rows by a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization \ref rocsolver_sgelqf "GELQF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                The Householder vectors as returned by \ref rocsolver_sgelqf "GELQF"
                in the first k rows of its argument A.
    @param[in]
    lda         rocblas_int. lda >= k. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNML2 multiplies a complex matrix Q with orthonormal rows by a general
    m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_k^HH_{k-1}^H\cdots H_1^H
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization \ref rocsolver_sgelqf "GELQF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                The Householder vectors as returned by \ref rocsolver_sgelqf "GELQF"
                in the first k rows of its argument A.
    @param[in]
    lda         rocblas_int. lda >= k. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORMLQ multiplies a matrix Q with orthonormal rows by a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization \ref rocsolver_sgelqf "GELQF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                The Householder vectors as returned by \ref rocsolver_sgelqf "GELQF"
                in the first k rows of its argument A.
    @param[in]
    lda         rocblas_int. lda >= k. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNMLQ multiplies a complex matrix Q with orthonormal rows by a general
    m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_k^HH_{k-1}^H\cdots H_1^H
    \f]

    of order m if applying from the left, or n if applying from the right. Q is never stored, it is
    calculated from the Householder vectors and scalars returned by the LQ factorization \ref rocsolver_sgelqf "GELQF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*m if side is left, or lda*n if side is right.\n
                The Householder vectors as returned by \ref rocsolver_sgelqf "GELQF"
                in the first k rows of its argument A.
    @param[in]
    lda         rocblas_int. lda >= k. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by \ref rocsolver_sgelqf "GELQF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORM2L multiplies a matrix Q with orthonormal columns by a general m-by-n
    matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization \ref rocsolver_sgeqlf "GEQLF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be
                applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqlf "GEQLF" in the last k columns of its
                argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by
                \ref rocsolver_sgeqlf "GEQLF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNM2L multiplies a complex matrix Q with orthonormal columns by a
    general m-by-n matrix C.

    \details
    (This is the unblocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization \ref rocsolver_sgeqlf "GEQLF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate
                transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqlf "GEQLF" in the last k columns of its
                argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by
                \ref rocsolver_sgeqlf "GEQLF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORMQL multiplies a matrix Q with orthonormal columns by a general m-by-n
    matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization \ref rocsolver_sgeqlf "GEQLF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be
                applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqlf "GEQLF" in the last k columns of its
                argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by
                \ref rocsolver_sgeqlf "GEQLF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNMQL multiplies a complex matrix Q with orthonormal columns by a
    general m-by-n matrix C.

    \details
    (This is the blocked version of the algorithm).

    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    Q is defined as the product of k Householder reflectors

    \f[
        Q = H_kH_{k-1}\cdots H_1
    \f]

    of order m if applying from the left, or n if applying from the right. Q is
    never stored, it is calculated from the Householder vectors and scalars
    returned by the QL factorization \ref rocsolver_sgeqlf "GEQLF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate
                transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0; k <= m if side is left, k <= n if side is right.\n
                The number of Householder reflectors that form Q.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*k.\n
                The Householder vectors as returned by \ref rocsolver_sgeqlf "GEQLF" in the last k columns of its
                argument A.
    @param[in]
    lda         rocblas_int. lda >= m if side is left, lda >= n if side is right.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least k.\n
                The Householder scalars as returned by
                \ref rocsolver_sgeqlf "GEQLF".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORMBR multiplies a matrix Q with orthonormal rows or columns by a
    general m-by-n matrix C.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns.
    If storev is row-wise, then the matrix Q has orthonormal rows.
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    The order q of the orthogonal matrix Q is q = m if applying from the left, or q = n if applying from the right.

    When storev is column-wise, if q >= k, then Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k,
    \f]

    and if q < k, then Q is defined as the product

    \f[
        Q = H_1H_2\cdots H_{q-1}.
    \f]

    When storev is row-wise, if q > k, then Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k,
    \f]

    and if q <= k, Q is defined as the product

    \f[
        Q = H_1H_2\cdots H_{q-1}.
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors and scalars as returned by \ref rocsolver_sgebrd "GEBRD" in its arguments A and tauq or taup.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies whether to work column-wise or row-wise.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0.\n
                The number of columns (if storev is column-wise) or rows (if row-wise) of the
                original matrix reduced by \ref rocsolver_sgebrd "GEBRD".
    @param[in]
    A           pointer to type. Array on the GPU of size lda*min(q,k) if column-wise, or lda*q if row-wise.\n
                The Householder vectors as returned by \ref rocsolver_sgebrd "GEBRD".
    @param[in]
    lda         rocblas_int. lda >= q if column-wise, or lda >= min(q,k) if row-wise. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least min(q,k).\n
                The Householder scalars as returned by \ref rocsolver_sgebrd "GEBRD".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNMBR multiplies a complex matrix Q with orthonormal rows or columns by
    a general m-by-n matrix C.

    \details
    If storev is column-wise, then the matrix Q has orthonormal columns.
    If storev is row-wise, then the matrix Q has orthonormal rows.
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    The order q of the unitary matrix Q is q = m if applying from the left, or q = n if applying from the right.

    When storev is column-wise, if q >= k, then Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k,
    \f]

    and if q < k, then Q is defined as the product

    \f[
        Q = H_1H_2\cdots H_{q-1}.
    \f]

    When storev is row-wise, if q > k, then Q is defined as the product of k Householder reflectors

    \f[
        Q = H_1H_2\cdots H_k,
    \f]

    and if q <= k, Q is defined as the product

    \f[
        Q = H_1H_2\cdots H_{q-1}.
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its corresponding
    Householder vectors and scalars as returned by \ref rocsolver_sgebrd "GEBRD" in its arguments A and tauq or taup.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    storev      #rocblas_storev.\n
                Specifies whether to work column-wise or row-wise.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    k           rocblas_int. k >= 0.\n
                The number of columns (if storev is column-wise) or rows (if row-wise) of the
                original matrix reduced by \ref rocsolver_sgebrd "GEBRD".
    @param[in]
    A           pointer to type. Array on the GPU of size lda*min(q,k) if column-wise, or lda*q if row-wise.\n
                The Householder vectors as returned by \ref rocsolver_sgebrd "GEBRD".
    @param[in]
    lda         rocblas_int. lda >= q if column-wise, or lda >= min(q,k) if row-wise. \n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least min(q,k).\n
                The Householder scalars as returned by \ref rocsolver_sgebrd "GEBRD".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief ORMTR multiplies an orthogonal matrix Q by a general m-by-n matrix C.

    \details
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^TC & \: \text{Transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^T & \: \text{Transpose from the right.}
        \end{array}
    \f]

    The order q of the orthogonal matrix Q is q = m if applying from the left, or
    q = n if applying from the right.

    Q is defined as a product of q-1 Householder reflectors. If
    uplo indicates upper, then Q has the form

    \f[
        Q = H_{q-1}H_{q-2}\cdots H_1.
    \f]

    On the other hand, if uplo indicates lower, then Q has the form

    \f[
        Q = H_1H_2\cdots H_{q-1}
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors and scalars as returned by
    \ref rocsolver_ssytrd "SYTRD" in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the \ref rocsolver_ssytrd "SYTRD" factorization was upper or
                lower triangular. If uplo indicates lower (or upper), then the upper (or
                lower) part of A is not used.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its transpose is to be
                applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*q.\n
                On entry, the Householder vectors as
                returned by \ref rocsolver_ssytrd "SYTRD".
    @param[in]
    lda         rocblas_int. lda >= q.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least q-1.\n
                The Householder scalars as returned by
                \ref rocsolver_ssytrd "SYTRD".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief UNMTR multiplies a unitary matrix Q by a general m-by-n matrix C.

    \details
    The matrix Q is applied in one of the following forms, depending on
    the values of side and trans:

    \f[
        \begin{array}{cl}
        QC & \: \text{No transpose from the left,}\\
        Q^HC & \: \text{Conjugate transpose from the left,}\\
        CQ & \: \text{No transpose from the right, and}\\
        CQ^H & \: \text{Conjugate transpose from the right.}
        \end{array}
    \f]

    The order q of the unitary matrix Q is q = m if applying from the left, or
    q = n if applying from the right.

    Q is defined as a product of q-1 Householder reflectors. If
    uplo indicates upper, then Q has the form

    \f[
        Q = H_{q-1}H_{q-2}\cdots H_1.
    \f]

    On the other hand, if uplo indicates lower, then Q has the form

    \f[
        Q = H_1H_2\cdots H_{q-1}
    \f]

    The Householder matrices \f$H_i\f$ are never stored, they are computed from its
    corresponding Householder vectors and scalars as returned by
    \ref rocsolver_chetrd "HETRD" in its arguments A and tau.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        rocblas_side.\n
                Specifies from which side to apply Q.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the \ref rocsolver_chetrd "HETRD" factorization was upper or
                lower triangular. If uplo indicates lower (or upper), then the upper (or
                lower) part of A is not used.
    @param[in]
    trans       rocblas_operation.\n
                Specifies whether the matrix Q or its conjugate
                transpose is to be applied.
    @param[in]
    m           rocblas_int. m >= 0.\n
                Number of rows of matrix C.
    @param[in]
    n           rocblas_int. n >= 0.\n
                Number of columns of matrix C.
    @param[in]
    A           pointer to type. Array on the GPU of size lda*q.\n
                On entry, the Householder vectors as
                returned by \ref rocsolver_chetrd "HETRD".
    @param[in]
    lda         rocblas_int. lda >= q.\n
                Leading dimension of A.
    @param[in]
    ipiv        pointer to type. Array on the GPU of dimension at least q-1.\n
                The Householder scalars as returned by
                \ref rocsolver_chetrd "HETRD".
    @param[inout]
    C           pointer to type. Array on the GPU of size ldc*n.\n
                On entry, the matrix C. On exit, it is overwritten with
                Q*C, C*Q, Q'*C, or C*Q'.
    @param[in]
    ldc         rocblas_int. ldc >= m.\n
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
    \brief BDSQR computes the singular value decomposition (SVD) of an
    n-by-n bidiagonal matrix B, using the implicit QR algorithm.

    \details
    The SVD of B has the form:

    \f[
        B = QSP'
    \f]

    where S is the n-by-n diagonal matrix of singular values of B, the columns of Q are the left
    singular vectors of B, and the columns of P are its right singular vectors.

    The computation of the singular vectors is optional; this function accepts input matrices
    U (of size nu-by-n) and V (of size n-by-nv) that are overwritten with \f$UQ\f$ and \f$P'V\f$. If nu = 0
    no left vectors are computed; if nv = 0 no right vectors are computed.

    Optionally, this function can also compute \f$Q'C\f$ for a given n-by-nc input matrix C.

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
                On entry, the matrix V. On exit, it is overwritten with P'*V.
                (Not referenced if nv = 0).
    @param[in]
    ldv         rocblas_int. ldv >= n if nv > 0, or ldv >=1 if nv = 0.\n
                The leading dimension of V.
    @param[inout]
    U           pointer to type. Array on the GPU of dimension ldu*n.\n
                On entry, the matrix U. On exit, it is overwritten with U*Q.
                (Not referenced if nu = 0).
    @param[in]
    ldu         rocblas_int. ldu >= nu.\n
                The leading dimension of U.
    @param[inout]
    C           pointer to type. Array on the GPU of dimension ldc*nc.\n
                On entry, the matrix C. On exit, it is overwritten with Q'*C.
                (Not referenced if nc = 0).
    @param[in]
    ldc         rocblas_int. ldc >= n if nc > 0, or ldc >=1 if nc = 0.\n
                The leading dimension of C.
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
    diagonal elements D and the array of symmetric off-diagonal elements E.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the tridiagonal matrix.
    @param[inout]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On entry, the diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, the eigenvalues in increasing order.
                If info > 0, the diagonal elements of a tridiagonal matrix
                that is similar to the original matrix (i.e. has the same
                eigenvalues).
    @param[inout]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                On entry, the off-diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, this array converges to zero.
                If info > 0, the off-diagonal elements of a tridiagonal matrix
                that is similar to the original matrix (i.e. has the same
                eigenvalues).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
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
    diagonal elements D and the array of symmetric off-diagonal elements E.
    When D and E correspond to the tridiagonal form of a full symmetric/Hermitian matrix, as returned by, e.g.,
    \ref rocsolver_ssytrd "SYTRD" or \ref rocsolver_chetrd "HETRD", the eigenvectors of the original matrix can also
    be computed, depending on the value of evect.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies how the eigenvectors are computed.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the tridiagonal matrix.
    @param[inout]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On entry, the diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, the eigenvalues in increasing order.
                If info > 0, the diagonal elements of a tridiagonal matrix
                that is similar to the original matrix (i.e. has the same
                eigenvalues).
    @param[inout]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                On entry, the off-diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, this array converges to zero.
                If info > 0, the off-diagonal elements of a tridiagonal matrix
                that is similar to the original matrix (i.e. has the same
                eigenvalues).
    @param[inout]
    C           pointer to type. Array on the GPU of dimension ldc*n.\n
                On entry, if evect is original, the orthogonal/unitary matrix
                used for the reduction to tridiagonal form as returned by, e.g.,
                \ref rocsolver_sorgtr "ORGTR" or \ref rocsolver_cungtr "UNGTR".
                On exit, it is overwritten with the eigenvectors of the original
                symmetric/Hermitian matrix (if evect is original), or the
                eigenvectors of the tridiagonal matrix (if evect is tridiagonal).
                (Not referenced if evect is none).
    @param[in]
    ldc         rocblas_int. ldc >= n if evect is original or tridiagonal.\n
                Specifies the leading dimension of C.
                (Not referenced if evect is none).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, STEQR did not converge. i elements of E did not
                converge to zero.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssteqr(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 float* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsteqr(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 double* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_csteqr(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsteqr(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief STEDC computes the eigenvalues and (optionally) eigenvectors of
    a symmetric tridiagonal matrix.

    \details
    This function uses the divide and conquer method to compute the eigenvectors.
    The eigenvalues are returned in increasing order.

    The matrix is not represented explicitly, but rather as the array of
    diagonal elements D and the array of symmetric off-diagonal elements E.
    When D and E correspond to the tridiagonal form of a full symmetric/Hermitian matrix, as returned by, e.g.,
    \ref rocsolver_ssytrd "SYTRD" or \ref rocsolver_chetrd "HETRD", the eigenvectors of the original matrix can also
    be computed, depending on the value of evect.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies how the eigenvectors are computed.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the tridiagonal matrix.
    @param[inout]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On entry, the diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, the eigenvalues in increasing order.
    @param[inout]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                On entry, the off-diagonal elements of the tridiagonal matrix.
                On exit, if info = 0, the values of this array are destroyed.
    @param[inout]
    C           pointer to type. Array on the GPU of dimension ldc*n.\n
                On entry, if evect is original, the orthogonal/unitary matrix
                used for the reduction to tridiagonal form as returned by, e.g.,
                \ref rocsolver_sorgtr "ORGTR" or \ref rocsolver_cungtr "UNGTR".
                On exit, if info = 0, it is overwritten with the eigenvectors of the original
                symmetric/Hermitian matrix (if evect is original), or the
                eigenvectors of the tridiagonal matrix (if evect is tridiagonal).
                (Not referenced if evect is none).
    @param[in]
    ldc         rocblas_int. ldc >= n if evect is original or tridiagonal.\n
                Specifies the leading dimension of C. (Not referenced if evect is none).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, STEDC failed to compute an eigenvalue on the sub-matrix formed by
                the rows and columns info/(n+1) through mod(info,n+1).
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sstedc(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 float* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dstedc(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 double* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cstedc(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zstedc(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief STEIN computes the eigenvectors associated with a set of 
    provided eigenvalues of a symmetric tridiagonal matrix.

    \details
    The eigenvectors of the symmetric tridiagonal matrix are computed using
    inverse iteration.

    The matrix is not represented explicitly, but rather as the array of
    diagonal elements D and the array of symmetric off-diagonal elements E.
    The eigenvalues must be provided in the array W, as returned by \ref rocsolver_sstebz "STEBZ".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the tridiagonal matrix.
    @param[in]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The diagonal elements of the tridiagonal matrix.
    @param[in]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                The off-diagonal elements of the tridiagonal matrix.
    @param[in]
    nev         pointer to a rocblas_int on the GPU. 0 <= nev <= n.\n
                The number of provided eigenvalues, and the number of eigenvectors
                to be computed.
    @param[in]
    W           pointer to real type. Array on the GPU of dimension >= nev.\n
                A subset of nev eigenvalues of the tridiagonal matrix, as returned
                by \ref rocsolver_sstebz "STEBZ".
    @param[in]
    iblock      pointer to rocblas_int. Array on the GPU of dimension n.\n
                The block indices corresponding to each eigenvalue, as
                returned by \ref rocsolver_sstebz "STEBZ". If iblock[i] = k,
                then eigenvalue W[i] belongs to the k-th block from the top.
    @param[in]
    isplit      pointer to rocblas_int. Array on the GPU of dimension n.\n
                The splitting indices that divide the tridiagonal matrix into
                diagonal blocks, as returned by \ref rocsolver_sstebz "STEBZ".
                The k-th block stretches from the end of the (k-1)-th
                block (or the top left corner of the tridiagonal matrix,
                in the case of the 1st block) to the isplit[k]-th row/column.
    @param[out]
    Z           pointer to type. Array on the GPU of dimension ldz*nev.\n
                On exit, contains the eigenvectors of the tridiagonal matrix
                corresponding to the provided eigenvalues, stored by columns.
    @param[in]
    ldc         rocblas_int. ldz >= n.\n
                Specifies the leading dimension of Z.
    @param[out]
    ifail       pointer to rocblas_int. Array on the GPU of dimension n.\n
                If info = 0, the first nev elements of ifail are zero.
                Otherwise, contains the indices of those eigenvectors that failed
                to converge.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, i eigenvectors did not converge; their indices are stored in
                IFAIL.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sstein(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* nev,
                                                 float* W,
                                                 rocblas_int* iblock,
                                                 rocblas_int* isplit,
                                                 float* Z,
                                                 const rocblas_int ldz,
                                                 rocblas_int* ifail,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dstein(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_int* nev,
                                                 double* W,
                                                 rocblas_int* iblock,
                                                 rocblas_int* isplit,
                                                 double* Z,
                                                 const rocblas_int ldz,
                                                 rocblas_int* ifail,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cstein(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* nev,
                                                 float* W,
                                                 rocblas_int* iblock,
                                                 rocblas_int* isplit,
                                                 rocblas_float_complex* Z,
                                                 const rocblas_int ldz,
                                                 rocblas_int* ifail,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zstein(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_int* nev,
                                                 double* W,
                                                 rocblas_int* iblock,
                                                 rocblas_int* isplit,
                                                 rocblas_double_complex* Z,
                                                 const rocblas_int ldz,
                                                 rocblas_int* ifail,
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization has the form

    \f[
        A = LU
    \f]

    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetf2 "GETF2" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix A to be factored.
                On exit, the factors L and U from the factorization.
                The unit diagonal elements of L are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = L_jU_j
    \f]

    where \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetf2_batched "GETF2_BATCHED" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorizations.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = L_jU_j
    \f]

    where \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetf2_strided_batched "GETF2_STRIDED_BATCHED" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorization.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization has the form

    \f[
        A = LU
    \f]

    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetrf "GETRF" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix A to be factored.
                On exit, the factors L and U from the factorization.
                The unit diagonal elements of L are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = L_jU_j
    \f]

    where \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetrf_batched "GETRF_BATCHED" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorizations.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = L_jU_j
    \f]

    where \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    Note: Although this routine can offer better performance, Gaussian elimination without pivoting is not backward stable.
    If numerical accuracy is compromised, use the legacy-LAPACK-like API \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED" routines instead.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorization.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero element in the diagonal. The factorization from
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization has the form

    \f[
        A = PLU
    \f]

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix A to be factored.
                On exit, the factors L and U from the factorization.
                The unit diagonal elements of L are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= i <= min(m,n), the row i of the
                matrix was interchanged with row ipiv[i].
                Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = P_jL_jU_j
    \f]

    where \f$P_j\f$ is a permutation matrix, \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorizations.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors of pivot indices ipiv_j (corresponding to A_j).
                Dimension of ipiv_j is min(m,n).
                Elements of ipiv_j are 1-based indices.
                For each instance A_j in the batch and for 1 <= i <= min(m,n), the row i of the
                matrix A_j was interchanged with row ipiv_j[i].
                Matrix P_j of the factorization can be derived from ipiv_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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
    could be executed with small and mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = P_jL_jU_j
    \f]

    where \f$P_j\f$ is a permutation matrix, \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorization.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors of pivots indices ipiv_j (corresponding to A_j).
                Dimension of ipiv_j is min(m,n).
                Elements of ipiv_j are 1-based indices.
                For each instance A_j in the batch and for 1 <= i <= min(m,n), the row i of the
                matrix A_j was interchanged with row ipiv_j[i].
                Matrix P_j of the factorization can be derived from ipiv_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization has the form

    \f[
        A = PLU
    \f]

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix A to be factored.
                On exit, the factors L and U from the factorization.
                The unit diagonal elements of L are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= i <= min(m,n), the row i of the
                matrix was interchanged with row ipiv[i].
                Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = P_jL_jU_j
    \f]

    where \f$P_j\f$ is a permutation matrix, \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorizations.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors of pivot indices ipiv_j (corresponding to A_j).
                Dimension of ipiv_j is min(m,n).
                Elements of ipiv_j are 1-based indices.
                For each instance A_j in the batch and for 1 <= i <= min(m,n), the row i of the
                matrix A_j was interchanged with row ipiv_j[i].
                Matrix P_j of the factorization can be derived from ipiv_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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
    could be executed with mid-size matrices if optimizations are enabled (default option). For more details, see the
    "Tuning rocSOLVER performance" section of the Library Design Guide).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = P_jL_jU_j
    \f]

    where \f$P_j\f$ is a permutation matrix, \f$L_j\f$ is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and \f$U_j\f$ is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the factors L_j and U_j from the factorization.
                The unit diagonal elements of L_j are not stored.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors of pivots indices ipiv_j (corresponding to A_j).
                Dimension of ipiv_j is min(m,n).
                Elements of ipiv_j are 1-based indices.
                For each instance A_j in the batch and for 1 <= i <= min(m,n), the row i of the
                matrix A_j was interchanged with row ipiv_j[i].
                Matrix P_j of the factorization can be derived from ipiv_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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

    \f[
        A = Q\left[\begin{array}{c}
        R\\
        0
        \end{array}\right]
    \f]

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R; the elements below the diagonal are the last m - i elements
                of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        R_j\\
        0
        \end{array}\right]
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R_j. The elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        R_j\\
        0
        \end{array}\right]
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R_j. The elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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
    \brief GERQ2 computes a RQ factorization of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

    \f[
        A = \left[\begin{array}{cc}
        0 & R
        \end{array}\right] Q
    \f]

    where R is upper triangular (upper trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_1'H_2' \cdots H_k', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerq2(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GERQ2_BATCHED computes the RQ factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        0 & R_j
        \end{array}\right] Q_j
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}'H_{j_2}' \cdots H_{j_k}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last n-i elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R_j; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerq2_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GERQ2_STRIDED_BATCHED computes the RQ factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        0 & R_j
        \end{array}\right] Q_j
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}'H_{j_2}' \cdots H_{j_k}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last n-i elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R_j; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerq2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerq2_strided_batched(rocblas_handle handle,
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

    \f[
        A = Q\left[\begin{array}{c}
        0\\
        L
        \end{array}\right]
    \f]

    where L is lower triangular (lower trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_kH_{k-1}\cdots H_1, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the last m-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        0\\
        L_j
        \end{array}\right]
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_{j_k}H_{j_{k-1}}\cdots H_{j_1}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last m-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L_j; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        0\\
        L_j
        \end{array}\right]
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_{j_k}H_{j_{k-1}}\cdots H_{j_1}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last m-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L_j; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        A = \left[\begin{array}{cc}
        L & 0
        \end{array}\right] Q
    \f]

    where L is lower triangular (lower trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_k'H_{k-1}' \cdots H_1', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i' v_i
    \f]

    where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L; the elements above the diagonal are the last n - i elements
                of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        L_j & 0
        \end{array}\right] Q_j
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_k}'H_{j_{k-1}}' \cdots H_{j_1}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i}' v_{j_i}
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L_j. The elements above the diagonal are the last n - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        L_j & 0
        \end{array}\right] Q_j
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_k}'H_{j_{k-1}}' \cdots H_{j_1}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i}' v_{j_i}
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L_j. The elements above the diagonal are the last n - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        A = Q\left[\begin{array}{c}
        R\\
        0
        \end{array}\right]
    \f]

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R; the elements below the diagonal are the last m - i elements
                of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        R_j\\
        0
        \end{array}\right]
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R_j. The elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        R_j\\
        0
        \end{array}\right]
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the diagonal contain the
                factor R_j. The elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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
    \brief GERQF computes a RQ factorization of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

    \f[
        A = \left[\begin{array}{cc}
        0 & R
        \end{array}\right] Q
    \f]

    where R is upper triangular (upper trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_1'H_2' \cdots H_k', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerqf(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* ipiv);
//! @}

/*! @{
    \brief GERQF_BATCHED computes the RQ factorization of a batch of general
    m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        0 & R_j
        \end{array}\right] Q_j
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}'H_{j_2}' \cdots H_{j_k}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last n-i elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R_j; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerqf_batched(rocblas_handle handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* ipiv,
                                                         const rocblas_stride strideP,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief GERQF_STRIDED_BATCHED computes the RQ factorization of a batch of
    general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        0 & R_j
        \end{array}\right] Q_j
    \f]

    where \f$R_j\f$ is upper triangular (upper trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_1}'H_{j_2}' \cdots H_{j_k}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last n-i elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and above the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor R_j; the elements below the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgerqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgerqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgerqf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgerqf_strided_batched(rocblas_handle handle,
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

    \f[
        A = Q\left[\begin{array}{c}
        0\\
        L
        \end{array}\right]
    \f]

    where L is lower triangular (lower trapezoidal if m < n), and Q is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_kH_{k-1}\cdots H_1, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i v_i'
    \f]

    where the last m-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        0\\
        L_j
        \end{array}\right]
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_{j_k}H_{j_{k-1}}\cdots H_{j_1}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last m-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L_j; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = Q_j\left[\begin{array}{c}
        0\\
        L_j
        \end{array}\right]
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m < n), and \f$Q_j\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_{j_k}H_{j_{k-1}}\cdots H_{j_1}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i} v_{j_i}'
    \f]

    where the last m-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the (m-n)-th subdiagonal (when
                m >= n) or the (n-m)-th superdiagonal (when n > m) contain the
                factor L_j; the elements above the sub/superdiagonal are the first i - 1
                elements of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        A = \left[\begin{array}{cc}
        L & 0
        \end{array}\right] Q
    \f]

    where L is lower triangular (lower trapezoidal if m > n), and Q is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q = H_k'H_{k-1}' \cdots H_1', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{ipiv}[i] \cdot v_i' v_i
    \f]

    where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L; the elements above the diagonal are the last n - i elements
                of Householder vector v_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars.
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        L_j & 0
        \end{array}\right] Q_j
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_k}'H_{j_{k-1}}' \cdots H_{j_1}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i}' v_{j_i}
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L_j. The elements above the diagonal are the last n - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The factorization of matrix \f$A_j\f$ in the batch has the form

    \f[
        A_j = \left[\begin{array}{cc}
        L_j & 0
        \end{array}\right] Q_j
    \f]

    where \f$L_j\f$ is lower triangular (lower trapezoidal if m > n), and \f$Q_j\f$ is
    a n-by-n orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_j = H_{j_k}'H_{j_{k-1}}' \cdots H_{j_1}', \quad \text{with} \: k = \text{min}(m,n).
    \f]

    Each Householder matrices \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{ipiv}_j[i] \cdot v_{j_i}' v_{j_i}
    \f]

    where the first i-1 elements of Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on and below the diagonal contain the
                factor L_j. The elements above the diagonal are the last n - i elements
                of Householder vector v_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        B = Q'  A  P
    \f]

    where B is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_n\:  \text{and} \: P = G_1G_2\cdots G_{n-1}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q = H_1H_2\cdots H_{m-1}\:  \text{and} \: P = G_1G_2\cdots G_{m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ and \f$G_i\f$ is given by

    \f[
        \begin{array}{cl}
        H_i = I - \text{tauq}[i] \cdot v_i v_i', & \: \text{and}\\
        G_i = I - \text{taup}[i] \cdot u_i' u_i.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_i\f$ are zero, and \f$u_i[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_i\f$ are zero, and \f$u_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_i, and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_i.
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_i, and the elements above the
                diagonal are the last n - i elements of Householder vector u_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                specifies the leading dimension of A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension min(m,n).\n
                The diagonal elements of B.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
                The off-diagonal elements of B.
    @param[out]
    tauq        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars associated with matrix Q.
    @param[out]
    taup        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars associated with matrix P.
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

    For each instance in the batch, the bidiagonal form is given by:

    \f[
        B_j = Q_j'  A_j  P_j
    \f]

    where \f$B_j\f$ is upper bidiagonal if m >= n and lower bidiagonal if m < n, and \f$Q_j\f$ and
    \f$P_j\f$ are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_n}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_{n-1}}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{m-1}}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ and \f$G_{j_i}\f$ is given by

    \f[
        \begin{array}{cl}
        H_{j_i} = I - \text{tauq}_j[i] \cdot v_{j_i} v_{j_i}', & \: \text{and}\\
        G_{j_i} = I - \text{taup}_j[i] \cdot u_{j_i}' u_{j_i}.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B_j.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i), and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_(j_i).
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_(j_i), and the elements above the
                diagonal are the last n - i elements of Householder vector u_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of B_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of B_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq        pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
                Contains the vectors tauq_j of Householder scalars associated with matrices Q_j.
    @param[in]
    strideQ     rocblas_stride.\n
                Stride from the start of one vector tauq_j to the next one tauq_(j+1).
                There is no restriction for the value
                of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors taup_j of Householder scalars associated with matrices P_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector taup_j to the next one taup_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the bidiagonal form is given by:

    \f[
        B_j = Q_j'  A_j  P_j
    \f]

    where \f$B_j\f$ is upper bidiagonal if m >= n and lower bidiagonal if m < n, and \f$Q_j\f$ and
    \f$P_j\f$ are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_n}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_{n-1}}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{m-1}}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ and \f$G_{j_i}\f$ is given by

    \f[
        \begin{array}{cl}
        H_{j_i} = I - \text{tauq}_j[i] \cdot v_{j_i} v_{j_i}', & \: \text{and}\\
        G_{j_i} = I - \text{taup}_j[i] \cdot u_{j_i}' u_{j_i}.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B_j.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i), and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_(j_i).
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_(j_i), and the elements above the
                diagonal are the last n - i elements of Householder vector u_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of B_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of B_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq        pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
                Contains the vectors tauq_j of Householder scalars associated with matrices Q_j.
    @param[in]
    strideQ     rocblas_stride.\n
                Stride from the start of one vector tauq_j to the next one tauq_(j+1).
                There is no restriction for the value
                of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors taup_j of Householder scalars associated with matrices P_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector taup_j to the next one taup_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        B = Q'  A  P
    \f]

    where B is upper bidiagonal if m >= n and lower bidiagonal if m < n, and Q and
    P are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_n\:  \text{and} \: P = G_1G_2\cdots G_{n-1}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q = H_1H_2\cdots H_{m-1}\:  \text{and} \: P = G_1G_2\cdots G_{m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ and \f$G_i\f$ is given by

    \f[
        \begin{array}{cl}
        H_i = I - \text{tauq}[i] \cdot v_i v_i', & \: \text{and}\\
        G_i = I - \text{taup}[i] \cdot u_i' u_i.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_i\f$ are zero, and \f$u_i[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_i\f$ are zero, and \f$u_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of the matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrix to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_i, and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_i.
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_i, and the elements above the
                diagonal are the last n - i elements of Householder vector u_i.
    @param[in]
    lda         rocblas_int. lda >= m.\n
                specifies the leading dimension of A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension min(m,n).\n
                The diagonal elements of B.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
                The off-diagonal elements of B.
    @param[out]
    tauq        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars associated with matrix Q.
    @param[out]
    taup        pointer to type. Array on the GPU of dimension min(m,n).\n
                The Householder scalars associated with matrix P.
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

    For each instance in the batch, the bidiagonal form is given by:

    \f[
        B_j = Q_j'  A_j  P_j
    \f]

    where \f$B_j\f$ is upper bidiagonal if m >= n and lower bidiagonal if m < n, and \f$Q_j\f$ and
    \f$P_j\f$ are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_n}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_{n-1}}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{m-1}}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ and \f$G_{j_i}\f$ is given by

    \f[
        \begin{array}{cl}
        H_{j_i} = I - \text{tauq}_j[i] \cdot v_{j_i} v_{j_i}', & \: \text{and}\\
        G_{j_i} = I - \text{taup}_j[i] \cdot u_{j_i}' u_{j_i}.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B_j.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i), and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_(j_i).
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_(j_i), and the elements above the
                diagonal are the last n - i elements of Householder vector u_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of B_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of B_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq        pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
                Contains the vectors tauq_j of Householder scalars associated with matrices Q_j.
    @param[in]
    strideQ     rocblas_stride.\n
                Stride from the start of one vector tauq_j to the next one tauq_(j+1).
                There is no restriction for the value
                of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors taup_j of Householder scalars associated with matrices P_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector taup_j to the next one taup_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the bidiagonal form is given by:

    \f[
        B_j = Q_j'  A_j  P_j
    \f]

    where \f$B_j\f$ is upper bidiagonal if m >= n and lower bidiagonal if m < n, and \f$Q_j\f$ and
    \f$P_j\f$ are orthogonal/unitary matrices represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_n}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_{n-1}}, & \: \text{if}\: m >= n, \:\text{or}\\
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{m-1}}\:  \text{and} \: P_j = G_{j_1}G_{j_2}\cdots G_{j_m}, & \: \text{if}\: m < n.
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ and \f$G_{j_i}\f$ is given by

    \f[
        \begin{array}{cl}
        H_{j_i} = I - \text{tauq}_j[i] \cdot v_{j_i} v_{j_i}', & \: \text{and}\\
        G_{j_i} = I - \text{taup}_j[i] \cdot u_{j_i}' u_{j_i}.
        \end{array}
    \f]

    If m >= n, the first i-1 elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$;
    while the first i elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i+1] = 1\f$.
    If m < n, the first i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$;
    while the first i-1 elements of the Householder vector \f$u_{j_i}\f$ are zero, and \f$u_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all the matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all the matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the m-by-n matrices A_j to be factored.
                On exit, the elements on the diagonal and superdiagonal (if m >= n), or
                subdiagonal (if m < n) contain the bidiagonal form B_j.
                If m >= n, the elements below the diagonal are the last m - i elements
                of Householder vector v_(j_i), and the elements above the
                superdiagonal are the last n - i - 1 elements of Householder vector u_(j_i).
                If m < n, the elements below the subdiagonal are the last m - i - 1
                elements of Householder vector v_(j_i), and the elements above the
                diagonal are the last n - i elements of Householder vector u_(j_i).
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of B_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= min(m,n).
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of B_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= min(m,n)-1.
    @param[out]
    tauq        pointer to type. Array on the GPU (the size depends on the value of strideQ).\n
                Contains the vectors tauq_j of Householder scalars associated with matrices Q_j.
    @param[in]
    strideQ     rocblas_stride.\n
                Stride from the start of one vector tauq_j to the next one tauq_(j+1).
                There is no restriction for the value
                of strideQ. Normal use is strideQ >= min(m,n).
    @param[out]
    taup        pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors taup_j of Householder scalars associated with matrices P_j.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector taup_j to the next one taup_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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
    \brief GETRS solves a system of n linear equations on n variables in its factorized form.

    \details
    It solves one of the following systems, depending on the value of trans:

    \f[
        \begin{array}{cl}
        A X = B & \: \text{not transposed,}\\
        A^T X = B & \: \text{transposed, or}\\
        A^H X = B & \: \text{conjugate transposed.}
        \end{array}
    \f]

    Matrix A is defined by its triangular factors as returned by \ref rocsolver_sgetrf "GETRF".

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
                The factors L and U of the factorization A = P*L*U returned by \ref rocsolver_sgetrf "GETRF".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by \ref rocsolver_sgetrf "GETRF".
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
    variables in its factorized forms.

    \details
    For each instance j in the batch, it solves one of the following systems, depending on the value of trans:

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed,}\\
        A_j^T X_j = B_j & \: \text{transposed, or}\\
        A_j^H X_j = B_j & \: \text{conjugate transposed.}
        \end{array}
    \f]

    Matrix \f$A_j\f$ is defined by its triangular factors as returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".

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
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of pivot indices returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
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
    on n variables in its factorized forms.

    \details
    For each instance j in the batch, it solves one of the following systems, depending on the value of trans:

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed,}\\
        A_j^T X_j = B_j & \: \text{transposed, or}\\
        A_j^H X_j = B_j & \: \text{conjugate transposed.}
        \end{array}
    \f]

    Matrix \f$A_j\f$ is defined by its triangular factors as returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".

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
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors ipiv_j of pivot indices returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[in,out]
    B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
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
    \brief GESV solves a general system of n linear equations on n variables.

    \details
    The linear system is of the form

    \f[
        A X = B
    \f]

    where A is a general n-by-n matrix. Matrix A is first factorized in triangular factors L and U
    using \ref rocsolver_sgetrf "GETRF"; then, the solution is computed with \ref rocsolver_sgetrs "GETRS".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of the matrix B.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A.
                On exit, if info = 0, the factors L and U of the LU decomposition of A returned by
                \ref rocsolver_sgetrf "GETRF".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by \ref rocsolver_sgetrf "GETRF".
    @param[in,out]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of B.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular, and the solution could not be computed.
                U[i,i] is the first zero element in the diagonal.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesv(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                float* A,
                                                const rocblas_int lda,
                                                rocblas_int* ipiv,
                                                float* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesv(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                double* A,
                                                const rocblas_int lda,
                                                rocblas_int* ipiv,
                                                double* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesv(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                rocblas_int* ipiv,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesv(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                rocblas_int* ipiv,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief GESV_BATCHED solves a batch of general systems of n linear equations on n
    variables.

    \details
    The linear systems are of the form

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a general n-by-n matrix. Matrix \f$A_j\f$ is first factorized in triangular factors \f$L_j\f$ and \f$U_j\f$
    using \ref rocsolver_sgetrf_batched "GETRF_BATCHED"; then, the solutions are computed with \ref rocsolver_sgetrs_batched "GETRS_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j.
                On exit, if info_j = 0, the factors L_j and U_j of the LU decomposition of A_j returned by
                \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The vectors ipiv_j of pivot indices returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[in,out]
    B           Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for A_j.
                If info[i] = j > 0, U_i is singular, and the solution could not be computed.
                U_j[i,i] is the first zero element in the diagonal.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesv_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_int* ipiv,
                                                        const rocblas_stride strideP,
                                                        float* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesv_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_int* ipiv,
                                                        const rocblas_stride strideP,
                                                        double* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesv_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_int* ipiv,
                                                        const rocblas_stride strideP,
                                                        rocblas_float_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesv_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_int* ipiv,
                                                        const rocblas_stride strideP,
                                                        rocblas_double_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief GESV_STRIDED_BATCHED solves a batch of general systems of n linear equations
    on n variables.

    \details
    The linear systems are of the form

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a general n-by-n matrix. Matrix \f$A_j\f$ is first factorized in triangular factors \f$L_j\f$ and \f$U_j\f$
    using \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED"; then, the solutions are computed with
    \ref rocsolver_sgetrs_strided_batched "GETRS_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j.
                On exit, if info_j = 0, the factors L_j and U_j of the LU decomposition of A_j returned by
                \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The vectors ipiv_j of pivot indices returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[in,out]
    B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for A_j.
                If info[i] = j > 0, U_i is singular, and the solution could not be computed.
                U_j[i,i] is the first zero element in the diagonal.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesv_strided_batched(rocblas_handle handle,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                float* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_int* ipiv,
                                                                const rocblas_stride strideP,
                                                                float* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesv_strided_batched(rocblas_handle handle,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                double* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_int* ipiv,
                                                                const rocblas_stride strideP,
                                                                double* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesv_strided_batched(rocblas_handle handle,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                rocblas_float_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_int* ipiv,
                                                                const rocblas_stride strideP,
                                                                rocblas_float_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesv_strided_batched(rocblas_handle handle,
                                                                const rocblas_int n,
                                                                const rocblas_int nrhs,
                                                                rocblas_double_complex* A,
                                                                const rocblas_int lda,
                                                                const rocblas_stride strideA,
                                                                rocblas_int* ipiv,
                                                                const rocblas_stride strideP,
                                                                rocblas_double_complex* B,
                                                                const rocblas_int ldb,
                                                                const rocblas_stride strideB,
                                                                rocblas_int* info,
                                                                const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI inverts a general n-by-n matrix A using the LU factorization
    computed by \ref rocsolver_sgetrf "GETRF".

    \details
    The inverse is computed by solving the linear system

    \f[
        A^{-1}L = U^{-1}
    \f]

    where L is the lower triangular factor of A with unit diagonal elements, and U is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the factors L and U of the factorization A = P*L*U returned by \ref rocsolver_sgetrf "GETRF".
                On exit, the inverse of A if info = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by \ref rocsolver_sgetrf "GETRF".
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
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
    the LU factorization computed by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed by solving the linear system

    \f[
        A_j^{-1} L_j = U_j^{-1}
    \f]

    where \f$L_j\f$ is the lower triangular factor of \f$A_j\f$ with unit diagonal elements, and \f$U_j\f$ is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the factors L_j and U_j of the factorization A = P_j*L_j*U_j returned by
                \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
                On exit, the inverses of A_j if info[j] = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The pivot indices returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(i+j).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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
    using the LU factorization computed by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed by solving the linear system

    \f[
        A_j^{-1} L_j = U_j^{-1}
    \f]

    where \f$L_j\f$ is the lower triangular factor of \f$A_j\f$ with unit diagonal elements, and \f$U_j\f$ is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by
                \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
                On exit, the inverses of A_j if info[j] = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The pivot indices returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
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
    \brief GETRI_NPVT inverts a general n-by-n matrix A using the LU factorization
    computed by \ref rocsolver_sgetrf_npvt "GETRF_NPVT".

    \details
    The inverse is computed by solving the linear system

    \f[
        A^{-1}L = U^{-1}
    \f]

    where L is the lower triangular factor of A with unit diagonal elements, and U is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the factors L and U of the factorization A = L*U returned by \ref rocsolver_sgetrf_npvt "GETRF_NPVT".
                On exit, the inverse of A if info = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_npvt(rocblas_handle handle,
                                                      const rocblas_int n,
                                                      float* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_npvt(rocblas_handle handle,
                                                      const rocblas_int n,
                                                      double* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_npvt(rocblas_handle handle,
                                                      const rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_npvt(rocblas_handle handle,
                                                      const rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      const rocblas_int lda,
                                                      rocblas_int* info);
//! @}

/*! @{
    \brief GETRI_NPVT_BATCHED inverts a batch of general n-by-n matrices using
    the LU factorization computed by \ref rocsolver_sgetrf_npvt_batched "GETRF_NPVT_BATCHED".

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed by solving the linear system

    \f[
        A_j^{-1} L_j = U_j^{-1}
    \f]

    where \f$L_j\f$ is the lower triangular factor of \f$A_j\f$ with unit diagonal elements, and \f$U_j\f$ is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the factors L_j and U_j of the factorization A = L_j*U_j returned by
                \ref rocsolver_sgetrf_npvt_batched "GETRF_NPVT_BATCHED".
                On exit, the inverses of A_j if info[j] = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int n,
                                                              float* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int n,
                                                              double* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int n,
                                                              rocblas_float_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_npvt_batched(rocblas_handle handle,
                                                              const rocblas_int n,
                                                              rocblas_double_complex* const A[],
                                                              const rocblas_int lda,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI_NPVT_STRIDED_BATCHED inverts a batch of general n-by-n matrices
    using the LU factorization computed by \ref rocsolver_sgetrf_npvt_strided_batched "GETRF_NPVT_STRIDED_BATCHED".

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed by solving the linear system

    \f[
        A_j^{-1} L_j = U_j^{-1}
    \f]

    where \f$L_j\f$ is the lower triangular factor of \f$A_j\f$ with unit diagonal elements, and \f$U_j\f$ is the
    upper triangular factor.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the factors L_j and U_j of the factorization A_j = L_j*U_j returned by
                \ref rocsolver_sgetrf_npvt_strided_batched "GETRF_NPVT_STRIDED_BATCHED".
                On exit, the inverses of A_j if info[j] = 0; otherwise undefined.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int n,
                                                                      float* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int n,
                                                                      double* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int n,
                                                                      rocblas_float_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_npvt_strided_batched(rocblas_handle handle,
                                                                      const rocblas_int n,
                                                                      rocblas_double_complex* A,
                                                                      const rocblas_int lda,
                                                                      const rocblas_stride strideA,
                                                                      rocblas_int* info,
                                                                      const rocblas_int batch_count);
//! @}

/*! @{
    \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
    matrix A, and a corresponding matrix B, using the QR factorization computed by \ref rocsolver_sgeqrf "GEQRF" (or the LQ
    factorization computed by \ref rocsolver_sgelqf "GELQF").

    \details
    Depending on the value of trans, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A X = B & \: \text{not transposed, or}\\
        A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
        \end{array}
    \f]

    If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X is found by minimizing

    \f[
        || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
    \f]

    If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of matrix A.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of matrix A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of columns of matrices B and X;
                i.e., the columns on the right hand side.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A.
                On exit, the QR (or LQ) factorization of A as returned by \ref rocsolver_sgeqrf "GEQRF" (or \ref rocsolver_sgelqf "GELQF").
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrix A.
    @param[inout]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the matrix B.
                On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.
    @param[in]
    ldb         rocblas_int. ldb >= max(m,n).\n
                Specifies the leading dimension of matrix B.
    @param[out]
    info        pointer to rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, the solution could not be computed because input matrix A is
                rank deficient; the i-th diagonal element of its triangular factor is zero.
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
    \brief GELS_BATCHED solves a batch of overdetermined (or underdetermined) linear systems
    defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$, using the
    QR factorizations computed by \ref rocsolver_sgeqrf_batched "GEQRF_BATCHED" (or the LQ factorizations computed by \ref rocsolver_sgelqf_batched "GELQF_BATCHED").

    \details
    For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed, or}\\
        A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
        \end{array}
    \f]

    If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X_j is found by minimizing

    \f[
        || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
    \f]

    If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of columns of all matrices B_j and X_j in the batch;
                i.e., the columns on the right hand side.
    @param[inout]
    A           array of pointer to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j.
                On exit, the QR (or LQ) factorizations of A_j as returned by \ref rocsolver_sgeqrf_batched "GEQRF_BATCHED"
                (or \ref rocsolver_sgelqf_batched "GELQF_BATCHED").
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[inout]
    B           array of pointer to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
                On entry, the matrices B_j.
                On exit, when info[j] = 0, B_j is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.
    @param[in]
    ldb         rocblas_int. ldb >= max(m,n).\n
                Specifies the leading dimension of matrices B_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for solution of A_j.
                If info[j] = i > 0, the solution of A_j could not be computed because input
                matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
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
    \brief GELS_STRIDED_BATCHED solves a batch of overdetermined (or underdetermined) linear
    systems defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$,
    using the QR factorizations computed by \ref rocsolver_sgeqrf_strided_batched "GEQRF_STRIDED_BATCHED"
    (or the LQ factorizations computed by \ref rocsolver_sgelqf_strided_batched "GELQF_STRIDED_BATCHED").

    \details
    For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed, or}\\
        A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
        \end{array}
    \f]

    If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X_j is found by minimizing

    \f[
        || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
    \f]

    If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations.
    @param[in]
    m           rocblas_int. m >= 0.\n
                The number of rows of all matrices A_j in the batch.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of columns of all matrices A_j in the batch.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of columns of all matrices B_j and X_j in the batch;
                i.e., the columns on the right hand side.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j.
                On exit, the QR (or LQ) factorizations of A_j as returned by \ref rocsolver_sgeqrf_strided_batched "GEQRF_STRIDED_BATCHED"
                (or \ref rocsolver_sgelqf_strided_batched "GELQF_STRIDED_BATCHED").
    @param[in]
    lda         rocblas_int. lda >= m.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[inout]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                On entry, the matrices B_j.
                On exit, when info[j] = 0, each B_j is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.
    @param[in]
    ldb         rocblas_int. ldb >= max(m,n).\n
                Specifies the leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for solution of A_j.
                If info[j] = i > 0, the solution of A_j could not be computed because input
                matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
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
    \brief POTF2 computes the Cholesky factorization of a real symmetric (complex
    Hermitian) positive definite matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form:

    \f[
        \begin{array}{cl}
        A = U'U & \: \text{if uplo is upper, or}\\
        A = LL' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    U is an upper triangular matrix and L is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A to be factored. On exit, the lower or upper triangular factor.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
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
    batch of real symmetric (complex Hermitian) positive definite matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form:

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    \f$U_j\f$ is an upper triangular matrix and \f$L_j\f$ is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful factorization of matrix A_j.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th factorization stopped at this point.
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
    batch of real symmetric (complex Hermitian) positive definite matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form:

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    \f$U_j\f$ is an upper triangular matrix and \f$L_j\f$ is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA    rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful factorization of matrix A_j.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th factorization stopped at this point.
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
    \brief POTRF computes the Cholesky factorization of a real symmetric (complex
    Hermitian) positive definite matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form:

    \f[
        \begin{array}{cl}
        A = U'U & \: \text{if uplo is upper, or}\\
        A = LL' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    U is an upper triangular matrix and L is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A to be factored. On exit, the lower or upper triangular factor.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
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
    batch of real symmetric (complex Hermitian) positive definite matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form:

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    \f$U_j\f$ is an upper triangular matrix and \f$L_j\f$ is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful factorization of matrix A_j.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th factorization stopped at this point.
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
    batch of real symmetric (complex Hermitian) positive definite matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix \f$A_j\f$ in the batch has the form:

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    \f$U_j\f$ is an upper triangular matrix and \f$L_j\f$ is lower triangular.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored. On exit, the upper or lower triangular factors.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful factorization of matrix A_j.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th factorization stopped at this point.
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
    \brief POTRS solves a symmetric/hermitian system of n linear equations on n variables in its factorized form.

    \details
    It solves the system

    \f[
        A X = B
    \f]

    where A is a real symmetric (complex hermitian) positive definite matrix defined by its triangular factor

    \f[
        \begin{array}{cl}
        A = U'U & \: \text{if uplo is upper, or}\\
        A = LL' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    as returned by \ref rocsolver_spotrf "POTRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of the matrix B.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                The factor L or U of the Cholesky factorization of A returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[in,out]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of B.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrs(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrs(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrs(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* B,
                                                 const rocblas_int ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrs(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* B,
                                                 const rocblas_int ldb);
//! @}

/*! @{
    \brief POTRS_BATCHED solves a batch of symmetric/hermitian systems of n linear equations on n
    variables in its factorized forms.

    \details
    For each instance j in the batch, it solves the system

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a real symmetric (complex hermitian) positive definite matrix defined by its
    triangular factor

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    as returned by \ref rocsolver_spotrf "POTRF_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                The factor L_j or U_j of the Cholesky factorization of A_j returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
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

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrs_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         float* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrs_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         double* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrs_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrs_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         const rocblas_int nrhs,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* const B[],
                                                         const rocblas_int ldb,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief POTRS_STRIDED_BATCHED solves a batch of symmetric/hermitian systems of n linear equations
    on n variables in its factorized forms.

    \details
    For each instance j in the batch, it solves the system

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a real symmetric (complex hermitian) positive definite matrix defined by its
    triangular factor

    \f[
        \begin{array}{cl}
        A_j = U_j'U_j & \: \text{if uplo is upper, or}\\
        A_j = L_jL_j' & \: \text{if uplo is lower.}
        \end{array}
    \f]

    as returned by \ref rocsolver_spotrf "POTRF_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                The factor L_j or U_j of the Cholesky factorization of A_j returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[in,out]
    B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 float* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 double* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_float_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrs_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 const rocblas_int nrhs,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_double_complex* B,
                                                                 const rocblas_int ldb,
                                                                 const rocblas_stride strideB,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief POSV solves a symmetric/hermitian system of n linear equations on n variables.

    \details
    It solves the system

    \f[
        A X = B
    \f]

    where A is a real symmetric (complex hermitian) positive definite matrix. Matrix A is first
    factorized as \f$A=LL'\f$ or \f$A=U'U\f$, depending on the value of uplo, using \ref rocsolver_spotrf "POTRF";
    then, the solution is computed with \ref rocsolver_spotrs "POTRS".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of the matrix B.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric/hermitian matrix A.
                On exit, if info = 0, the factor L or U of the Cholesky factorization of A returned by
                \ref rocsolver_spotrf "POTRF".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[in,out]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of B.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, the leading minor of order i of A is not positive definite.
                The solution could not be computed.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sposv(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                float* A,
                                                const rocblas_int lda,
                                                float* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dposv(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                double* A,
                                                const rocblas_int lda,
                                                double* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cposv(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zposv(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                const rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                rocblas_int* info);
//! @}

/*! @{
    \brief POSV_BATCHED solves a batch of symmetric/hermitian systems of n linear equations on n
    variables.

    \details
    For each instance j in the batch, it solves the system

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a real symmetric (complex hermitian) positive definite matrix. Matrix \f$A_j\f$ is first
    factorized as \f$A_j=L_jL_j'\f$ or \f$A_j=U_j'U_j\f$, depending on the value of uplo, using \ref rocsolver_spotrf_batched "POTRF_BATCHED";
    then, the solution is computed with \ref rocsolver_spotrs_batched "POTRS_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the symmetric/hermitian matrices A_j.
                On exit, if info[j] = 0, the factor L_j or U_j of the Cholesky factorization of A_j returned by
                \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in,out]
    B           Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th solution could not be computed.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sposv_batched(rocblas_handle handle,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dposv_batched(rocblas_handle handle,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cposv_batched(rocblas_handle handle,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_float_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zposv_batched(rocblas_handle handle,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        const rocblas_int nrhs,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_double_complex* const B[],
                                                        const rocblas_int ldb,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count);
//! @}

/*! @{
    \brief POSV_STRIDED_BATCHED solves a batch of symmetric/hermitian systems of n linear equations
    on n variables.

    \details
    For each instance j in the batch, it solves the system

    \f[
        A_j X_j = B_j
    \f]

    where \f$A_j\f$ is a real symmetric (complex hermitian) positive definite matrix. Matrix \f$A_j\f$ is first
    factorized as \f$A_j=L_jL_j'\f$ or \f$A_j=U_j'U_j\f$, depending on the value of uplo, using \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED";
    then, the solution is computed with \ref rocsolver_spotrs_strided_batched "POTRS_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns and rows of all A_j matrices.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of all the matrices B_j.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the symmetric/hermitian matrices A_j.
                On exit, if info[j] = 0, the factor L_j or U_j of the Cholesky factorization of A_j returned by
                \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[in,out]
    B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
                On entry, the right hand side matrices B_j.
                On exit, the solution matrix X_j of each system in the batch.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of matrices B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit.
                If info[j] = i > 0, the leading minor of order i of A_j is not positive definite.
                The j-th solution could not be computed.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of instances (systems) in the batch.
   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sposv_strided_batched(rocblas_handle handle,
                                                                const rocblas_fill uplo,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_dposv_strided_batched(rocblas_handle handle,
                                                                const rocblas_fill uplo,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_cposv_strided_batched(rocblas_handle handle,
                                                                const rocblas_fill uplo,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_zposv_strided_batched(rocblas_handle handle,
                                                                const rocblas_fill uplo,
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
//! @}

/*! @{
    \brief POTRI inverts a symmetric/hermitian positive definite matrix A.

    \details
    The inverse of matrix \f$A\f$ is computed as

    \f[
        \begin{array}{cl}
        A^{-1} = U^{-1} {U^{-1}}' & \: \text{if uplo is upper, or}\\
        A^{-1} = {L^{-1}}' L^{-1} & \: \text{if uplo is lower.}
        \end{array}
    \f]

    where \f$U\f$ or \f$L\f$ is the triangular factor of the Cholesky factorization of \f$A\f$ returned by
    \ref rocsolver_spotrf "POTRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the factor L or U of the Cholesky factorization of A returned by
                \ref rocsolver_spotrf "POTRF".
                On exit, the inverse of A if info = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit for inversion of A.
                If info = i > 0, A is singular. L[i,i] or U[i,i] is zero.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief POTRI_BATCHED inverts a batch of symmetric/hermitian positive definite matrices \f$A_j\f$.

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed as

    \f[
        \begin{array}{cl}
        A_j^{-1} = U_j^{-1} {U_j^{-1}}' & \: \text{if uplo is upper, or}\\
        A_j^{-1} = {L_j^{-1}}' L_j^{-1} & \: \text{if uplo is lower.}
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is the triangular factor of the Cholesky factorization of \f$A_j\f$ returned by
    \ref rocsolver_spotrf_batched "POTRF_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the factor L_j or U_j of the Cholesky factorization of A_j returned by
                \ref rocsolver_spotrf_batched "POTRF_BATCHED".
                On exit, the inverses of A_j if info[j] = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, A_j is singular. L_j[i,i] or U_j[i,i] is zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief POTRI_STRIDED_BATCHED inverts a batch of symmetric/hermitian positive definite matrices \f$A_j\f$.

    \details
    The inverse of matrix \f$A_j\f$ in the batch is computed as

    \f[
        \begin{array}{cl}
        A_j^{-1} = U_j^{-1} {U_j^{-1}}' & \: \text{if uplo is upper, or}\\
        A_j^{-1} = {L_j^{-1}}' L_j^{-1} & \: \text{if uplo is lower.}
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is the triangular factor of the Cholesky factorization of \f$A_j\f$ returned by
    \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the factorization is upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of matrix A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the factor L_j or U_j of the Cholesky factorization of A_j returned by
                \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
                On exit, the inverses of A_j if info[j] = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, A_j is singular. L_j[i,i] or U_j[i,i] is zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief GESVD computes the singular values and optionally the singular
    vectors of a general m-by-n matrix A (Singular Value Decomposition).

    \details
    The SVD of matrix A is given by:

    \f[
        A = U  S  V'
    \f]

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
    controls whether the fast algorithm is executed or not. For more details, see
    the "Tuning rocSOLVER performance" and "Memory model" sections of the documentation.

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
                On entry, the matrix A.
                On exit, if left_svect (or right_svect) is equal to overwrite,
                the first columns (or rows) contain the left (or right) singular vectors;
                otherwise, the contents of A are destroyed.
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
                set to singular; or ldv >= 1 otherwise.\n
                The leading dimension of V.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension min(m,n)-1.\n
                This array is used to work internally with the bidiagonal matrix
                B associated with A (using \ref rocsolver_sbdsqr "BDSQR"). On exit, if info > 0, it contains the
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
                If info = i > 0, \ref rocsolver_sbdsqr "BDSQR" did not converge. i elements of E did not converge to zero.
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
    \brief GESVD_BATCHED computes the singular values and optionally the
    singular vectors of a batch of general m-by-n matrix A (Singular Value
    Decomposition).

    \details
    The SVD of matrix A_j in the batch is given by:

    \f[
        A_j = U_j  S_j  V_j'
    \f]

    where the m-by-n matrix \f$S_j\f$ is zero except, possibly, for its min(m,n)
    diagonal elements, which are the singular values of \f$A_j\f$. \f$U_j\f$ and \f$V_j\f$ are
    orthogonal (unitary) matrices. The first min(m,n) columns of \f$U_j\f$ and \f$V_j\f$ are
    the left and right singular vectors of \f$A_j\f$, respectively.

    The computation of the singular vectors is optional and it is controlled by
    the function arguments left_svect and right_svect as described below. When
    computed, this function returns the transpose (or transpose conjugate) of the
    right singular vectors, i.e. the rows of \f$V_j'\f$.

    left_svect and right_svect are #rocblas_svect enums that can take the
    following values:

    - rocblas_svect_all: the entire matrix \f$U_j\f$ (or \f$V_j'\f$) is computed,
    - rocblas_svect_singular: only the singular vectors (first min(m,n)
      columns of \f$U_j\f$ or rows of \f$V_j'\f$) are computed,
    - rocblas_svect_overwrite: the
      first columns (or rows) of \f$A_j\f$ are overwritten with the singular vectors, or
    - rocblas_svect_none: no columns (or rows) of \f$U_j\f$ (or \f$V_j'\f$) are computed,
      i.e. no singular vectors.

    left_svect and right_svect cannot both be set to overwrite. When neither is
    set to overwrite, the contents of \f$A_j\f$ are destroyed by the time the function
    returns.

    \note
    When m >> n (or n >> m) the algorithm could be sped up by compressing
    the matrix \f$A_j\f$ via a QR (or LQ) factorization, and working with the
    triangular factor afterwards (thin-SVD). If the singular vectors are also
    requested, its computation could be sped up as well via executing some
    intermediate operations out-of-place, and relying more on matrix
    multiplications (GEMMs); this will require, however, a larger memory
    workspace. The parameter fast_alg controls whether the fast algorithm is
    executed or not. For more details, see the "Tuning rocSOLVER performance"
    and "Memory model" sections of the documentation.

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
                On entry, the matrices A_j.
                On exit, if left_svect (or right_svect) is equal to overwrite,
                the first columns (or rows) of A_j contain the left (or right)
                corresponding singular vectors; otherwise, the contents of A_j are destroyed.
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
                This array is used to work internally with the bidiagonal matrix B_j associated with A_j (using \ref rocsolver_sbdsqr "BDSQR").
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of B_j (or properly speaking,
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
                If info[j] = 0, successful exit.
                If info[j] = i > 0, \ref rocsolver_sbdsqr "BDSQR" did not converge. i elements of E_j did not converge to zero.
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
    \brief GESVD_STRIDED_BATCHED computes the singular values and optionally the
    singular vectors of a batch of general m-by-n matrix A (Singular Value
    Decomposition).

    \details
    The SVD of matrix A_j in the batch is given by:

    \f[
        A_j = U_j  S_j  V_j'
    \f]

    where the m-by-n matrix \f$S_j\f$ is zero except, possibly, for its min(m,n)
    diagonal elements, which are the singular values of \f$A_j\f$. \f$U_j\f$ and \f$V_j\f$ are
    orthogonal (unitary) matrices. The first min(m,n) columns of \f$U_j\f$ and \f$V_j\f$ are
    the left and right singular vectors of \f$A_j\f$, respectively.

    The computation of the singular vectors is optional and it is controlled by
    the function arguments left_svect and right_svect as described below. When
    computed, this function returns the transpose (or transpose conjugate) of the
    right singular vectors, i.e. the rows of \f$V_j'\f$.

    left_svect and right_svect are #rocblas_svect enums that can take the
    following values:

    - rocblas_svect_all: the entire matrix \f$U_j\f$ (or \f$V_j'\f$) is computed,
    - rocblas_svect_singular: only the singular vectors (first min(m,n)
      columns of \f$U_j\f$ or rows of \f$V_j'\f$) are computed,
    - rocblas_svect_overwrite: the
      first columns (or rows) of \f$A_j\f$ are overwritten with the singular vectors, or
    - rocblas_svect_none: no columns (or rows) of \f$U_j\f$ (or \f$V_j'\f$) are computed,
      i.e. no singular vectors.

    left_svect and right_svect cannot both be set to overwrite. When neither is
    set to overwrite, the contents of \f$A_j\f$ are destroyed by the time the function
    returns.

    \note
    When m >> n (or n >> m) the algorithm could be sped up by compressing
    the matrix \f$A_j\f$ via a QR (or LQ) factorization, and working with the
    triangular factor afterwards (thin-SVD). If the singular vectors are also
    requested, its computation could be sped up as well via executing some
    intermediate operations out-of-place, and relying more on matrix
    multiplications (GEMMs); this will require, however, a larger memory
    workspace. The parameter fast_alg controls whether the fast algorithm is
    executed or not. For more details, see the "Tuning rocSOLVER performance"
    and "Memory model" sections of the documentation.

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
                On entry, the matrices A_j. On exit, if left_svect (or right_svect) is equal to
                overwrite, the first columns (or rows) of A_j contain the left (or right)
                corresponding singular vectors; otherwise, the contents of A_j are destroyed.
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
                This array is used to work internally with the bidiagonal matrix B_j associated with A_j (using \ref rocsolver_sbdsqr "BDSQR").
                On exit, if info > 0, E_j contains the unconverged off-diagonal elements of B_j (or properly speaking,
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
                If info[j] = 0, successful exit.
                If info[j] = i > 0, BDSQR did not converge. i elements of E_j did not converge to zero.
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

    \f[
        T = Q'  A  Q
    \f]

    where T is symmetric tridiagonal and Q is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_{n-1} & \: \text{if uplo indicates lower, or}\\
        Q = H_{n-1}H_{n-2}\cdots H_1 & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{tau}[i] \cdot v_i  v_i'
    \f]

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_i stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_i stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                The diagonal elements of T.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n-1.\n
                The off-diagonal elements of T.
    @param[out]
    tau         pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars.
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

    \f[
        T = Q'  A  Q
    \f]

    where T is hermitian tridiagonal and Q is an unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_{n-1} & \: \text{if uplo indicates lower, or}\\
        Q = H_{n-1}H_{n-2}\cdots H_1 & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{tau}[i] \cdot v_i  v_i'
    \f]

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T; the elements above the superdiagonal contain
                the first i-1 elements of the Householders vector v_i stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_i stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The diagonal elements of T.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                The off-diagonal elements of T.
    @param[out]
    tau         pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars.
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is symmetric tridiagonal and \f$Q_j\f$ is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is Hermitian tridiagonal and \f$Q_j\f$ is a unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is symmetric tridiagonal and \f$Q_j\f$ is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is Hermitian tridiagonal and \f$Q_j\f$ is a unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        T = Q'  A  Q
    \f]

    where T is symmetric tridiagonal and Q is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_{n-1} & \: \text{if uplo indicates lower, or}\\
        Q = H_{n-1}H_{n-2}\cdots H_1 & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{tau}[i] \cdot v_i  v_i'
    \f]

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_i stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_i stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                The diagonal elements of T.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n-1.\n
                The off-diagonal elements of T.
    @param[out]
    tau         pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars.
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

    \f[
        T = Q'  A  Q
    \f]

    where T is hermitian tridiagonal and Q is an unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H_1H_2\cdots H_{n-1} & \: \text{if uplo indicates lower, or}\\
        Q = H_{n-1}H_{n-2}\cdots H_1 & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_i\f$ is given by

    \f[
        H_i = I - \text{tau}[i] \cdot v_i  v_i'
    \f]

    where tau[i] is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_i stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_i stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The diagonal elements of T.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n-1.\n
                The off-diagonal elements of T.
    @param[out]
    tau         pointer to type. Array on the GPU of dimension n-1.\n
                The Householder scalars.
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is symmetric tridiagonal and \f$Q_j\f$ is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is Hermitian tridiagonal and \f$Q_j\f$ is a unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E          pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is symmetric tridiagonal and \f$Q_j\f$ is an orthogonal matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the symmetric matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    The tridiagonal form of \f$A_j\f$ is given by:

    \f[
        T_j = Q_j'  A_j  Q_j
    \f]

    where \f$T_j\f$ is Hermitian tridiagonal and \f$Q_j\f$ is a unitary matrix represented as the product
    of Householder matrices

    \f[
        \begin{array}{cl}
        Q_j = H_{j_1}H_{j_2}\cdots H_{j_{n-1}} & \: \text{if uplo indicates lower, or}\\
        Q_j = H_{j_{n-1}}H_{j_{n-2}}\cdots H_{j_1} & \: \text{if uplo indicates upper.}
        \end{array}
    \f]

    Each Householder matrix \f$H_{j_i}\f$ is given by

    \f[
        H_{j_i} = I - \text{tau}_j[i] \cdot v_{j_i}  v_{j_i}'
    \f]

    where \f$\text{tau}_j[i]\f$ is the corresponding Householder scalar. When uplo indicates lower, the first i
    elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i+1] = 1\f$. If uplo indicates upper,
    the last n-i elements of the Householder vector \f$v_{j_i}\f$ are zero, and \f$v_{j_i}[i] = 1\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the hermitian matrix A_j is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j to be factored.
                On exit, if upper, then the elements on the diagonal and superdiagonal
                contain the tridiagonal form T_j; the elements above the superdiagonal contain
                the first i-1 elements of the Householder vectors v_(j_i) stored as columns.
                If lower, then the elements on the diagonal and subdiagonal
                contain the tridiagonal form T_j; the elements below the subdiagonal contain
                the last n-i-1 elements of the Householder vectors v_(j_i) stored as columns.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The diagonal elements of T_j.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                The off-diagonal elements of T_j.
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n-1.
    @param[out]
    tau         pointer to type. Array on the GPU (the size depends on the value of strideP).\n
                Contains the vectors tau_j of corresponding Householder scalars.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector tau_j to the next one tau_(j+1).
                There is no restriction for the value
                of strideP. Normal use is strideP >= n-1.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U^{-T} A U^{-1}, & \: \text{or}\\
        L^{-1} A L^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix B has been factorized as either \f$U^T U\f$ or
    \f$L L^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U A U^T, & \: \text{or}\\
        L^T A L,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored, and
                whether the factorization applied to B was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A and
                B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the transformed matrix associated with
                the equivalent standard eigenvalue problem.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                The triangular factor of the matrix B, as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U^{-H} A U^{-1}, & \: \text{or}\\
        L^{-1} A L^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix B has been factorized as either \f$U^H U\f$ or
    \f$L L^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U A U^H, & \: \text{or}\\
        L^H A L,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored, and
                whether the factorization applied to B was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A and
                B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the transformed matrix associated with
                the equivalent standard eigenvalue problem.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                The triangular factor of the matrix B, as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-T} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^T U_j\f$ or
    \f$L_j L_j^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^T, & \: \text{or}\\
        L_j^T A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-H} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^H U_j\f$ or
    \f$L_j L_j^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^H, & \: \text{or}\\
        L_j^H A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-T} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^T U_j\f$ or
    \f$L_j L_j^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^T, & \: \text{or}\\
        L_j^T A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-H} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^H U_j\f$ or
    \f$L_j L_j^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^H, & \: \text{or}\\
        L_j^H A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U^{-T} A U^{-1}, & \: \text{or}\\
        L^{-1} A L^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix B has been factorized as either \f$U^T U\f$ or
    \f$L L^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U A U^T, & \: \text{or}\\
        L^T A L,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored, and
                whether the factorization applied to B was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A and
                B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the transformed matrix associated with
                the equivalent standard eigenvalue problem.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                The triangular factor of the matrix B, as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U^{-H} A U^{-1}, & \: \text{or}\\
        L^{-1} A L^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix B has been factorized as either \f$U^H U\f$ or
    \f$L L^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U A U^H, & \: \text{or}\\
        L^H A L,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored, and
                whether the factorization applied to B was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A and
                B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the transformed matrix associated with
                the equivalent standard eigenvalue problem.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                The triangular factor of the matrix B, as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-T} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^T U_j\f$ or
    \f$L_j L_j^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^T, & \: \text{or}\\
        L_j^T A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-H} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^H U_j\f$ or
    \f$L_j L_j^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^H, & \: \text{or}\\
        L_j^H A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-T} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-T},
        \end{array}
    \f]

    where the symmetric-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^T U_j\f$ or
    \f$L_j L_j^T\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^T, & \: \text{or}\\
        L_j^T A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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

    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype.

    If the problem is of the 1st form, then \f$A_j\f$ is overwritten with

    \f[
        \begin{array}{cl}
        U_j^{-H} A_j U_j^{-1}, & \: \text{or}\\
        L_j^{-1} A_j L_j^{-H},
        \end{array}
    \f]

    where the hermitian-definite matrix \f$B_j\f$ has been factorized as either \f$U_j^H U_j\f$ or
    \f$L_j L_j^H\f$ as returned by \ref rocsolver_spotrf "POTRF", depending on the value of uplo.

    If the problem is of the 2nd or 3rd form, then A is overwritten with

    \f[
        \begin{array}{cl}
        U_j A_j U_j^H, & \: \text{or}\\
        L_j^H A_j L_j,
        \end{array}
    \f]

    also depending on the value of uplo.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored, and
                whether the factorization applied to B_j was upper or lower triangular.
                If uplo indicates lower (or upper), then the upper (or lower) parts of A_j and
                B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the transformed matrices associated with
                the equivalent standard eigenvalue problems.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                The triangular factors of the matrices B_j, as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use case is strideB >= ldb*n.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise the contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with A.
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise the contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with A.
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j. If info[j] = i > 0, the algorithm did not converge.
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j. If info[j] = i > 0, the algorithm did not converge.
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j. If info[j] = i > 0, the algorithm did not converge.
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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j. If info[j] = i > 0, the algorithm did not converge.
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
    \brief SYEVD computes the eigenvalues and optionally the eigenvectors of a real symmetric
    matrix A.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise the contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with A.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues of A (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E did not converge to zero.
                If info = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyevd(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyevd(rocblas_handle handle,
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
    \brief HEEVD computes the eigenvalues and optionally the eigenvectors of a Hermitian matrix A.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
                the algorithm converged; otherwise the contents of A are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrix A.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                The eigenvalues of A in increasing order.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with A.
                On exit, if info > 0, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues of A (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E did not converge to zero.
                If info = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheevd(rocblas_handle handle,
                                                 const rocblas_evect evect,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zheevd(rocblas_handle handle,
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
    \brief SYEVD_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    real symmetric matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j.
                If info[j] = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E_j did not converge to zero.
                If info[j] = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyevd_batched(rocblas_handle handle,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyevd_batched(rocblas_handle handle,
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
    \brief HEEVD_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    Hermitian matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j.
                If info[j] = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E_j did not converge to zero.
                If info[j] = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheevd_batched(rocblas_handle handle,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_zheevd_batched(rocblas_handle handle,
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
    \brief SYEVD_STRIDED_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    real symmetric matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j.
                If info[j] = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E_j did not converge to zero.
                If info[j] = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyevd_strided_batched(rocblas_handle handle,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyevd_strided_batched(rocblas_handle handle,
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
    \brief HEEVD_STRIDED_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of
    Hermitian matrices A_j.

    \details
    The eigenvalues are returned in ascending order. The eigenvectors are computed using a
    divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
    are orthonormal.

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
    n           rocblas_int. n >= 0.\n
                Number of rows and columns of matrices A_j.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the matrices A_j. On exit, the eigenvectors of A_j if they were computed and
                the algorithm converged; otherwise the contents of A_j are destroyed.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                The eigenvalues of A_j in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use case is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with A_j.
                On exit, if info[j] > 0, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues of A_j (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use case is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for matrix A_j.
                If info[j] = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
                i elements of E_j did not converge to zero.
                If info[j] = i > 0 and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    **************************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_cheevd_strided_batched(rocblas_handle handle,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_zheevd_strided_batched(rocblas_handle handle,
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix Z of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z^T B Z=I & \: \text{if 1st or 2nd form, or}\\
        Z^T B^{-1} Z=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A and B are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A and B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrix A. On exit, if evect is original,
                the normalized matrix Z of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrix A (including the diagonal) is destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                On entry, the symmetric positive definite matrix B. On exit, the
                triangular factor of B as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                On exit, the eigenvalues in increasing order.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with
                the reduced eigenvalue problem.
                On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info = n + i, the leading minor of order i of B is not
                positive definite.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
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
                                                const rocblas_evect evect,
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

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix Z of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z^H B Z=I & \: \text{if 1st or 2nd form, or}\\
        Z^H B^{-1} Z=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A and B are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A and B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the hermitian matrix A. On exit, if evect is original,
                the normalized matrix Z of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrix A (including the diagonal) is destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                On entry, the hermitian positive definite matrix B. On exit, the
                triangular factor of B as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On exit, the eigenvalues in increasing order.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with
                the reduced eigenvalue problem.
                On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info = n + i, the leading minor of order i of B is not
                positive definite.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
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
                                                const rocblas_evect evect,
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
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^T B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^T B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                On entry, the symmetric positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, E_j contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch instance j.
                If info[j] = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect evect,
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
                                                        const rocblas_evect evect,
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
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^H B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^H B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the hermitian matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                On entry, the hermitian positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv_batched(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect evect,
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
                                                        const rocblas_evect evect,
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
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^T B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^T B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the symmetric matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                On entry, the symmetric positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect evect,
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
                                                                const rocblas_evect evect,
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
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^H B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^H B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the hermitian matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                On entry, the hermitian positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n, i off-diagonal elements of an intermediate
                tridiagonal form did not converge to zero.
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegv_strided_batched(rocblas_handle handle,
                                                                const rocblas_eform itype,
                                                                const rocblas_evect evect,
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
                                                                const rocblas_evect evect,
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

/*! @{
    \brief SYGVD computes the eigenvalues and (optionally) eigenvectors of
    a real generalized symmetric-definite eigenproblem.

    \details
    The problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix Z of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z^T B Z=I & \: \text{if 1st or 2nd form, or}\\
        Z^T B^{-1} Z=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A and B are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A and B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrix A. On exit, if evect is original,
                the normalized matrix Z of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrix A (including the diagonal) is destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                On entry, the symmetric positive definite matrix B. On exit, the
                triangular factor of B as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B.
    @param[out]
    D           pointer to type. Array on the GPU of dimension n.\n
                On exit, the eigenvalues in increasing order.
    @param[out]
    E           pointer to type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with
                the reduced eigenvalue problem.
                On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info = n + i, the leading minor of order i of B is not
                positive definite.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygvd(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_evect evect,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* B,
                                                 const rocblas_int ldb,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygvd(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_evect evect,
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
    \brief HEGVD computes the eigenvalues and (optionally) eigenvectors of
    a complex generalized hermitian-definite eigenproblem.

    \details
    The problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A X = \lambda B X & \: \text{1st form,}\\
        A B X = \lambda X & \: \text{2nd form, or}\\
        B A X = \lambda X & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix Z of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z^H B Z=I & \: \text{if 1st or 2nd form, or}\\
        Z^H B^{-1} Z=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblem.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A and B are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A and B are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the hermitian matrix A. On exit, if evect is original,
                the normalized matrix Z of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrix A (including the diagonal) is destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    B           pointer to type. Array on the GPU of dimension ldb*n.\n
                On entry, the hermitian positive definite matrix B. On exit, the
                triangular factor of B as returned by \ref rocsolver_spotrf "POTRF".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B.
    @param[out]
    D           pointer to real type. Array on the GPU of dimension n.\n
                On exit, the eigenvalues in increasing order.
    @param[out]
    E           pointer to real type. Array on the GPU of dimension n.\n
                This array is used to work internally with the tridiagonal matrix T associated with
                the reduced eigenvalue problem.
                On exit, if 0 < info <= n, it contains the unconverged off-diagonal elements of T
                (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
                of this matrix are in D; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info = n + i, the leading minor of order i of B is not
                positive definite.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegvd(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_evect evect,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* B,
                                                 const rocblas_int ldb,
                                                 float* D,
                                                 float* E,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegvd(rocblas_handle handle,
                                                 const rocblas_eform itype,
                                                 const rocblas_evect evect,
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
    \brief SYGVD_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of real generalized symmetric-definite eigenproblems.

    \details
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^T B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^T B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                On entry, the symmetric positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info[j] = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygvd_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_evect evect,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygvd_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_evect evect,
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
    \brief HEGVD_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of complex generalized hermitian-definite eigenproblems.

    \details
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^H B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^H B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the hermitian matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[out]
    B           array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*n.\n
                On entry, the hermitian positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_batched "POTRF_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info[j] = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegvd_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_evect evect,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegvd_batched(rocblas_handle handle,
                                                         const rocblas_eform itype,
                                                         const rocblas_evect evect,
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
    \brief SYGVD_STRIDED_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of real generalized symmetric-definite eigenproblems.

    \details
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^T B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^T B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the symmetric matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                On entry, the symmetric positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D           pointer to type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info[j] = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_evect evect,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_evect evect,
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
    \brief HEGVD_STRIDED_BATCHED computes the eigenvalues and (optionally)
    eigenvectors of a batch of complex generalized hermitian-definite eigenproblems.

    \details
    For each instance in the batch, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = \lambda B_j X_j & \: \text{1st form,}\\
        A_j B_j X_j = \lambda X_j & \: \text{2nd form, or}\\
        B_j A_j X_j = \lambda X_j & \: \text{3rd form,}
        \end{array}
    \f]

    depending on the value of itype. The eigenvectors are computed using a divide-and-conquer algorithm, depending on the
    value of evect.

    When computed, the matrix \f$Z_j\f$ of eigenvectors is normalized as follows:

    \f[
        \begin{array}{cl}
        Z_j^H B_j Z_j=I & \: \text{if 1st or 2nd form, or}\\
        Z_j^H B_j^{-1} Z_j=I & \: \text{if 3rd form.}
        \end{array}
    \f]

    @param[in]
    handle      rocblas_handle.
    @param[in]
    itype       #rocblas_eform.\n
                Specifies the form of the generalized eigenproblems.
    @param[in]
    evect       #rocblas_evect.\n
                Specifies whether the eigenvectors are to be computed.
                If evect is rocblas_evect_original, then the eigenvectors are computed.
                rocblas_evect_tridiagonal is not supported.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower parts of the matrices
                A_j and B_j are stored. If uplo indicates lower (or upper),
                then the upper (or lower) parts of A_j and B_j are not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The matrix dimensions.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the hermitian matrices A_j. On exit, if evect is original,
                the normalized matrix Z_j of eigenvectors. If evect is none, then the upper or lower triangular
                part of the matrices A_j (including the diagonal) are destroyed,
                depending on the value of uplo.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use is strideA >= lda*n.
    @param[out]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
                On entry, the hermitian positive definite matrices B_j. On exit, the
                triangular factor of B_j as returned by \ref rocsolver_spotrf_strided_batched "POTRF_STRIDED_BATCHED".
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                Specifies the leading dimension of B_j.
    @param[in]
    strideB     rocblas_stride.\n
                Stride from the start of one matrix B_j to the next one B_(j+1).
                There is no restriction for the value of strideB. Normal use is strideB >= ldb*n.
    @param[out]
    D           pointer to real type. Array on the GPU (the size depends on the value of strideD).\n
                On exit, the eigenvalues in increasing order.
    @param[in]
    strideD     rocblas_stride.\n
                Stride from the start of one vector D_j to the next one D_(j+1).
                There is no restriction for the value of strideD. Normal use is strideD >= n.
    @param[out]
    E           pointer to real type. Array on the GPU (the size depends on the value of strideE).\n
                This array is used to work internally with the tridiagonal matrix T_j associated with
                the jth reduced eigenvalue problem.
                On exit, if 0 < info[j] <= n, it contains the unconverged off-diagonal elements of T_j
                (or properly speaking, a tridiagonal matrix equivalent to T_j). The diagonal elements
                of this matrix are in D_j; those that converged correspond to a subset of the
                eigenvalues (not necessarily ordered).
    @param[in]
    strideE     rocblas_stride.\n
                Stride from the start of one vector E_j to the next one E_(j+1).
                There is no restriction for the value of strideE. Normal use is strideE >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit of batch j.
                If info[j] = i <= n and evect is rocblas_evect_none, i off-diagonal elements of an
                intermediate tridiagonal form did not converge to zero.
                If info[j] = i <= n and evect is rocblas_evect_original, the algorithm failed to
                compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].
                If info[j] = n + i, the leading minor of order i of B_j is not
                positive definite.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_chegvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_evect evect,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegvd_strided_batched(rocblas_handle handle,
                                                                 const rocblas_eform itype,
                                                                 const rocblas_evect evect,
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

/*! @{
    \brief GETRI_OUTOFPLACE computes the inverse \f$C = A^{-1}\f$ of a general n-by-n matrix A.

    \details
    The inverse is computed by solving the linear system

    \f[
        AC = I
    \f]

    where I is the identity matrix, and A is factorized as \f$A = PLU\f$ as given by \ref rocsolver_sgetrf "GETRF".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                The factors L and U of the factorization A = P*L*U returned by \ref rocsolver_sgetrf "GETRF".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by \ref rocsolver_sgetrf "GETRF".
    @param[out]
    C           pointer to type. Array on the GPU of dimension ldc*n.\n
                If info = 0, the inverse of A. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_outofplace(rocblas_handle handle,
                                                            const rocblas_int n,
                                                            float* A,
                                                            const rocblas_int lda,
                                                            rocblas_int* ipiv,
                                                            float* C,
                                                            const rocblas_int ldc,
                                                            rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_outofplace(rocblas_handle handle,
                                                            const rocblas_int n,
                                                            double* A,
                                                            const rocblas_int lda,
                                                            rocblas_int* ipiv,
                                                            double* C,
                                                            const rocblas_int ldc,
                                                            rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_outofplace(rocblas_handle handle,
                                                            const rocblas_int n,
                                                            rocblas_float_complex* A,
                                                            const rocblas_int lda,
                                                            rocblas_int* ipiv,
                                                            rocblas_float_complex* C,
                                                            const rocblas_int ldc,
                                                            rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_outofplace(rocblas_handle handle,
                                                            const rocblas_int n,
                                                            rocblas_double_complex* A,
                                                            const rocblas_int lda,
                                                            rocblas_int* ipiv,
                                                            rocblas_double_complex* C,
                                                            const rocblas_int ldc,
                                                            rocblas_int* info);
//! @}

/*! @{
    \brief GETRI_OUTOFPLACE_BATCHED computes the inverse \f$C_j = A_j^{-1}\f$ of a batch of general n-by-n matrices \f$A_j\f$.

    \details
    The inverse is computed by solving the linear system

    \f[
        A_j C_j = I
    \f]

    where I is the identity matrix, and \f$A_j\f$ is factorized as \f$A_j = P_j  L_j  U_j\f$ as given by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[in]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The pivot indices returned by \ref rocsolver_sgetrf_batched "GETRF_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(i+j).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    C           array of pointers to type. Each pointer points to an array on the GPU of dimension ldc*n.\n
                If info[j] = 0, the inverse of matrices A_j. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    float* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    float* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    double* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    double* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    rocblas_float_complex* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    rocblas_float_complex* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    rocblas_double_complex* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    rocblas_double_complex* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI_OUTOFPLACE_STRIDED_BATCHED computes the inverse \f$C_j = A_j^{-1}\f$ of a batch of general n-by-n matrices \f$A_j\f$.

    \details
    The inverse is computed by solving the linear system

    \f[
        A_j C_j = I
    \f]

    where I is the identity matrix, and \f$A_j\f$ is factorized as \f$A_j = P_j  L_j  U_j\f$ as given by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                The factors L_j and U_j of the factorization A_j = P_j*L_j*U_j returned by
                \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
                The pivot indices returned by \ref rocsolver_sgetrf_strided_batched "GETRF_STRIDED_BATCHED".
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    C           pointer to type. Array on the GPU (the size depends on the value of strideC).\n
                If info[j] = 0, the inverse of matrices A_j. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C_j.
    @param[in]
    strideC     rocblas_stride.\n
                Stride from the start of one matrix C_j to the next one C_(j+1).
                There is no restriction for the value of strideC. Normal use case is strideC >= ldc*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status
    rocsolver_sgetri_outofplace_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                float* C,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_dgetri_outofplace_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                double* C,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_cgetri_outofplace_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_float_complex* C,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_zgetri_outofplace_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_double_complex* C,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI_NPVT_OUTOFPLACE computes the inverse \f$C = A^{-1}\f$ of a general n-by-n matrix A without partial pivoting.

    \details
    The inverse is computed by solving the linear system

    \f[
        AC = I
    \f]

    where I is the identity matrix, and A is factorized as \f$A = LU\f$ as given by \ref rocsolver_sgetrf_npvt "GETRF_NPVT".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                The factors L and U of the factorization A = L*U returned by \ref rocsolver_sgetrf_npvt "GETRF_NPVT".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    C           pointer to type. Array on the GPU of dimension ldc*n.\n
                If info = 0, the inverse of A. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, U is singular. U[i,i] is the first zero pivot.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_npvt_outofplace(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 float* C,
                                                                 const rocblas_int ldc,
                                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_npvt_outofplace(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 double* C,
                                                                 const rocblas_int ldc,
                                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_npvt_outofplace(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 rocblas_float_complex* C,
                                                                 const rocblas_int ldc,
                                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_npvt_outofplace(rocblas_handle handle,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 rocblas_double_complex* C,
                                                                 const rocblas_int ldc,
                                                                 rocblas_int* info);
//! @}

/*! @{
    \brief GETRI_NPVT_OUTOFPLACE_BATCHED computes the inverse \f$C_j = A_j^{-1}\f$ of a batch of general n-by-n matrices \f$A_j\f$
    without partial pivoting.

    \details
    The inverse is computed by solving the linear system

    \f[
        A_j C_j = I
    \f]

    where I is the identity matrix, and \f$A_j\f$ is factorized as \f$A_j = L_j  U_j\f$ as given by \ref rocsolver_sgetrf_npvt_batched "GETRF_NPVT_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[in]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                The factors L_j and U_j of the factorization A_j = L_j*U_j returned by \ref rocsolver_sgetrf_npvt_batched "GETRF_NPVT_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    C           array of pointers to type. Each pointer points to an array on the GPU of dimension ldc*n.\n
                If info[j] = 0, the inverse of matrices A_j. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                                         const rocblas_int n,
                                                                         float* const A[],
                                                                         const rocblas_int lda,
                                                                         float* const C[],
                                                                         const rocblas_int ldc,
                                                                         rocblas_int* info,
                                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                                         const rocblas_int n,
                                                                         double* const A[],
                                                                         const rocblas_int lda,
                                                                         double* const C[],
                                                                         const rocblas_int ldc,
                                                                         rocblas_int* info,
                                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_cgetri_npvt_outofplace_batched(rocblas_handle handle,
                                             const rocblas_int n,
                                             rocblas_float_complex* const A[],
                                             const rocblas_int lda,
                                             rocblas_float_complex* const C[],
                                             const rocblas_int ldc,
                                             rocblas_int* info,
                                             const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_zgetri_npvt_outofplace_batched(rocblas_handle handle,
                                             const rocblas_int n,
                                             rocblas_double_complex* const A[],
                                             const rocblas_int lda,
                                             rocblas_double_complex* const C[],
                                             const rocblas_int ldc,
                                             rocblas_int* info,
                                             const rocblas_int batch_count);
//! @}

/*! @{
    \brief GETRI_NPVT_OUTOFPLACE_STRIDED_BATCHED computes the inverse \f$C_j = A_j^{-1}\f$ of a batch of general n-by-n matrices \f$A_j\f$
    without partial pivoting.

    \details
    The inverse is computed by solving the linear system

    \f[
        A_j C_j = I
    \f]

    where I is the identity matrix, and \f$A_j\f$ is factorized as \f$A_j = L_j  U_j\f$ as given by \ref rocsolver_sgetrf_npvt_strided_batched "GETRF_NPVT_STRIDED_BATCHED".

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[in]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                The factors L_j and U_j of the factorization A_j = L_j*U_j returned by
                \ref rocsolver_sgetrf_npvt_strided_batched "GETRF_NPVT_STRIDED_BATCHED".
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    C           pointer to type. Array on the GPU (the size depends on the value of strideC).\n
                If info[j] = 0, the inverse of matrices A_j. Otherwise, undefined.
    @param[in]
    ldc         rocblas_int. ldc >= n.\n
                Specifies the leading dimension of C_j.
    @param[in]
    strideC     rocblas_stride.\n
                Stride from the start of one matrix C_j to the next one C_(j+1).
                There is no restriction for the value of strideC. Normal use case is strideC >= ldc*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, U_j is singular. U_j[i,i] is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status
    rocsolver_sgetri_npvt_outofplace_strided_batched(rocblas_handle handle,
                                                     const rocblas_int n,
                                                     float* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     float* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_dgetri_npvt_outofplace_strided_batched(rocblas_handle handle,
                                                     const rocblas_int n,
                                                     double* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     double* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_cgetri_npvt_outofplace_strided_batched(rocblas_handle handle,
                                                     const rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_float_complex* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status
    rocsolver_zgetri_npvt_outofplace_strided_batched(rocblas_handle handle,
                                                     const rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_double_complex* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count);
//! @}

/*! @{
    \brief TRTRI inverts a triangular n-by-n matrix A.

    \details
    A can be upper or lower triangular, depending on the value of uplo, and unit or non-unit
    triangular, depending on the value of diag.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    diag        rocblas_diagonal.\n
                If diag indicates unit, then the diagonal elements of A are not referenced and
                assumed to be one.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the triangular matrix.
                On exit, the inverse of A if info = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, A is singular. A[i,i] is the first zero element in the diagonal.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_strtri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_diagonal diag,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dtrtri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_diagonal diag,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_ctrtri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_diagonal diag,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_ztrtri(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_diagonal diag,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief TRTRI_BATCHED inverts a batch of triangular n-by-n matrices \f$A_j\f$.

    \details
    \f$A_j\f$ can be upper or lower triangular, depending on the value of uplo, and unit or non-unit
    triangular, depending on the value of diag.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    diag        rocblas_diagonal.\n
                If diag indicates unit, then the diagonal elements of matrices A_j are not referenced and
                assumed to be one.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the triangular matrices A_j.
                On exit, the inverses of A_j if info[j] = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, A_j is singular. A_j[i,i] is the first zero element in the diagonal.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_strtri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_diagonal diag,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dtrtri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_diagonal diag,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_ctrtri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_diagonal diag,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_ztrtri_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_diagonal diag,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief TRTRI_STRIDED_BATCHED inverts a batch of triangular n-by-n matrices \f$A_j\f$.

    \details
    \f$A_j\f$ can be upper or lower triangular, depending on the value of uplo, and unit or non-unit
    triangular, depending on the value of diag.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    diag        rocblas_diagonal.\n
                If diag indicates unit, then the diagonal elements of matrices A_j are not referenced and
                assumed to be one.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the triangular matrices A_j.
                On exit, the inverses of A_j if info[j] = 0.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for inversion of A_j.
                If info[j] = i > 0, A_j is singular. A_j[i,i] is the first zero element in the diagonal.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_strtri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_diagonal diag,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dtrtri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_diagonal diag,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_ctrtri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_diagonal diag,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_ztrtri_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_diagonal diag,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTF2 computes the factorization of a symmetric indefinite matrix \f$A\f$
    using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A = U D U^T & \: \text{or}\\
        A = L D L^T &
        \end{array}
    \f]

    where \f$U\f$ or \f$L\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D(k)\f$.

    Specifically, \f$U\f$ and \f$L\f$ are computed as

    \f[
        \begin{array}{cl}
        U = P(n) U(n) \cdots P(k) U(k) \cdots & \: \text{and}\\
        L = P(1) L(1) \cdots P(k) L(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D(k)\f$, and \f$P(k)\f$ is a permutation matrix defined by
    \f$ipiv[k]\f$. If we let \f$s\f$ denote the order of block \f$D(k)\f$, then \f$U(k)\f$
    and \f$L(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D(k)\f$ is stored in \f$A[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D(k)\f$ is stored in \f$A[k-1,k-1]\f$, \f$A[k-1,k]\f$,
    and \f$A[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D(k)\f$ is stored in \f$A[k,k]\f$, \f$A[k+1,k]\f$,
    and \f$A[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrix A to be factored.
                On exit, the block diagonal matrix D and the multipliers needed to
                compute U or L.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv[k] > 0 then rows and columns k and ipiv[k]
                were interchanged and D[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv[k] = ipiv[k-1] < 0 and uplo is upper (or ipiv[k]
                = ipiv[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv[k] (or rows and columns k+1 and -ipiv[k]) were interchanged
                and D[k-1,k-1] to D[k,k] (or D[k,k] to D[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, D is singular. D[i,i] is the first diagonal zero.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytf2(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief SYTF2_BATCHED computes the factorization of a batch of symmetric indefinite
    matrices using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A_j = U_j D_j U_j^T & \: \text{or}\\
        A_j = L_j D_j L_j^T &
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D_j\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D_j(k)\f$.

    Specifically, \f$U_j\f$ and \f$L_j\f$ are computed as

    \f[
        \begin{array}{cl}
        U_j = P_j(n) U_j(n) \cdots P_j(k) U_j(k) \cdots & \: \text{and}\\
        L_j = P_j(1) L_j(1) \cdots P_j(k) L_j(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D_j(k)\f$, and \f$P_j(k)\f$ is a permutation matrix defined by
    \f$ipiv_j[k]\f$. If we let \f$s\f$ denote the order of block \f$D_j(k)\f$, then \f$U_j(k)\f$
    and \f$L_j(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U_j(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L_j(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D_j(k)\f$ is stored in \f$A_j[k-1,k-1]\f$, \f$A_j[k-1,k]\f$,
    and \f$A_j[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$, \f$A_j[k+1,k]\f$,
    and \f$A_j[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A_j\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrices A_j to be factored.
                On exit, the block diagonal matrices D_j and the multipliers needed to
                compute U_j or L_j.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv_j[k] > 0 then rows and columns k and ipiv_j[k]
                were interchanged and D_j[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv_j[k] = ipiv_j[k-1] < 0 and uplo is upper (or ipiv_j[k]
                = ipiv_j[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv_j[k] (or rows and columns k+1 and -ipiv_j[k]) were interchanged
                and D_j[k-1,k-1] to D_j[k,k] (or D_j[k,k] to D_j[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, D_j is singular. D_j[i,i] is the first diagonal zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytf2_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTF2_STRIDED_BATCHED computes the factorization of a batch of symmetric indefinite
    matrices using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A_j = U_j D_j U_j^T & \: \text{or}\\
        A_j = L_j D_j L_j^T &
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D_j\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D_j(k)\f$.

    Specifically, \f$U_j\f$ and \f$L_j\f$ are computed as

    \f[
        \begin{array}{cl}
        U_j = P_j(n) U_j(n) \cdots P_j(k) U_j(k) \cdots & \: \text{and}\\
        L_j = P_j(1) L_j(1) \cdots P_j(k) L_j(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D_j(k)\f$, and \f$P_j(k)\f$ is a permutation matrix defined by
    \f$ipiv_j[k]\f$. If we let \f$s\f$ denote the order of block \f$D_j(k)\f$, then \f$U_j(k)\f$
    and \f$L_j(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U_j(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L_j(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D_j(k)\f$ is stored in \f$A_j[k-1,k-1]\f$, \f$A_j[k-1,k]\f$,
    and \f$A_j[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$, \f$A_j[k+1,k]\f$,
    and \f$A_j[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A_j\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the symmetric matrices A_j to be factored.
                On exit, the block diagonal matrices D_j and the multipliers needed to
                compute U_j or L_j.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv_j[k] > 0 then rows and columns k and ipiv_j[k]
                were interchanged and D_j[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv_j[k] = ipiv_j[k-1] < 0 and uplo is upper (or ipiv_j[k]
                = ipiv_j[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv_j[k] (or rows and columns k+1 and -ipiv_j[k]) were interchanged
                and D_j[k-1,k-1] to D_j[k,k] (or D_j[k,k] to D_j[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, D_j is singular. D_j[i,i] is the first diagonal zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytf2_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
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
    \brief SYTRF computes the factorization of a symmetric indefinite matrix \f$A\f$
    using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A = U D U^T & \: \text{or}\\
        A = L D L^T &
        \end{array}
    \f]

    where \f$U\f$ or \f$L\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D(k)\f$.

    Specifically, \f$U\f$ and \f$L\f$ are computed as

    \f[
        \begin{array}{cl}
        U = P(n) U(n) \cdots P(k) U(k) \cdots & \: \text{and}\\
        L = P(1) L(1) \cdots P(k) L(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D(k)\f$, and \f$P(k)\f$ is a permutation matrix defined by
    \f$ipiv[k]\f$. If we let \f$s\f$ denote the order of block \f$D(k)\f$, then \f$U(k)\f$
    and \f$L(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D(k)\f$ is stored in \f$A[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D(k)\f$ is stored in \f$A[k-1,k-1]\f$, \f$A[k-1,k]\f$,
    and \f$A[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D(k)\f$ is stored in \f$A[k,k]\f$, \f$A[k+1,k]\f$,
    and \f$A[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrix A is stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of the matrix A.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrix A to be factored.
                On exit, the block diagonal matrix D and the multipliers needed to
                compute U or L.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of A.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv[k] > 0 then rows and columns k and ipiv[k]
                were interchanged and D[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv[k] = ipiv[k-1] < 0 and uplo is upper (or ipiv[k]
                = ipiv[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv[k] (or rows and columns k+1 and -ipiv[k]) were interchanged
                and D[k-1,k-1] to D[k,k] (or D[k,k] to D[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[out]
    info        pointer to a rocblas_int on the GPU.\n
                If info = 0, successful exit.
                If info = i > 0, D is singular. D[i,i] is the first diagonal zero.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* ipiv,
                                                 rocblas_int* info);
//! @}

/*! @{
    \brief SYTRF_BATCHED computes the factorization of a batch of symmetric indefinite
    matrices using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A_j = U_j D_j U_j^T & \: \text{or}\\
        A_j = L_j D_j L_j^T &
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D_j\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D_j(k)\f$.

    Specifically, \f$U_j\f$ and \f$L_j\f$ are computed as

    \f[
        \begin{array}{cl}
        U_j = P_j(n) U_j(n) \cdots P_j(k) U_j(k) \cdots & \: \text{and}\\
        L_j = P_j(1) L_j(1) \cdots P_j(k) L_j(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D_j(k)\f$, and \f$P_j(k)\f$ is a permutation matrix defined by
    \f$ipiv_j[k]\f$. If we let \f$s\f$ denote the order of block \f$D_j(k)\f$, then \f$U_j(k)\f$
    and \f$L_j(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U_j(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L_j(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D_j(k)\f$ is stored in \f$A_j[k-1,k-1]\f$, \f$A_j[k-1,k]\f$,
    and \f$A_j[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$, \f$A_j[k+1,k]\f$,
    and \f$A_j[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A_j\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
                On entry, the symmetric matrices A_j to be factored.
                On exit, the block diagonal matrices D_j and the multipliers needed to
                compute U_j or L_j.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv_j[k] > 0 then rows and columns k and ipiv_j[k]
                were interchanged and D_j[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv_j[k] = ipiv_j[k-1] < 0 and uplo is upper (or ipiv_j[k]
                = ipiv_j[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv_j[k] (or rows and columns k+1 and -ipiv_j[k]) were interchanged
                and D_j[k-1,k-1] to D_j[k,k] (or D_j[k,k] to D_j[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, D_j is singular. D_j[i,i] is the first diagonal zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         float* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         double* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_float_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytrf_batched(rocblas_handle handle,
                                                         const rocblas_fill uplo,
                                                         const rocblas_int n,
                                                         rocblas_double_complex* const A[],
                                                         const rocblas_int lda,
                                                         rocblas_int* ipiv,
                                                         const rocblas_stride strideP,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count);
//! @}

/*! @{
    \brief SYTRF_STRIDED_BATCHED computes the factorization of a batch of symmetric indefinite
    matrices using Bunch-Kaufman diagonal pivoting.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

    \f[
        \begin{array}{cl}
        A_j = U_j D_j U_j^T & \: \text{or}\\
        A_j = L_j D_j L_j^T &
        \end{array}
    \f]

    where \f$U_j\f$ or \f$L_j\f$ is a product of permutation and unit upper/lower
    triangular matrices (depending on the value of uplo), and \f$D_j\f$ is a symmetric
    block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks \f$D_j(k)\f$.

    Specifically, \f$U_j\f$ and \f$L_j\f$ are computed as

    \f[
        \begin{array}{cl}
        U_j = P_j(n) U_j(n) \cdots P_j(k) U_j(k) \cdots & \: \text{and}\\
        L_j = P_j(1) L_j(1) \cdots P_j(k) L_j(k) \cdots &
        \end{array}
    \f]

    where \f$k\f$ decreases from \f$n\f$ to 1 (increases from 1 to \f$n\f$) in steps of 1 or 2,
    depending on the order of block \f$D_j(k)\f$, and \f$P_j(k)\f$ is a permutation matrix defined by
    \f$ipiv_j[k]\f$. If we let \f$s\f$ denote the order of block \f$D_j(k)\f$, then \f$U_j(k)\f$
    and \f$L_j(k)\f$ are unit upper/lower triangular matrices defined as

    \f[
        U_j(k) = \left[ \begin{array}{ccc}
        I_{k-s} & v & 0 \\
        0 & I_s & 0 \\
        0 & 0 & I_{n-k}
        \end{array} \right]
    \f]

    and

    \f[
        L_j(k) = \left[ \begin{array}{ccc}
        I_{k-1} & 0 & 0 \\
        0 & I_s & 0 \\
        0 & v & I_{n-k-s+1}
        \end{array} \right].
    \f]

    If \f$s = 1\f$, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$ and \f$v\f$ is stored in the upper/lower
    part of column \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is upper, then \f$D_j(k)\f$ is stored in \f$A_j[k-1,k-1]\f$, \f$A_j[k-1,k]\f$,
    and \f$A_j[k,k]\f$, and \f$v\f$ is stored in the upper parts of columns \f$k-1\f$ and \f$k\f$ of \f$A_j\f$.
    If \f$s = 2\f$ and uplo is lower, then \f$D_j(k)\f$ is stored in \f$A_j[k,k]\f$, \f$A_j[k+1,k]\f$,
    and \f$A_j[k+1,k+1]\f$, and \f$v\f$ is stored in the lower parts of columns \f$k\f$ and \f$k+1\f$ of \f$A_j\f$.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    uplo        rocblas_fill.\n
                Specifies whether the upper or lower part of the matrices A_j are stored.
                If uplo indicates lower (or upper), then the upper (or lower)
                part of A_j is not used.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The number of rows and columns of all matrices A_j in the batch.
    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
                On entry, the symmetric matrices A_j to be factored.
                On exit, the block diagonal matrices D_j and the multipliers needed to
                compute U_j or L_j.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA     rocblas_stride.\n
                Stride from the start of one matrix A_j to the next one A_(j+1).
                There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The vector of pivot indices. Elements of ipiv are 1-based indices.
                For 1 <= k <= n, if ipiv_j[k] > 0 then rows and columns k and ipiv_j[k]
                were interchanged and D_j[k,k] is a 1-by-1 diagonal block.
                If, instead, ipiv_j[k] = ipiv_j[k-1] < 0 and uplo is upper (or ipiv_j[k]
                = ipiv_j[k+1] < 0 and uplo is lower), then rows and columns k-1 and
                -ipiv_j[k] (or rows and columns k+1 and -ipiv_j[k]) were interchanged
                and D_j[k-1,k-1] to D_j[k,k] (or D_j[k,k] to D_j[k+1,k+1]) is a 2-by-2
                diagonal block.
    @param[in]
    strideP     rocblas_stride.\n
                Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
                There is no restriction for the value of strideP. Normal use case is strideP >= n.
    @param[out]
    info        pointer to rocblas_int. Array of batch_count integers on the GPU.\n
                If info[j] = 0, successful exit for factorization of A_j.
                If info[j] = i > 0, D_j is singular. D_j[i,i] is the first diagonal zero.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.
    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 float* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 double* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_csytrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_float_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytrf_strided_batched(rocblas_handle handle,
                                                                 const rocblas_fill uplo,
                                                                 const rocblas_int n,
                                                                 rocblas_double_complex* A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_stride strideA,
                                                                 rocblas_int* ipiv,
                                                                 const rocblas_stride strideP,
                                                                 rocblas_int* info,
                                                                 const rocblas_int batch_count);
//! @}

#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */
