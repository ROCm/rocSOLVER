/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSOLVER_EXTRAS_H_
#define ROCSOLVER_EXTRAS_H_

#include <stdint.h>

/*! \brief Used to specify the logging layer mode using a bitwise combination
 *of rocblas_layer_mode values.
 ********************************************************************************/
typedef uint32_t rocblas_layer_mode_flags;

/*! \brief Used to expand the logging layer modes offered for rocSOLVER logging.
 ********************************************************************************/
typedef enum rocblas_layer_mode_ex_
{
    rocblas_layer_mode_ex_log_kernel = 0x10, /**< Enable logging for kernel calls. */
} rocblas_layer_mode_ex;

/*! \brief Used to specify the order in which multiple Householder matrices are
 *applied together
 ********************************************************************************/
typedef enum rocblas_direct_
{
    rocblas_forward_direction = 171, /**< Householder matrices applied from the right. */
    rocblas_backward_direction = 172, /**< Householder matrices applied from the left. */
} rocblas_direct;

/*! \brief Used to specify how householder vectors are stored in a matrix of
 *vectors
 ********************************************************************************/
typedef enum rocblas_storev_
{
    rocblas_column_wise = 181, /**< Householder vectors are stored in the columns of a matrix. */
    rocblas_row_wise = 182, /**< Householder vectors are stored in the rows of a matrix. */
} rocblas_storev;

/*! \brief Used to specify how the singular vectors are to be computed and
 *stored
 ********************************************************************************/
typedef enum rocblas_svect_
{
    rocblas_svect_all = 191, /**< The entire associated orthogonal/unitary matrix is computed. */
    rocblas_svect_singular = 192, /**< Only the singular vectors are computed and
                                       stored in output array. */
    rocblas_svect_overwrite = 193, /**< Only the singular vectors are computed and
                                        overwrite the input matrix. */
    rocblas_svect_none = 194, /**< No singular vectors are computed. */
} rocblas_svect;

/*! \brief Used to enable the use of fast algorithms (with out-of-place
 *computations) in some of the routines
 ********************************************************************************/
typedef enum rocblas_workmode_
{
    rocblas_outofplace = 201, /**< Out-of-place computations are allowed; this
                                   requires extra device memory for workspace. */
    rocblas_inplace = 202, /**< If not enough memory is available, this forces in-place computations.  */
} rocblas_workmode;

/*! \brief Used to specify how the eigenvectors are to be computed
 ********************************************************************************/
typedef enum rocblas_evect_
{
    rocblas_evect_original = 211, /**< Compute eigenvectors for the original symmetric/Hermitian
                                       matrix. */
    rocblas_evect_tridiagonal = 212, /**< Compute eigenvectors for the symmetric tridiagonal
                                           matrix. */
    rocblas_evect_none = 213, /**< No eigenvectors are computed. */
} rocblas_evect;

/*! \brief Used to specify the form of the generalized eigenproblem
 ********************************************************************************/
typedef enum rocblas_eform_
{
    rocblas_eform_ax = 221, /**< The problem is \f$Ax = \lambda Bx\f$. */
    rocblas_eform_abx = 222, /**< The problem is \f$ABx = \lambda x\f$. */
    rocblas_eform_bax = 223, /**< The problem is \f$BAx = \lambda x\f$. */
} rocblas_eform;

/*! \brief Used to specify the type of range in which eigenvalues will be found
 *in partial eigenvalue decompositions
 ********************************************************************************/
typedef enum rocblas_erange_
{
    rocblas_erange_all = 231, /**< All eigenvalues will be found. */
    rocblas_erange_value = 232, /**< All eigenvalues in the half-open interval
                                     \f$(vl, vu]\f$ will be found. */
    rocblas_erange_index = 233, /**< The \f$il\f$-th through \f$iu\f$-th eigenvalues will be found.*/
} rocblas_erange;

/*! \brief Used to specify whether the eigenvalues are grouped and ordered by blocks
 ********************************************************************************/
typedef enum rocblas_eorder_
{
    rocblas_eorder_blocks = 241, /**< The computed eigenvalues will be grouped by split-off
                                      blocks and arranged in increasing order within each block. */
    rocblas_eorder_entire = 242, /**< All computed eigenvalues of the entire matrix will be
                                      ordered from smallest to largest. */
} rocblas_eorder;

/*! \brief Used in the Jacobi methods to specify whether the eigenvalues are sorted
 *in increasing order
 ********************************************************************************/
typedef enum rocblas_esort_
{
    rocblas_esort_none = 251, /**< The computed eigenvalues will not be sorted. */
    rocblas_esort_ascending = 252, /**< The computed eigenvalues will be sorted in ascending order. */
} rocblas_esort;

/*! \brief Used to specify the type of range in which singular values will be found
 *in partial singular value decompositions
 ********************************************************************************/
typedef enum rocblas_srange_
{
    rocblas_srange_all = 261, /**< All singular values will be found. */
    rocblas_srange_value = 262, /**< All singular values in the half-open interval
                                     \f$(vl, vu]\f$ will be found. */
    rocblas_srange_index = 263, /**< The \f$il\f$-th through \f$iu\f$-th singular values will be found.*/
} rocblas_srange;

#endif /* ROCSOLVER_EXTRAS_H_ */
