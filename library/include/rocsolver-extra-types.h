/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSOLVER_EXTRAS_H_
#define ROCSOLVER_EXTRAS_H_

/*! \brief Used to specify the logging layer mode using a bitwise combination
 *of rocblas_layer_mode values.
 ********************************************************************************/
typedef uint32_t rocblas_layer_mode_flags;

/*! \brief Used to specify the order in which multiple elementary matrices are
 *applied together
 ********************************************************************************/
typedef enum rocblas_direct_
{
    rocblas_forward_direction = 171, /**< Elementary matrices applied from the right. */
    rocblas_backward_direction = 172, /**< Elementary matrices applied from the left. */
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
                               requires enough free memory. */
    rocblas_inplace = 202, /**< When not enough memory, this forces in-place computations  */
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
    rocblas_eform_ax = 221, /**< The problem is A*x = lambda*B*x. */
    rocblas_eform_abx = 222, /**< The problem is A*B*x = lambda*x. */
    rocblas_eform_bax = 223, /**< The problem is B*A*x = lambda*x. */
} rocblas_eform;

#endif /* ROCSOLVER_EXTRAS_H_ */
