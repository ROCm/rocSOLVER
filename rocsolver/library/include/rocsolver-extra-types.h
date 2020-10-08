/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSOLVER_EXTRAS_H_
#define ROCSOLVER_EXTRAS_H_

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

#endif
