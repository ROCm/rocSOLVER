/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
    \brief rocsolver-types.h defines data types used by rocsolver.
 ***************************************************************************/

#ifndef _ROCSOLVER_TYPES_H_
#define _ROCSOLVER_TYPES_H_

#include <rocblas.h>

/*! \brief Used to specify int32 or int64. 
    \details rocsolver_int is a rocblas_int 
 ******************************************************************/
typedef rocblas_int rocsolver_int;

typedef rocblas_float_complex rocsolver_float_complex;
typedef rocblas_double_complex rocsolver_double_complex;
typedef rocblas_half rocsolver_half;

/*! \brief A structure holding the rocsolver library context. 
    \details 
    It must be initialized using rocsolver_create_handle()
    and the returned handle must be passed to all subsequent library 
    function calls. It should be destroyed at the end using rocsolver_destroy_handle().\n
    rocsolver_handle is a rocblas_handle. 
 *************************************************************************/
typedef rocblas_handle rocsolver_handle;

/*! \brief Used to specify whether the matrix is to be transposed.
    \details rocsolver_operation is a rocblas_operation
 ***************************************************************************/
typedef rocblas_operation rocsolver_operation;

/*! \brief Used to specify whether the upper or lower triangle in a matrix is referenced
    \details rocsolver_fill is a rocblas_fill
 ***************************************************************************/
typedef rocblas_fill rocsolver_fill;

/*! \brief Used to specify whether a matrix has ones along the diagonal
    \details rocsolver_diagonal is a rocblas_diagonal
 ***************************************************************************/
typedef rocblas_diagonal rocsolver_diagonal;

/*! \brief Used to specify whether matrix multiplication is done by the right or left
    \details rocsolver_side is a rocblas_side
 ***************************************************************************/
typedef rocblas_side rocsolver_side;

/*! \brief The rocSOLVER status code definition
    \details rocsolver_status is a rocblas_status
 ***************************************************************************/
typedef rocblas_status rocsolver_status;

typedef rocblas_layer_mode rocsolver_layer_mode;

/*! \brief Used to specify the order in which multiple elementary matrices are applied together 
 ********************************************************************************/ 
typedef enum rocsolver_direct_
{
    rocsolver_forward_direction = 171, /**< Elementary matrices applied from the right. */
    rocsolver_backward_direction = 172, /**< Elementary matrices applied from the left. */
} rocsolver_direct;

/*! \brief Used to specify how householder vectors are stored in a matrix of vectors 
 ********************************************************************************/ 
typedef enum rocsolver_storev_
{
    rocsolver_column_wise = 181, /**< Householder vectors are stored in the columns of a matrix. */
    rocsolver_row_wise = 182, /**< Householder vectors are stored in the rows of a matrix. */
} rocsolver_storev;
#endif
