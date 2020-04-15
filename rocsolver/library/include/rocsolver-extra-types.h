/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _EXTRAS_H_
#define _EXTRAS_H_

/*! \brief Used to specify the order in which multiple elementary matrices are applied together 
 ********************************************************************************/ 
typedef enum rocblas_direct_
{
    rocblas_forward_direction = 171, /**< Elementary matrices applied from the right. */
    rocblas_backward_direction = 172, /**< Elementary matrices applied from the left. */
} rocblas_direct;

/*! \brief Used to specify how householder vectors are stored in a matrix of vectors 
 ********************************************************************************/ 
typedef enum rocblas_storev_
{
    rocblas_column_wise = 181, /**< Householder vectors are stored in the columns of a matrix. */
    rocblas_row_wise = 182, /**< Householder vectors are stored in the rows of a matrix. */
} rocblas_storev;

#endif
