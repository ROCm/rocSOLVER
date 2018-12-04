/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 * \brief rocsolver-types.h defines data types used by rocsolver
 */

#ifndef _ROCSOLVER_TYPES_H_
#define _ROCSOLVER_TYPES_H_

#include <rocblas.h>

typedef rocblas_int rocsolver_int;
typedef rocblas_float_complex rocsolver_float_complex;
typedef rocblas_double_complex rocsolver_double_complex;
typedef rocblas_half rocsolver_half;
typedef rocblas_half_complex rocsolver_half_complex;
typedef rocblas_handle rocsolver_handle;
typedef rocblas_operation rocsolver_operation;
typedef rocblas_fill rocsolver_fill;
typedef rocblas_diagonal rocsolver_diagonal;
typedef rocblas_side rocsolver_side;
typedef rocblas_status rocsolver_status;
typedef rocblas_datatype rocsolver_precision;
typedef rocblas_layer_mode rocsolver_layer_mode;

#endif
