/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//
//
// IMPORTANT:
//
// THESE ALIASES ARE NOT MAINTAINED ANYMORE. THEY EXIST ONLY FOR BACKWARDS
// COMPATIBILITY. USE ROCBLAS TYPES AND FUNCTIONS DIRECTLY.
//
//

#ifndef _ALIASES_H_
#define _ALIASES_H_

#include <rocblas.h>
#include "rocsolver-extra-types.h"
#include "rocsolver-export.h"

//rocblas original types

/*! \deprecated Use ::rocblas_int.
*/
typedef rocblas_int rocsolver_int
        __attribute__((deprecatred("use rocblas_int")));

/*! \deprecated Use ::rocblas_stride.
*/
typedef rocblas_stride rocsolver_stride
        __attribute__((deprecatred("use rocblas_stride")));

/*! \deprecated Use ::rocblas_float_complex.
*/
typedef rocblas_float_complex rocsolver_float_complex
        __attribute__((deprecatred("use rocblas_float_complex")));

/*! \deprecated Use ::rocblas_double_complex.
*/
typedef rocblas_double_complex rocsolver_double_complex
        __attribute__((deprecatred("use rocblas_double_complex")));

/*! \deprecated Use ::rocblas_half.
*/
typedef rocblas_half rocsolver_half
        __attribute__((deprecatred("use rocblas_half")));

/*! \deprecated Use ::rocblas_handle.
*/
typedef rocblas_handle rocsolver_handle
        __attribute__((deprecatred("use rocblas_handle")));

/*! \deprecated Use ::rocblas_operation.
*/
typedef rocblas_operation rocsolver_operation
        __attribute__((deprecatred("use rocblas_operation")));

/*! \deprecated Use ::rocblas_fill.
*/
typedef rocblas_fill rocsolver_fill
        __attribute__((deprecatred("use rocblas_fill")));

/*! \deprecated Use ::rocblas_diagonal.
*/
typedef rocblas_diagonal rocsolver_diagonal
        __attribute__((deprecatred("use rocblas_diagonal")));

/*! \deprecated Use ::rocblas_stide.
*/
typedef rocblas_side rocsolver_side;
        __attribute__((deprecatred("use rocblas_side")));

/*! \deprecated Use ::rocblas_status.
*/
typedef rocblas_status rocsolver_status;
        __attribute__((deprecatred("use rocblas_status")));

/*! \deprecated Use ::rocblas_layer_mode.
*/
typedef rocblas_layer_mode rocsolver_layer_mode;
        __attribute__((deprecatred("use rocblas_layer_mode")));


//extras types only used in rocsolver

/*! \deprecated Use ::rocblas_direction.
*/
typedef rocblas_direct rocsolver_direction
        __attribute__((deprecatred("use rocblas_direction")));

/*! \deprecated Use ::rocblas_storev.
*/
typedef rocblas_storev rocsolver_storev
        __attribute__((deprecatred("use rocblas_storev")));


//auxiliaries
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Creates a handle and sets the pointer mode to ::rocblas_pointer_mode_device.
    \deprecated Use ::rocblas_create_handle.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_create_handle(rocsolver_handle *handle)
        __attribute__((deprecatred("use rocblas_create_handle")))
{
  const rocblas_status stat = rocblas_create_handle(handle);
  if (stat != rocblas_status_success) {
    return stat;
  }
  return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

/*! \deprecated Use ::rocblas_destroy_handle.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_destroy_handle(rocsolver_handle handle)
        __attribute__((deprecatred("use rocblas_destroy_handle")))
{
  return rocblas_destroy_handle(handle);
}

// rocblas_add_stream was removed in ROCm 3.6; use rocblas_set_stream
//
//ROCSOLVER_EXPORT __inline rocsolver_status
//rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
//  return rocblas_add_stream(handle, stream);
//}

/*! \deprecated Use ::rocblas_set_stream.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream)
        __attribute__((deprecatred("use rocblas_set_stream")))
{
  return rocblas_set_stream(handle, stream);
}

/*! \deprecated Use ::rocblas_get_stream.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_stream(rocsolver_handle handle, hipStream_t *stream)
        __attribute__((deprecatred("use rocblas_get_stream")))
{
  return rocblas_get_stream(handle, stream);
}

/*! \deprecated Use ::rocblas_set_vector.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy)
{
        __attribute__((deprecatred("use rocblas_set_vector")))
  return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

/*! \deprecated Use ::rocblas_get_vector.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy)
        __attribute__((deprecatred("use rocblas_get_vector")))
{
  return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

/*! \deprecated Use ::rocblas_set_matrix.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb)
        __attribute__((deprecatred("use rocblas_set_matrix")))
{
  return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

/*! \deprecated Use ::rocblas_get_matrix.
*/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb)
        __attribute__((deprecatred("use rocblas_get_matrix")))
{
  return rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

#ifdef __cplusplus
}
#endif

#endif
