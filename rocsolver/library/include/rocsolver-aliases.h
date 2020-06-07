/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//rocsolver types and auxiliaries as aliases to rocblas types and auxiliaries
// DEPECRATED: ALIASES ARE NOT MAINTAINED ANYMORE. ONLY KEPT FOR BACKWARD
// COMPATIBILITY. EVERYTHING IS SWITCHED TO ROCBLAS NAMES.
// ***************************************************************************/

#ifndef _ALIASES_H_
#define _ALIASES_H_

#include <rocblas.h>
#include "rocsolver-extra-types.h"

//rocblas original types
typedef rocblas_int rocsolver_int;
typedef rocblas_stride rocsolver_stride;
typedef rocblas_float_complex rocsolver_float_complex;
typedef rocblas_double_complex rocsolver_double_complex;
typedef rocblas_half rocsolver_half;
typedef rocblas_handle rocsolver_handle;
typedef rocblas_operation rocsolver_operation;
typedef rocblas_fill rocsolver_fill;
typedef rocblas_diagonal rocsolver_diagonal;
typedef rocblas_side rocsolver_side;
typedef rocblas_status rocsolver_status;
typedef rocblas_layer_mode rocsolver_layer_mode;

//extras types only used in rocsolver
typedef rocblas_direct rocsolver_direction;
typedef rocblas_storev rocsolver_storev;

//auxiliaries
#ifdef __cplusplus
extern "C" {
#endif

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_create_handle(rocsolver_handle *handle) {

  const rocblas_status stat = rocblas_create_handle(handle);
  if (stat != rocblas_status_success) {
    return stat;
  }

  return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_destroy_handle(rocsolver_handle handle) {
  return rocblas_destroy_handle(handle);
}

//ROCSOLVER_EXPORT __inline rocsolver_status
//rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
//  return rocblas_add_stream(handle, stream);
//}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream) {
  return rocblas_set_stream(handle, stream);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_stream(rocsolver_handle handle, hipStream_t *stream) {
  return rocblas_get_stream(handle, stream);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb) {
  return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb) {
  return rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

#ifdef __cplusplus
}
#endif

#endif
