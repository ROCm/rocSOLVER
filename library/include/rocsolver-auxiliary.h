/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCSOLVER_AUXILIARY_H_
#define _ROCSOLVER_AUXILIARY_H_

#include "rocsolver-types.h"
#include <hip/hip_runtime_api.h>
#include <rocblas.h>

/*! \file
    \brief rocsolver-auxiliary.h provides auxilary functions in rocsolver
 ****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create rocSOLVER handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_create_handle(rocsolver_handle *handle) {

  const rocblas_status stat = rocblas_create_handle(handle);
  if (stat != rocblas_status_success) {
    return stat;
  }

  return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

/*! \brief Destroy rocSOLVER handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_destroy_handle(rocsolver_handle handle) {
  return rocblas_destroy_handle(handle);
}

/*! \brief Add stream to handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
  return rocblas_add_stream(handle, stream);
}

/*! \brief Remove any streams from handle, and add one
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream) {
  return rocblas_set_stream(handle, stream);
}

/*! \brief Get stream [0] from handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_stream(rocsolver_handle handle, hipStream_t *stream) {
  return rocblas_get_stream(handle, stream);
}

/*! \brief Copy vector from host to device
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

/*! \brief Copy vector from device to host
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

/*! \brief Copy matrix from host to device
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb) {
  return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

/*! \brief Copy matrix from device to host
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb) {
  return rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

#ifdef __cplusplus
}
#endif

#endif /* _ROCSOLVER_AUXILIARY_H_ */
