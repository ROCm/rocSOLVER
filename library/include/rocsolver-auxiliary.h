/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCSOLVER_AUXILIARY_H_
#define _ROCSOLVER_AUXILIARY_H_

#include "rocsolver-types.h"
#include <hip/hip_runtime_api.h>
#include <rocblas.h>

/*!\file
 * \brief rocsolver-auxiliary.h provides auxilary functions in rocsolver
 */

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsolver_handle is a structure holding the rocsolver library context.
 * It must be initialized using rocsolver_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsolver_destroy_handle().
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_create_handle(rocsolver_handle *handle) {

  const rocblas_status stat = rocblas_create_handle(handle);
  if (stat != rocblas_status_success) {
    return stat;
  }

  return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_destroy_handle(rocsolver_handle handle) {
  return rocblas_destroy_handle(handle);
}

/********************************************************************************
 * \brief add stream to handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
  return rocblas_add_stream(handle, stream);
}

/********************************************************************************
 * \brief remove any streams from handle, and add one
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream) {
  return rocblas_set_stream(handle, stream);
}

/********************************************************************************
 * \brief get stream [0] from handle
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_stream(rocsolver_handle handle, hipStream_t *stream) {
  return rocblas_get_stream(handle, stream);
}

/********************************************************************************
 * \brief copy vector from host to device
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

/********************************************************************************
 * \brief copy vector from device to host
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_get_vector(rocsolver_int n, rocsolver_int elem_size, const void *x,
                     rocsolver_int incx, void *y, rocsolver_int incy) {
  return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

/********************************************************************************
 * \brief copy matrix from host to device
 *******************************************************************************/
ROCSOLVER_EXPORT __inline rocsolver_status
rocsolver_set_matrix(rocsolver_int rows, rocsolver_int cols,
                     rocsolver_int elem_size, const void *a, rocsolver_int lda,
                     void *b, rocsolver_int ldb) {
  return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

/********************************************************************************
 * \brief copy matrix from device to host
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
