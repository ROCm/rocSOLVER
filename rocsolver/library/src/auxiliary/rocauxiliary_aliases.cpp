/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver-aliases.h"

// We need to include extern definitions for these inline functions to ensure
// that librocsolver.so will contain these symbols for FFI or when inlining
// is disabled.

extern "C" {

extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_create_handle(
        rocsolver_handle *handle);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_destroy_handle(
        rocsolver_handle handle);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_set_stream(
        rocsolver_handle handle,
        hipStream_t stream);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_get_stream(
        rocsolver_handle handle,
        hipStream_t *stream);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_set_vector(
        rocsolver_int n,
        rocsolver_int elem_size,
        const void *x,
        rocsolver_int incx,
        void *y,
        rocsolver_int incy);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_get_vector(
        rocsolver_int n,
        rocsolver_int elem_size,
        const void *x,
        rocsolver_int incx,
        void *y,
        rocsolver_int incy);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_set_matrix(
        rocsolver_int rows,
        rocsolver_int cols,
        rocsolver_int elem_size,
        const void *a,
        rocsolver_int lda,
        void *b,
        rocsolver_int ldb);
extern ROCSOLVER_EXPORT __inline rocsolver_status rocsolver_get_matrix(
        rocsolver_int rows,
        rocsolver_int cols,
        rocsolver_int elem_size,
        const void *a,
        rocsolver_int lda,
        void *b,
        rocsolver_int ldb);

}
