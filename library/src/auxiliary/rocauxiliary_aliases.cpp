/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver-aliases.h"

// We need to include extern definitions for these inline functions to ensure
// that librocsolver.so will contain these symbols for FFI or when inlining
// is disabled.

extern "C" {

// deprecated functions use deprecated types
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

rocsolver_status rocsolver_create_handle(rocsolver_handle* handle)
{
    const rocblas_status stat = rocblas_create_handle(handle);
    if(stat != rocblas_status_success)
    {
        return stat;
    }
    return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

rocsolver_status rocsolver_destroy_handle(rocsolver_handle handle)
{
    return rocblas_destroy_handle(handle);
}

rocsolver_status rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream)
{
    return rocblas_set_stream(handle, stream);
}

rocsolver_status rocsolver_get_stream(rocsolver_handle handle, hipStream_t* stream)
{
    return rocblas_get_stream(handle, stream);
}

rocsolver_status rocsolver_set_vector(rocsolver_int n,
                                      rocsolver_int elem_size,
                                      const void* x,
                                      rocsolver_int incx,
                                      void* y,
                                      rocsolver_int incy)
{
    return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

rocsolver_status rocsolver_get_vector(rocsolver_int n,
                                      rocsolver_int elem_size,
                                      const void* x,
                                      rocsolver_int incx,
                                      void* y,
                                      rocsolver_int incy)
{
    return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

rocsolver_status rocsolver_set_matrix(rocsolver_int rows,
                                      rocsolver_int cols,
                                      rocsolver_int elem_size,
                                      const void* a,
                                      rocsolver_int lda,
                                      void* b,
                                      rocsolver_int ldb)
{
    return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

rocsolver_status rocsolver_get_matrix(rocsolver_int rows,
                                      rocsolver_int cols,
                                      rocsolver_int elem_size,
                                      const void* a,
                                      rocsolver_int lda,
                                      void* b,
                                      rocsolver_int ldb)
{
    return rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

#pragma GCC diagnostic pop // reenable deprecation warnings
}
