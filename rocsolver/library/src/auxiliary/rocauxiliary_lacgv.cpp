/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_lacgv.hpp"

template <typename T>
rocblas_status rocsolver_lacgv_impl(rocblas_handle handle, const rocblas_int n, T* x, const rocblas_int incx)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    if (n < 0 || !incx)
        return rocblas_status_invalid_size;
    if (!x)
        return rocblas_status_invalid_pointer;

    rocblas_stride stridex = 0;
    rocblas_int batch_count = 1;

    // execution
    rocblas_status status =
           rocsolver_lacgv_template<T>(handle,
                                       n,
                                       x,0,    //vector shifted 0 entries
                                       incx,
                                       stridex,
                                       batch_count);

    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_clacgv(rocblas_handle handle, const rocblas_int n, rocblas_float_complex* x, const rocblas_int incx)
{
    return rocsolver_lacgv_impl<rocblas_float_complex>(handle, n, x, incx);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlacgv(rocblas_handle handle, const rocblas_int n, rocblas_double_complex* x, const rocblas_int incx)
{
    return rocsolver_lacgv_impl<rocblas_double_complex>(handle, n, x, incx);
}

} //extern C

