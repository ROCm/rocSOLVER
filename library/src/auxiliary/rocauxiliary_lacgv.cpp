/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include "rocauxiliary_lacgv.hpp"

template <typename T>
rocblas_status
    rocsolver_lacgv_impl(rocblas_handle handle, const rocblas_int n, T* x, const rocblas_int incx)
{
    ROCSOLVER_ENTER_TOP("lacgv", "-n", n, "--incx", incx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lacgv_argCheck(handle, n, incx, x);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    rocblas_int batch_count = 1;

    // this function does not requiere memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_lacgv_template<T>(handle, n, x, shiftx, incx, stridex, batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_clacgv(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* x,
                                const rocblas_int incx)
{
    return rocsolver_lacgv_impl<rocblas_float_complex>(handle, n, x, incx);
}

rocblas_status rocsolver_zlacgv(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* x,
                                const rocblas_int incx)
{
    return rocsolver_lacgv_impl<rocblas_double_complex>(handle, n, x, incx);
}

} // extern C
