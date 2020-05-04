/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LACGV_HPP
#define ROCLAPACK_LACGV_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U, bool COMPLEX = !std::is_floating_point<T>::value>
rocblas_status rocsolver_lacgv_template(rocblas_handle handle, const rocblas_int n, U x, const rocblas_int shiftx,
                                        const rocblas_int incx, const rocblas_stride stridex, const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || !batch_count || !COMPLEX)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device);  

    // conjugate x
    rocblas_int blocks = (n - 1)/1024 + 1;
    hipLaunchKernelGGL(conj_in_place<T>, dim3(1,blocks,batch_count), dim3(1,1024,1), 0, stream,
                       1, n, x, shiftx, incx, stridex);

    rocblas_set_pointer_mode(handle,old_mode);  
    return rocblas_status_success;
}

#endif
