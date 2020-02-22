/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfg.hpp"

template <typename T>
rocblas_status rocsolver_larfg_impl(rocblas_handle handle, const rocblas_int n, T *alpha, T *x, const rocblas_int incx, T *tau) {
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (n < 0 || incx < 1)
        return rocblas_status_invalid_size;
    if (!x || !alpha || !tau)
        return rocblas_status_invalid_pointer;

    rocblas_int stridex = 0;
    rocblas_int strideP = 0;
    rocblas_int batch_count=1;

    return rocsolver_larfg_template<T>(handle,n,
                                        alpha,0,    //The pivot is the first pointed element
                                        x,0,        //the vector is shifted 0 entries,
                                        incx,
                                        stridex,
                                        tau,
                                        strideP, 
                                        batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfg(rocsolver_handle handle, const rocsolver_int n, float *alpha,
                 float *x, const rocblas_int incx, float *tau)
{
    return rocsolver_larfg_impl<float>(handle, n, alpha, x, incx, tau);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfg(rocsolver_handle handle, const rocsolver_int n, double *alpha,
                 double *x, const rocblas_int incx, double *tau)
{
    return rocsolver_larfg_impl<double>(handle, n, alpha, x, incx, tau);
}

} //extern C

