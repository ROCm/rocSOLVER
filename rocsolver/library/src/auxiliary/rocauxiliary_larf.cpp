/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larf.hpp"

template <typename T>
rocblas_status rocsolver_larf_impl(rocblas_handle handle, const rocblas_side side, const rocblas_int m, 
                                   const rocblas_int n, T* x, const rocblas_int incx, const T* alpha,
                                   T* A, const rocblas_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (n < 0 || m < 0 || lda < m || !incx)
        return rocblas_status_invalid_size;
    if (!x || !A || !alpha)
        return rocblas_status_invalid_pointer;

    rocblas_stride stridex = 0;
    rocblas_stride stridea = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count=1;

    return rocsolver_larf_template<T>(handle,side, 
                                      m,n,
                                      x,0,    //vector shifted 0 entries
                                      incx,
                                      stridex,
                                      alpha,
                                      stridep,
                                      A,0,       //matrix shifted 0 entries
                                      lda,
                                      stridea, 
                                      batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slarf(rocblas_handle handle, const rocblas_side side, const rocblas_int m, 
                                                const rocblas_int n, float* x, const rocblas_int incx, const float* alpha,
                                                float* A, const rocblas_int lda)
{
    return rocsolver_larf_impl<float>(handle, side, m, n, x, incx, alpha, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarf(rocblas_handle handle, const rocblas_side side, const rocblas_int m, 
                                                const rocblas_int n, double* x, const rocblas_int incx, const double* alpha,
                                                double* A, const rocblas_int lda)
{
    return rocsolver_larf_impl<double>(handle, side, m, n, x, incx, alpha, A, lda);
}

} //extern C

