/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfg.hpp"

template <typename T>
rocblas_status rocsolver_larfg_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    T* alpha,
                                    T* x,
                                    const rocblas_int incx,
                                    T* tau)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_larfg_argCheck(n, incx, alpha, x, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shifta = 0;
    rocblas_int shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_work;
    // size to store the norms
    size_t size_norms;
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work, &size_norms);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_norms);

    // memory workspace allocation
    void *work, *norms;
    rocblas_device_malloc mem(handle, size_work, size_norms);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    norms = mem[1];

    // execution
    return rocsolver_larfg_template<T>(handle, n, alpha, shifta, x, shiftx, incx, stridex, tau,
                                       strideP, batch_count, (T*)work, (T*)norms);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarfg(rocblas_handle handle,
                                const rocblas_int n,
                                float* alpha,
                                float* x,
                                const rocblas_int incx,
                                float* tau)
{
    return rocsolver_larfg_impl<float>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_dlarfg(rocblas_handle handle,
                                const rocblas_int n,
                                double* alpha,
                                double* x,
                                const rocblas_int incx,
                                double* tau)
{
    return rocsolver_larfg_impl<double>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_clarfg(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* alpha,
                                rocblas_float_complex* x,
                                const rocblas_int incx,
                                rocblas_float_complex* tau)
{
    return rocsolver_larfg_impl<rocblas_float_complex>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_zlarfg(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* alpha,
                                rocblas_double_complex* x,
                                const rocblas_int incx,
                                rocblas_double_complex* tau)
{
    return rocsolver_larfg_impl<rocblas_double_complex>(handle, n, alpha, x, incx, tau);
}

} // extern C
