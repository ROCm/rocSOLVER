/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_steqr.hpp"

template <typename S, typename T>
rocblas_status rocsolver_steqr_impl(rocblas_handle handle,
                                    const rocblas_evect compc,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    T* C,
                                    const rocblas_int ldc,
                                    rocblas_int* info)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_steqr_argCheck(compc, n, D, E, C, ldc, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_steqr_template<S, T>(handle, compc, n, D, shiftD, strideD, E, shiftE, strideE,
                                          C, shiftC, ldc, strideC, info, batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssteqr(rocblas_handle handle,
                                const rocblas_evect compc,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                float* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<float, float>(handle, compc, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_dsteqr(rocblas_handle handle,
                                const rocblas_evect compc,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                double* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<double, double>(handle, compc, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_csteqr(rocblas_handle handle,
                                const rocblas_evect compc,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_float_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<float, rocblas_float_complex>(handle, compc, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_zsteqr(rocblas_handle handle,
                                const rocblas_evect compc,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_double_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<double, rocblas_double_complex>(handle, compc, n, D, E, C, ldc, info);
}

} // extern C
