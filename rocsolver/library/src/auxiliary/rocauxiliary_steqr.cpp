/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
    ROCSOLVER_ENTER_TOP("sterf", "--evect", compc, "-n", n, "--ldc", ldc);

    if(!handle)
        ROCSOLVER_RETURN_TOP("steqr", rocblas_status_invalid_handle);

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_steqr_argCheck(handle, compc, n, D, E, C, ldc, info);
    if(st != rocblas_status_continue)
        ROCSOLVER_RETURN_TOP("steqr", st);

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lasrt stack/steqr workspace
    size_t size_work_stack;
    rocsolver_steqr_getMemorySize<S, T>(compc, n, batch_count, &size_work_stack);

    if(rocblas_is_device_memory_size_query(handle))
        ROCSOLVER_RETURN_TOP("steqr",
                             rocblas_set_optimal_device_memory_size(handle, size_work_stack));

    // memory workspace allocation
    void* work_stack;
    rocblas_device_malloc mem(handle, size_work_stack);
    if(!mem)
        ROCSOLVER_RETURN_TOP("steqr", rocblas_status_memory_error);

    work_stack = mem[0];

    // execution
    ROCSOLVER_RETURN_TOP("steqr",
                         rocsolver_steqr_template<S, T>(handle, compc, n, D, shiftD, strideD, E,
                                                        shiftE, strideE, C, shiftC, ldc, strideC,
                                                        info, batch_count, work_stack));
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
