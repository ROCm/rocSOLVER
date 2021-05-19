/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_steqr.hpp"

template <typename T, typename S>
rocblas_status rocsolver_steqr_impl(rocblas_handle handle,
                                    const rocblas_evect evect,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    T* C,
                                    const rocblas_int ldc,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("steqr", "--evect", evect, "-n", n, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_steqr_argCheck(handle, evect, n, D, E, C, ldc, info);
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

    // memory workspace sizes:
    // size for lasrt stack/steqr workspace
    size_t size_work_stack;
    rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &size_work_stack);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_stack);

    // memory workspace allocation
    void* work_stack;
    rocblas_device_malloc mem(handle, size_work_stack);
    if(!mem)
        return rocblas_status_memory_error;

    work_stack = mem[0];

    // execution
    return rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                       shiftC, ldc, strideC, info, batch_count, work_stack);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                float* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<float>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_dsteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                double* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<double>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_csteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_float_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<rocblas_float_complex>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_zsteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_double_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_steqr_impl<rocblas_double_complex>(handle, evect, n, D, E, C, ldc, info);
}

} // extern C
