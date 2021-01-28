/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_sterf.hpp"

template <typename T>
rocblas_status
    rocsolver_sterf_impl(rocblas_handle handle, const rocblas_int n, T* D, T* E, rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("sterf", "-n", n);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sterf_argCheck(handle, n, D, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lasrt stack
    size_t size_stack;
    rocsolver_sterf_getMemorySize<T>(n, batch_count, &size_stack);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_stack);

    // memory workspace allocation
    void* stack;
    rocblas_device_malloc mem(handle, size_stack);
    if(!mem)
        return rocblas_status_memory_error;

    stack = mem[0];

    // execution
    return rocsolver_sterf_template<T>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                       batch_count, (rocblas_int*)stack);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status
    rocsolver_ssterf(rocblas_handle handle, const rocblas_int n, float* D, float* E, rocblas_int* info)
{
    return rocsolver_sterf_impl<float>(handle, n, D, E, info);
}

rocblas_status rocsolver_dsterf(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver_sterf_impl<double>(handle, n, D, E, info);
}

} // extern C
