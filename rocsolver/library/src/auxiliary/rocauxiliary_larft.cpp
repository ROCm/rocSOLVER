/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larft.hpp"

template <typename T>
rocblas_status rocsolver_larft_impl(rocblas_handle handle,
                                    const rocblas_direct direct,
                                    const rocblas_storev storev,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    T* V,
                                    const rocblas_int ldv,
                                    T* tau,
                                    T* F,
                                    const rocblas_int ldf)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_larft_argCheck(direct, storev, n, k, ldv, ldf, V, tau, F);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftV = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridev = 0;
    rocblas_stride stridet = 0;
    rocblas_stride stridef = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of re-usable workspace
    size_t size_work;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_larft_getMemorySize<T, false>(n, k, batch_count, &size_scalars, &size_work,
                                            &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    workArr = mem[2];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_larft_template<T>(handle, direct, storev, n, k, V, shiftV, ldv, stridev, tau,
                                       stridet, F, ldf, stridef, batch_count, (T*)scalars, (T*)work,
                                       (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* V,
                                const rocblas_int ldv,
                                float* tau,
                                float* T,
                                const rocblas_int ldt)
{
    return rocsolver_larft_impl<float>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

rocblas_status rocsolver_dlarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* V,
                                const rocblas_int ldv,
                                double* tau,
                                double* T,
                                const rocblas_int ldt)
{
    return rocsolver_larft_impl<double>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

rocblas_status rocsolver_clarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_float_complex* tau,
                                rocblas_float_complex* T,
                                const rocblas_int ldt)
{
    return rocsolver_larft_impl<rocblas_float_complex>(handle, direct, storev, n, k, V, ldv, tau, T,
                                                       ldt);
}

rocblas_status rocsolver_zlarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_double_complex* tau,
                                rocblas_double_complex* T,
                                const rocblas_int ldt)
{
    return rocsolver_larft_impl<rocblas_double_complex>(handle, direct, storev, n, k, V, ldv, tau,
                                                        T, ldt);
}

} // extern C
