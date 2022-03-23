/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_stein.hpp"

template <typename T, typename S>
rocblas_status rocsolver_stein_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    rocblas_int* nev,
                                    S* W,
                                    rocblas_int* iblock,
                                    rocblas_int* isplit,
                                    T* Z,
                                    const rocblas_int ldz,
                                    rocblas_int* ifail,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stein", "-n", n, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_stein_argCheck(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;
    rocblas_int shiftW = 0;
    rocblas_int shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideIblock = 0;
    rocblas_stride strideIsplit = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideIfail = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lagtf/stein workspace
    size_t size_work, size_iwork;
    rocsolver_stein_getMemorySize<T, S>(n, batch_count, &size_work, &size_iwork);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_iwork);

    // memory workspace allocation
    void *work, *iwork;
    rocblas_device_malloc mem(handle, size_work, size_iwork);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    iwork = mem[1];

    // execution
    return rocsolver_stein_template<T>(handle, n, D, shiftD, strideD, E, shiftE, strideE, nev, W,
                                       shiftW, strideW, iblock, strideIblock, isplit, strideIsplit,
                                       Z, shiftZ, ldz, strideZ, ifail, strideIfail, info,
                                       batch_count, (S*)work, (rocblas_int*)iwork);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sstein(rocblas_handle handle,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                float* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                float* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_stein_impl<float, float>(handle, n, D, E, nev, W, iblock, isplit, Z, ldz,
                                              ifail, info);
}

rocblas_status rocsolver_dstein(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                double* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                double* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_stein_impl<double, double>(handle, n, D, E, nev, W, iblock, isplit, Z, ldz,
                                                ifail, info);
}

rocblas_status rocsolver_cstein(rocblas_handle handle,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                float* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_float_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_stein_impl<rocblas_float_complex, float>(handle, n, D, E, nev, W, iblock,
                                                              isplit, Z, ldz, ifail, info);
}

rocblas_status rocsolver_zstein(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                double* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_double_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_stein_impl<rocblas_double_complex, double>(handle, n, D, E, nev, W, iblock,
                                                                isplit, Z, ldz, ifail, info);
}

} // extern C
