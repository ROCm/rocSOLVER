/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_bdsqr.hpp"

template <typename T, typename S, typename W>
rocblas_status rocsolver_bdsqr_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    const rocblas_int nv,
                                    const rocblas_int nu,
                                    const rocblas_int nc,
                                    S* D,
                                    S* E,
                                    W V,
                                    const rocblas_int ldv,
                                    W U,
                                    const rocblas_int ldu,
                                    W C,
                                    const rocblas_int ldc,
                                    rocblas_int* info)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st
        = rocsolver_bdsqr_argCheck(uplo, n, nv, nu, nc, ldv, ldu, ldc, D, E, V, U, C, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftV = 0;
    rocblas_int shiftU = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_work;
    rocsolver_bdsqr_getMemorySize<S>(n, nv, nu, nc, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V,
                                       shiftV, ldv, strideV, U, shiftU, ldu, strideU, C, shiftC,
                                       ldc, strideC, info, batch_count, (S*)work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sbdsqr(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                float* D,
                                float* E,
                                float* V,
                                const rocblas_int ldv,
                                float* U,
                                const rocblas_int ldu,
                                float* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_bdsqr_impl<float>(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc,
                                       info);
}

rocblas_status rocsolver_dbdsqr(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                double* D,
                                double* E,
                                double* V,
                                const rocblas_int ldv,
                                double* U,
                                const rocblas_int ldu,
                                double* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_bdsqr_impl<double>(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc,
                                        info);
}

rocblas_status rocsolver_cbdsqr(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                float* D,
                                float* E,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_bdsqr_impl<rocblas_float_complex>(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U,
                                                       ldu, C, ldc, info);
}

rocblas_status rocsolver_zbdsqr(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                double* D,
                                double* E,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                rocblas_double_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver_bdsqr_impl<rocblas_double_complex>(handle, uplo, n, nv, nu, nc, D, E, V, ldv,
                                                        U, ldu, C, ldc, info);
}

} // extern C
