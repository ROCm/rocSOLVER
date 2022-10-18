/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvdj.hpp"

template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesvdj_batched_impl(rocblas_handle handle,
                                             const rocblas_svect left_svect,
                                             const rocblas_svect right_svect,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             W A,
                                             const rocblas_int lda,
                                             const SS abstol,
                                             SS* residual,
                                             const rocblas_int max_sweeps,
                                             rocblas_int* n_sweeps,
                                             SS* S,
                                             const rocblas_stride strideS,
                                             T* U,
                                             const rocblas_int ldu,
                                             const rocblas_stride strideU,
                                             T* V,
                                             const rocblas_int ldv,
                                             const rocblas_stride strideV,
                                             rocblas_int* info,
                                             const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gesvdj_batched", "--left_svect", left_svect, "--right_svect", right_svect,
                        "-m", m, "-n", n, "--lda", lda, "--abstol", abstol, "--max_sweeps",
                        max_sweeps, "--strideS", strideS, "--ldu", ldu, "--strideU", strideU,
                        "--ldv", ldv, "--strideV", strideV, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gesvdj_argCheck(handle, left_svect, right_svect, m, n, A, lda, residual,
                                    max_sweeps, n_sweeps, S, U, ldu, V, ldv, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;

    rocsolver_gesvdj_getMemorySize<true, T, SS>(left_svect, right_svect, m, n, batch_count,
                                                &size_scalars);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars);

    // memory workspace allocation
    void* scalars;
    rocblas_device_malloc mem(handle, size_scalars);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesvdj_template<true, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, abstol, residual, max_sweeps,
        n_sweeps, S, strideS, U, ldu, strideU, V, ldv, strideV, info, batch_count, (T*)scalars);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvdj_batched(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         float* const A[],
                                         const rocblas_int lda,
                                         const float abstol,
                                         float* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         float* S,
                                         const rocblas_stride strideS,
                                         float* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         float* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_gesvdj_batched_impl<float>(handle, left_svect, right_svect, m, n, A, lda,
                                                abstol, residual, max_sweeps, n_sweeps, S, strideS,
                                                U, ldu, strideU, V, ldv, strideV, info, batch_count);
}

rocblas_status rocsolver_dgesvdj_batched(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         double* const A[],
                                         const rocblas_int lda,
                                         const double abstol,
                                         double* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         double* S,
                                         const rocblas_stride strideS,
                                         double* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         double* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_gesvdj_batched_impl<double>(handle, left_svect, right_svect, m, n, A, lda,
                                                 abstol, residual, max_sweeps, n_sweeps, S, strideS,
                                                 U, ldu, strideU, V, ldv, strideV, info, batch_count);
}

rocblas_status rocsolver_cgesvdj_batched(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         rocblas_float_complex* const A[],
                                         const rocblas_int lda,
                                         const float abstol,
                                         float* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         float* S,
                                         const rocblas_stride strideS,
                                         rocblas_float_complex* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         rocblas_float_complex* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_gesvdj_batched_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, abstol, residual, max_sweeps, n_sweeps, S,
        strideS, U, ldu, strideU, V, ldv, strideV, info, batch_count);
}

rocblas_status rocsolver_zgesvdj_batched(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         rocblas_double_complex* const A[],
                                         const rocblas_int lda,
                                         const double abstol,
                                         double* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         double* S,
                                         const rocblas_stride strideS,
                                         rocblas_double_complex* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         rocblas_double_complex* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_gesvdj_batched_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, abstol, residual, max_sweeps, n_sweeps, S,
        strideS, U, ldu, strideU, V, ldv, strideV, info, batch_count);
}

} // extern C
