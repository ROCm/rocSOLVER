/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_syevj_heevj.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_impl(rocblas_handle handle,
                                          const rocblas_esort esort,
                                          const rocblas_evect evect,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          const S abstol,
                                          S* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          S* W,
                                          rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "syevj" : "heevj");
    ROCSOLVER_ENTER_TOP(name, "--esort", esort, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--abstol", abstol, "--max_sweeps", max_sweeps);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevj_heevj_argCheck(handle, esort, evect, uplo, n, A, lda,
                                                       residual, max_sweeps, n_sweeps, W, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of temporary workspace
    size_t size_Acpy, size_J, size_norms, size_top, size_bottom, size_completed;

    rocsolver_syevj_heevj_getMemorySize<false, T, S>(evect, uplo, n, batch_count, &size_Acpy,
                                                     &size_J, &size_norms, &size_top, &size_bottom,
                                                     &size_completed);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_Acpy, size_J, size_norms,
                                                      size_top, size_bottom, size_completed);

    // memory workspace allocation
    void *Acpy, *J, *norms, *top, *bottom, *completed;
    rocblas_device_malloc mem(handle, size_Acpy, size_J, size_norms, size_top, size_bottom,
                              size_completed, size_norms);

    if(!mem)
        return rocblas_status_memory_error;

    Acpy = mem[0];
    J = mem[1];
    norms = mem[2];
    top = mem[3];
    bottom = mem[4];
    completed = mem[5];

    // execution
    return rocsolver_syevj_heevj_template<false, false, T>(
        handle, esort, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual, max_sweeps,
        n_sweeps, W, strideW, info, batch_count, (T*)Acpy, (T*)J, (S*)norms, (rocblas_int*)top,
        (rocblas_int*)bottom, (rocblas_int*)completed);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevj(rocblas_handle handle,
                                const rocblas_esort esort,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver_syevj_heevj_impl<float>(handle, esort, evect, uplo, n, A, lda, abstol,
                                             residual, max_sweeps, n_sweeps, W, info);
}

rocblas_status rocsolver_dsyevj(rocblas_handle handle,
                                const rocblas_esort esort,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver_syevj_heevj_impl<double>(handle, esort, evect, uplo, n, A, lda, abstol,
                                              residual, max_sweeps, n_sweeps, W, info);
}

rocblas_status rocsolver_cheevj(rocblas_handle handle,
                                const rocblas_esort esort,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver_syevj_heevj_impl<rocblas_float_complex>(
        handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, n_sweeps, W, info);
}

rocblas_status rocsolver_zheevj(rocblas_handle handle,
                                const rocblas_esort esort,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver_syevj_heevj_impl<rocblas_double_complex>(
        handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, n_sweeps, W, info);
}

} // extern C
