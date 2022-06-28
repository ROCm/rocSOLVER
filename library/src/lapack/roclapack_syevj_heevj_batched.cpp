/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_syevj_heevj.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_batched_impl(rocblas_handle handle,
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
                                                  const rocblas_stride strideW,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "syevj_batched" : "heevj_batched");
    ROCSOLVER_ENTER_TOP(name, "--esort", esort, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--abstol", abstol, "--max_sweeps", max_sweeps, "--strideW", strideW,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevj_heevj_argCheck(
        handle, esort, evect, uplo, n, A, lda, residual, max_sweeps, n_sweeps, W, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size of temporary workspace
    size_t size_Acpy, size_resarr, size_cosines, size_sines, size_tbarr;

    rocsolver_syevj_heevj_getMemorySize<true, T, S>(evect, uplo, n, batch_count, &size_Acpy,
                                                    &size_resarr, &size_cosines, &size_sines,
                                                    &size_tbarr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_Acpy, size_resarr, size_cosines,
                                                      size_sines, size_tbarr);

    // memory workspace allocation
    void *Acpy, *resarr, *cosines, *sines, *tbarr;
    rocblas_device_malloc mem(handle, size_Acpy, size_resarr, size_cosines, size_sines, size_tbarr);

    if(!mem)
        return rocblas_status_memory_error;

    Acpy = mem[0];
    resarr = mem[1];
    cosines = mem[2];
    sines = mem[3];
    tbarr = mem[4];

    // execution
    return rocsolver_syevj_heevj_template<true, false, T>(
        handle, esort, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual, max_sweeps,
        n_sweeps, W, strideW, info, batch_count, (T*)Acpy, (S*)resarr, (S*)cosines, (T*)sines,
        (rocblas_int*)tbarr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevj_batched(rocblas_handle handle,
                                        const rocblas_esort esort,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        const float abstol,
                                        float* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        float* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_syevj_heevj_batched_impl<float>(handle, esort, evect, uplo, n, A, lda, abstol,
                                                     residual, max_sweeps, n_sweeps, W, strideW,
                                                     info, batch_count);
}

rocblas_status rocsolver_dsyevj_batched(rocblas_handle handle,
                                        const rocblas_esort esort,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        const double abstol,
                                        double* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        double* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_syevj_heevj_batched_impl<double>(handle, esort, evect, uplo, n, A, lda, abstol,
                                                      residual, max_sweeps, n_sweeps, W, strideW,
                                                      info, batch_count);
}

rocblas_status rocsolver_cheevj_batched(rocblas_handle handle,
                                        const rocblas_esort esort,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        const float abstol,
                                        float* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        float* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_syevj_heevj_batched_impl<rocblas_float_complex>(
        handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, n_sweeps, W, strideW,
        info, batch_count);
}

rocblas_status rocsolver_zheevj_batched(rocblas_handle handle,
                                        const rocblas_esort esort,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        const double abstol,
                                        double* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        double* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_syevj_heevj_batched_impl<rocblas_double_complex>(
        handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, n_sweeps, W, strideW,
        info, batch_count);
}

} // extern C
