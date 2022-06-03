/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvj_hegvj.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvj_hegvj_batched_impl(rocblas_handle handle,
                                                  const rocblas_eform itype,
                                                  const rocblas_evect evect,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int lda,
                                                  U B,
                                                  const rocblas_int ldb,
                                                  const S abstol,
                                                  S* residual,
                                                  const rocblas_int max_sweeps,
                                                  rocblas_int* n_sweeps,
                                                  S* W,
                                                  const rocblas_stride strideW,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvj_batched" : "hegvj_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb, "--abstol", abstol, "--max_sweeps", max_sweeps,
                        "--strideW", strideW, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygvj_hegvj_argCheck(handle, itype, evect, uplo, n, A, lda, B, ldb, residual,
                                         max_sweeps, n_sweeps, W, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_sygvj_hegvj_getMemorySize<true, false, T, S>(itype, evect, uplo, n, batch_count,
                                                           &size_scalars, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_workArr);

    // memory workspace allocation
    void *scalars, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    workArr = mem[1];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvj_hegvj_template<true, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, abstol,
        residual, max_sweeps, n_sweeps, W, strideW, info, batch_count, (T*)scalars, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvj_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* const B[],
                                        const rocblas_int ldb,
                                        const float abstol,
                                        float* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        float* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygvj_hegvj_batched_impl<float>(handle, itype, evect, uplo, n, A, lda, B, ldb,
                                                     abstol, residual, max_sweeps, n_sweeps, W,
                                                     strideW, info, batch_count);
}

rocblas_status rocsolver_dsygvj_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* const B[],
                                        const rocblas_int ldb,
                                        const double abstol,
                                        double* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        double* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygvj_hegvj_batched_impl<double>(handle, itype, evect, uplo, n, A, lda, B, ldb,
                                                      abstol, residual, max_sweeps, n_sweeps, W,
                                                      strideW, info, batch_count);
}

rocblas_status rocsolver_chegvj_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* const B[],
                                        const rocblas_int ldb,
                                        const float abstol,
                                        float* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        float* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygvj_hegvj_batched_impl<rocblas_float_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

rocblas_status rocsolver_zhegvj_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* const B[],
                                        const rocblas_int ldb,
                                        const double abstol,
                                        double* residual,
                                        const rocblas_int max_sweeps,
                                        rocblas_int* n_sweeps,
                                        double* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygvj_hegvj_batched_impl<rocblas_double_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

} // extern C
