/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_syevd_heevd.hpp"

template <typename T, typename S, typename W>
rocblas_status rocsolver_syevd_heevd_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          W A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          S* D,
                                                          const rocblas_stride strideD,
                                                          S* E,
                                                          const rocblas_stride strideE,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "syevd_strided_batched" : "heevd_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--uplo", uplo, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--strideD", strideD, "--strideE", strideE, "--batch_count",
                        batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_syev_heev_argCheck(handle, evect, uplo, n, A, lda, D, E, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    size_t size_work1;
    size_t size_work2;
    size_t size_work3;
    size_t size_tmptau_W;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syevd_heevd_getMemorySize<false, T, S>(evect, uplo, n, batch_count, &size_scalars,
                                                     &size_work1, &size_work2, &size_work3,
                                                     &size_tmptau_W, &size_tau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_tmptau_W, size_tau,
                                                      size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *tmptau_W, *tau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3,
                              size_tmptau_W, size_tau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    tmptau_W = mem[4];
    tau = mem[5];
    workArr = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevd_heevd_template<false, true, T>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        (T*)scalars, work1, work2, work3, (T*)tmptau_W, (T*)tau, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevd_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_syevd_heevd_strided_batched_impl<float>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_dsyevd_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_syevd_heevd_strided_batched_impl<double>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_cheevd_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_syevd_heevd_strided_batched_impl<rocblas_float_complex>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_zheevd_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_syevd_heevd_strided_batched_impl<rocblas_double_complex>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

} // extern C
