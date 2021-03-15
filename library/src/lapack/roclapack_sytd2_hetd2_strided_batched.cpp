/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sytd2_hetd2.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_sytd2_hetd2_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          S* D,
                                                          const rocblas_stride strideD,
                                                          S* E,
                                                          const rocblas_stride strideE,
                                                          T* tau,
                                                          const rocblas_stride strideP,
                                                          const rocblas_int batch_count)
{
    const char* name = (!is_complex<T> ? "sytd2_strided_batched" : "hetd2_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--uplo", uplo, "-n", n, "--lda", lda, "--strideA", strideA,
                        "--strideD", strideD, "--strideE", strideE, "--strideP", strideP,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sytd2_hetd2_argCheck(handle, uplo, n, lda, A, D, E, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling LARFG
    size_t size_norms, size_work;
    // size of temporary householder scalars
    size_t size_tmptau;
    // size of array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_sytd2_hetd2_getMemorySize<T, false>(n, batch_count, &size_scalars, &size_work,
                                                  &size_norms, &size_tmptau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_norms,
                                                      size_tmptau, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *norms, *tmptau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_norms, size_tmptau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    norms = mem[2];
    tmptau = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                          strideE, tau, strideP, batch_count, (T*)scalars, (T*)work,
                                          (T*)norms, (T*)tmptau, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                float* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver_sytd2_hetd2_strided_batched_impl<float, float>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_dsytd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                double* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver_sytd2_hetd2_strided_batched_impl<double, double>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_chetd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                rocblas_float_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver_sytd2_hetd2_strided_batched_impl<float, rocblas_float_complex>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_zhetd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                rocblas_double_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver_sytd2_hetd2_strided_batched_impl<double, rocblas_double_complex>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

} // extern C
