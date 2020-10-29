/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sytrd_hetrd.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_sytrd_hetrd_batched_impl(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int lda,
                                                  S* D,
                                                  const rocblas_stride strideD,
                                                  S* E,
                                                  const rocblas_stride strideE,
                                                  T* tau,
                                                  const rocblas_stride strideP,
                                                  const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_sytrd_hetrd_argCheck(uplo, n, lda, A, D, E, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling SYTD2/HETD2
    size_t size_norms, size_work, size_tmptau;
    // size of array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_sytrd_hetrd_getMemorySize<T, true>(n, batch_count, &size_scalars, &size_work,
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
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_sytrd_hetrd_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                          strideE, tau, strideP, batch_count, (T*)scalars, (T*)work,
                                          (T*)norms, (T*)tmptau, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytrd_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* D,
                                        const rocblas_stride strideD,
                                        float* E,
                                        const rocblas_stride strideE,
                                        float* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytrd_hetrd_batched_impl<float, float>(handle, uplo, n, A, lda, D, strideD, E,
                                                            strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_dsytrd_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* D,
                                        const rocblas_stride strideD,
                                        double* E,
                                        const rocblas_stride strideE,
                                        double* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytrd_hetrd_batched_impl<double, double>(handle, uplo, n, A, lda, D, strideD,
                                                              E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_chetrd_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        float* D,
                                        const rocblas_stride strideD,
                                        float* E,
                                        const rocblas_stride strideE,
                                        rocblas_float_complex* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytrd_hetrd_batched_impl<float, rocblas_float_complex>(
        handle, uplo, n, A, lda, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_zhetrd_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        double* D,
                                        const rocblas_stride strideD,
                                        double* E,
                                        const rocblas_stride strideE,
                                        rocblas_double_complex* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytrd_hetrd_batched_impl<double, rocblas_double_complex>(
        handle, uplo, n, A, lda, D, strideD, E, strideE, tau, strideP, batch_count);
}

} // extern C
