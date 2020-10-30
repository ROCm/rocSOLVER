/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_latrd.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_latrd_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int lda,
                                    S* E,
                                    T* tau,
                                    T* W,
                                    const rocblas_int ldw)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_latrd_argCheck(uplo, n, k, lda, ldw, A, E, tau, W);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftW = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARFG
    size_t size_norms;
    rocsolver_latrd_getMemorySize<T, false>(n, k, batch_count, &size_scalars, &size_work_workArr,
                                            &size_norms);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_norms);

    // memory workspace allocation
    void *scalars, *work_workArr, *norms;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_norms);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    norms = mem[2];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_latrd_template(handle, uplo, n, k, A, shiftA, lda, strideA, E,
                                    strideE, tau, strideP, W, shiftW, ldw, strideW, 
                                    batch_count, (T*)scalars, work_workArr, (T*)norms);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slatrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* E,
                                float* tau,
                                float* W,
                                const rocblas_int ldw)
{
    return rocsolver_latrd_impl<float, float>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

rocblas_status rocsolver_dlatrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* E,
                                double* tau,
                                double* W,
                                const rocblas_int ldw)
{
    return rocsolver_latrd_impl<double, double>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

rocblas_status rocsolver_clatrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* E,
                                rocblas_float_complex* tau,
                                rocblas_float_complex* W,
                                const rocblas_int ldw)
{
    return rocsolver_latrd_impl<float, rocblas_float_complex>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

rocblas_status rocsolver_zlatrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* E,
                                rocblas_double_complex* tau,
                                rocblas_double_complex* W,
                                const rocblas_int ldw)
{
    return rocsolver_latrd_impl<double, rocblas_double_complex>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

} // extern C
