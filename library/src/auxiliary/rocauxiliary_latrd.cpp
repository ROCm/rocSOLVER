/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_latrd.hpp"

template <typename T, typename S, typename U>
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
    ROCSOLVER_ENTER_TOP("latrd", "--uplo", uplo, "-n", n, "-k", k, "--lda", lda, "--ldw", ldw);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_latrd_argCheck(handle, uplo, n, k, lda, ldw, A, E, tau, W);
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
    size_t size_workArr;
    // extra requirements for calling LARFG
    size_t size_work, size_norms;
    rocsolver_latrd_getMemorySize<false, T>(n, k, batch_count, &size_scalars, &size_work,
                                            &size_norms, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_norms,
                                                      size_workArr);

    // memory workspace allocation
    void *scalars, *work, *norms, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_norms, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    norms = mem[2];
    workArr = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_latrd_template<T>(handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau,
                                       strideP, W, shiftW, ldw, strideW, batch_count, (T*)scalars,
                                       (T*)work, (T*)norms, (T**)workArr);
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
    return rocsolver_latrd_impl<float>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
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
    return rocsolver_latrd_impl<double>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
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
    return rocsolver_latrd_impl<rocblas_float_complex>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
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
    return rocsolver_latrd_impl<rocblas_double_complex>(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

} // extern C
