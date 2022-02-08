/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sytrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sytrf_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* ipiv,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("sytrf", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sytf2_sytrf_argCheck(handle, uplo, n, lda, A, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // using unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;
    rocsolver_sytrf_getMemorySize<T>(n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_sytrf_template<T>(handle, uplo, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       info, batch_count, (T*)work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytrf_impl<float>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_dsytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytrf_impl<double>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_csytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytrf_impl<rocblas_float_complex>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_zsytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytrf_impl<rocblas_double_complex>(handle, uplo, n, A, lda, ipiv, info);
}

} // extern C
