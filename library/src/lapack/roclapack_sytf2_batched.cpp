/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sytf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sytf2_batched_impl(rocblas_handle handle,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            const rocblas_stride strideP,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("sytf2_batched", "--uplo", uplo, "-n", n, "--lda", lda, "--strideP",
                        strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sytf2_sytrf_argCheck(handle, uplo, n, lda, A, ipiv, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // using unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // this function does not requiere memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_sytf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       info, batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytf2_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytf2_batched_impl<float>(handle, uplo, n, A, lda, ipiv, strideP, info,
                                               batch_count);
}

rocblas_status rocsolver_dsytf2_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytf2_batched_impl<double>(handle, uplo, n, A, lda, ipiv, strideP, info,
                                                batch_count);
}

rocblas_status rocsolver_csytf2_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytf2_batched_impl<rocblas_float_complex>(handle, uplo, n, A, lda, ipiv,
                                                               strideP, info, batch_count);
}

rocblas_status rocsolver_zsytf2_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_sytf2_batched_impl<rocblas_double_complex>(handle, uplo, n, A, lda, ipiv,
                                                                strideP, info, batch_count);
}

} // extern C
