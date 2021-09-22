/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sytf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sytf2_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* ipiv,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("sytf2", "--uplo", uplo, "-n", n, "--lda", lda);

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

rocblas_status rocsolver_ssytf2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytf2_impl<float>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_dsytf2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytf2_impl<double>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_csytf2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytf2_impl<rocblas_float_complex>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_zsytf2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_sytf2_impl<rocblas_double_complex>(handle, uplo, n, A, lda, ipiv, info);
}

} // extern C
