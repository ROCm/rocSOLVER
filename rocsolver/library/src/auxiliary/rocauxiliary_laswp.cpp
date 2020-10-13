/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_laswp.hpp"

template <typename T, typename U>
rocblas_status rocsolver_laswp_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    const rocblas_int k1,
                                    const rocblas_int k2,
                                    const rocblas_int* ipiv,
                                    const rocblas_int incx)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_laswp_argCheck(n, lda, k1, k2, incx, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // this function does not requiere memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_laswp_template<T>(handle, n, A, shiftA, lda, strideA, k1, k2, ipiv, shiftP,
                                       strideP, incx, batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slaswp(rocblas_handle handle,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incx)
{
    return rocsolver_laswp_impl<float>(handle, n, A, lda, k1, k2, ipiv, incx);
}

rocblas_status rocsolver_dlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incx)
{
    return rocsolver_laswp_impl<double>(handle, n, A, lda, k1, k2, ipiv, incx);
}

rocblas_status rocsolver_claswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incx)
{
    return rocsolver_laswp_impl<rocblas_float_complex>(handle, n, A, lda, k1, k2, ipiv, incx);
}

rocblas_status rocsolver_zlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incx)
{
    return rocsolver_laswp_impl<rocblas_double_complex>(handle, n, A, lda, k1, k2, ipiv, incx);
}

} // extern C
