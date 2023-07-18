/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
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
                                    const rocblas_int incp)
{
    ROCSOLVER_ENTER_TOP("laswp", "-n", n, "--lda", lda, "--k1", k1, "--k2", k2);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_laswp_argCheck(handle, n, lda, k1, k2, A, ipiv, incp);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // normal (non-batched non-strided) execution
    rocblas_int inca = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_laswp_template<T>(handle, n, A, shiftA, inca, lda, strideA, k1, k2, ipiv,
                                       shiftP, incp, strideP, batch_count);
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
                                const rocblas_int incp)
{
    return rocsolver_laswp_impl<float>(handle, n, A, lda, k1, k2, ipiv, incp);
}

rocblas_status rocsolver_dlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver_laswp_impl<double>(handle, n, A, lda, k1, k2, ipiv, incp);
}

rocblas_status rocsolver_claswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver_laswp_impl<rocblas_float_complex>(handle, n, A, lda, k1, k2, ipiv, incp);
}

rocblas_status rocsolver_zlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver_laswp_impl<rocblas_double_complex>(handle, n, A, lda, k1, k2, ipiv, incp);
}

} // extern C
