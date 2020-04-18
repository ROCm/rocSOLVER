/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_laswp.hpp"

template <typename T, typename U>
rocblas_status rocsolver_laswp_impl(rocblas_handle handle, const rocblas_int n, U A, const rocblas_int lda,
                                    const rocblas_int k1, const rocblas_int k2, const rocblas_int *ipiv, const rocblas_int incx) {
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (n < 0 || lda < 1 || !incx || k1 < 1 || k2 < 1)
        return rocblas_status_invalid_size;
    if (k2 < k1)
        return rocblas_status_invalid_size;
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count=1;

    return rocsolver_laswp_template<T>(handle,n,
                                        A,0,    //The matrix is shifted 0 entries (will work on the entire matrix)
                                        lda,strideA,
                                        k1,k2,
                                        ipiv,0, //the vector is shifted 0 entries (will work on the entire vector)
                                        strideP,
                                        incx,batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slaswp(rocblas_handle handle, const rocblas_int n,
                 float *A, const rocblas_int lda, const rocblas_int k1, const rocblas_int k2, const rocblas_int *ipiv, const rocblas_int incx)
{
    return rocsolver_laswp_impl<float>(handle, n, A, lda, k1, k2, ipiv, incx);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlaswp(rocblas_handle handle, const rocblas_int n,
                 double *A, const rocblas_int lda, const rocblas_int k1, const rocblas_int k2, const rocblas_int *ipiv, const rocblas_int incx)
{
    return rocsolver_laswp_impl<double>(handle, n, A, lda, k1, k2, ipiv, incx);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_claswp(rocblas_handle handle, const rocblas_int n,
                 rocblas_float_complex *A, const rocblas_int lda, const rocblas_int k1, const rocblas_int k2, 
                 const rocblas_int *ipiv, const rocblas_int incx)
{
    return rocsolver_laswp_impl<rocblas_float_complex>(handle, n, A, lda, k1, k2, ipiv, incx);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlaswp(rocblas_handle handle, const rocblas_int n,
                 rocblas_double_complex *A, const rocblas_int lda, const rocblas_int k1, const rocblas_int k2, 
                 const rocblas_int *ipiv, const rocblas_int incx)
{
    return rocsolver_laswp_impl<rocblas_double_complex>(handle, n, A, lda, k1, k2, ipiv, incx);
}



} //extern C

