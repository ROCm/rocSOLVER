/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrf_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        rocblas_int *ipiv, rocblas_int* info) {
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    

    if (m < 0 || n < 0 || lda < m) 
        return rocblas_status_invalid_size;
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    return rocsolver_getrf_template<T>(handle,m,n,
                                        A,0,    //The matrix is shifted 0 entries (will work on the entire matrix)
                                        lda,strideA,
                                        ipiv,0, //the vector is shifted 0 entries (will work on the entire vector)
                                        strideP,
                                        info,batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 float *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 double *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<double>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_float_complex *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_double_complex *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info);
}

} //extern C
