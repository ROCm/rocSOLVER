/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getf2_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        rocblas_int *ipiv, rocblas_int* info) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || lda < 1)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;
    rocblas_int strideP = 0;
    rocblas_int batch_count = 1;

    return rocsolver_getf2_template<T>(handle,m,n,
                                    A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getf2_impl<float>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info ) 
{
    return rocsolver_getf2_impl<double>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *A,
                 const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getf2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *A,
                 const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info ) 
{
    return rocsolver_getf2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info);
}

} //extern C
