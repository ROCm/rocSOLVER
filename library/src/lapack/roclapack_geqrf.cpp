/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_geqrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_geqrf_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        T* ipiv) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;
    rocblas_int stridep = 0;
    rocblas_int batch_count = 1;

    return rocsolver_geqrf_template<T>(handle,m,n,
                                    A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                    lda,strideA,
                                    ipiv,
                                    stridep,
                                    batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, float *ipiv) 
{
    return rocsolver_geqrf_impl<float>(handle, m, n, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, double *ipiv) 
{
    return rocsolver_geqrf_impl<double>(handle, m, n, A, lda, ipiv);
}

} //extern C
