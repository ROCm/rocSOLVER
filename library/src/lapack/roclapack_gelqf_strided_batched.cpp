/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gelqf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gelqf_strided_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda, const rocblas_int strideA,
                                        T* ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;


    return rocsolver_gelqf_template<T>(handle,m,n,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, const rocblas_int strideA, float *ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, const rocblas_int strideA, double *ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

} //extern C
