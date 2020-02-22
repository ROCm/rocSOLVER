/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_gelq2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gelq2_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        T* ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;

    return rocsolver_gelq2_template<T>(handle,m,n,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelq2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *const A[],
                 const rocblas_int lda, float *ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_batched_impl<float>(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelq2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *const A[],
                 const rocblas_int lda, double *ipiv, const rocblas_int stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_batched_impl<double>(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

} //extern C
#undef batched
