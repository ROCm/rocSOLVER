/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getf2_strided_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda, const rocblas_int strideA,
                                        rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{ 

    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;
        

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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, const rocblas_int strideA, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, const rocblas_int strideA, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *A,
                 const rocblas_int lda, const rocblas_int strideA, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_strided_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *A,
                 const rocblas_int lda, const rocblas_int strideA, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_strided_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

} //extern C
