/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_getf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getf2_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{ 

    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;
    return rocsolver_getf2_template<T>(handle,m,n,
                                            A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                            lda, strideA,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_batched_impl<float>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_batched_impl<double>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetf2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetf2_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getf2_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

} //extern C

#undef batched
