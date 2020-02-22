/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrf_strided_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda, const rocblas_int strideA,
                                        rocblas_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) {
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    

    if (m < 0 || n < 0 || batch_count  < 0 || lda < m) 
        return rocblas_status_invalid_size;
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;

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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_strided_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 float *A, const rocsolver_int lda, const rocblas_int strideA, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_strided_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 double *A, const rocsolver_int lda, const rocblas_int strideA, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_strided_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 rocblas_float_complex *A, const rocsolver_int lda, const rocblas_int strideA, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_strided_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_strided_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 rocblas_double_complex *A, const rocsolver_int lda, const rocblas_int strideA, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_strided_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

} //extern C
