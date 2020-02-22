/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_getrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrf_batched_impl(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, U A, rocblas_int lda, 
                                        rocblas_int *ipiv, const rocblas_int strideP, rocblas_int* info, rocblas_int batch_count) {
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    

    if (m < 0 || n < 0 || batch_count < 0 || lda < m) 
        return rocblas_status_invalid_size;
    if (!A || !ipiv || !info)
        return rocblas_status_invalid_pointer;

    rocblas_int strideA = 0;

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

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 float *const A[], const rocsolver_int lda, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_batched_impl<float>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 double *const A[], const rocsolver_int lda, rocsolver_int *ipiv, const rocblas_int strideP, rocsolver_int* info, const rocsolver_int batch_count) 
{
    return rocsolver_getrf_batched_impl<double>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 rocblas_float_complex *const A[], const rocsolver_int lda, rocsolver_int *ipiv, const rocblas_int strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_batched(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n,
                 rocblas_double_complex *const A[], const rocsolver_int lda, rocsolver_int *ipiv, const rocblas_int strideP, rocsolver_int* info, const rocsolver_int batch_count) 
{
    return rocsolver_getrf_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

} //extern C

#undef batched
