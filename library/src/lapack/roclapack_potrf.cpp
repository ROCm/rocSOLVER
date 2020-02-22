/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_potrf_impl(rocblas_handle handle, const rocblas_fill uplo,    
                                    const rocblas_int n, U A, const rocblas_int lda, rocblas_int* info) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !info)
        return rocblas_status_invalid_pointer;
    if (n < 0 || lda < n)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;
    rocblas_int batch_count = 1;

    return rocsolver_potrf_template<T>(handle,uplo,n,
                                    A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                    lda,strideA,
                                    info,batch_count);
}




/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_spotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
                 float *A, const rocblas_int lda, rocblas_int* info) {
  return rocsolver_potrf_impl<float>(handle, uplo, n, A, lda, info);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
                 double *A, const rocblas_int lda, rocblas_int* info) {
  return rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
}
