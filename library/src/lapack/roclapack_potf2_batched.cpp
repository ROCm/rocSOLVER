/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_potf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_potf2_batched_impl(rocblas_handle handle, const rocblas_fill uplo,    
                                            const rocblas_int n, U A, const rocblas_int lda, 
                                            rocblas_int* info, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    if (!A || !info)
        return rocblas_status_invalid_pointer;
    if (n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    rocblas_int strideA = 0;

    return rocsolver_potf2_template<T>(handle,uplo,n,
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
rocsolver_spotf2_batched(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
                 float *const A[], const rocblas_int lda, rocblas_int* info, const rocblas_int batch_count) {
  return rocsolver_potf2_batched_impl<float>(handle, uplo, n, A, lda, info, batch_count);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dpotf2_batched(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
                 double *const A[], const rocblas_int lda, rocblas_int* info, const rocblas_int batch_count) {
  return rocsolver_potf2_batched_impl<double>(handle, uplo, n, A, lda, info, batch_count);
}

#undef batched
