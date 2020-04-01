/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormlq.hpp"

template <typename T>
rocblas_status rocsolver_ormlq_impl(rocblas_handle handle, const rocblas_side side, const rocblas_operation trans, 
                                   const rocblas_int m, const rocblas_int n, 
                                   const rocblas_int k, T* A, const rocblas_int lda, T* ipiv, T *C, const rocblas_int ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (m < 0 || n < 0 ||  k < 0 || ldc < m || lda < k)
        return rocblas_status_invalid_size;
    if (side == rocblas_side_left && k > m)
        return rocblas_status_invalid_size;
    if (side == rocblas_side_right && k > n)
        return rocblas_status_invalid_size;
    if (!A || !ipiv || !C)
        return rocblas_status_invalid_pointer;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count=1;

    return rocsolver_ormlq_template<false,false,T>(handle,side,trans,
                                      m,n,k,
                                      A,0,    //shifted 0 entries
                                      lda,
                                      strideA,
                                      ipiv,
                                      strideP,
                                      C,0,  
                                      ldc,
                                      strideC,
                                      batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sormlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *A,
                                                 const rocblas_int lda,
                                                 float *ipiv,
                                                 float *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_ormlq_impl<float>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dormlq(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *A,
                                                 const rocblas_int lda,
                                                 double *ipiv,
                                                 double *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_ormlq_impl<double>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

} //extern C

