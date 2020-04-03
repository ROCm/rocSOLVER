/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orgbr.hpp"

template <typename T>
rocblas_status rocsolver_orgbr_impl(rocblas_handle handle, const rocsolver_storev storev, 
                                   const rocblas_int m, const rocblas_int n, 
                                   const rocblas_int k, T* A, const rocblas_int lda, T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (m < 0 || n < 0 || k < 0 || lda < m)
        return rocblas_status_invalid_size;
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;

    if (storev == rocsolver_column_wise && (n > m || n < min(m,k)))
        return rocblas_status_invalid_size;
    if (storev == rocsolver_row_wise && (m > n || m < min(n,k)))
        return rocblas_status_invalid_size;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count=1;

    return rocsolver_orgbr_template<false,false,T>(handle,storev,
                                                  m,n,k,
                                                  A,0,    //shifted 0 entries
                                                  lda,
                                                  strideA,
                                                  ipiv,
                                                  strideP,
                                                  batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgbr(rocblas_handle handle,
                                                 const rocsolver_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *A,
                                                 const rocblas_int lda,
                                                 float *ipiv)
{
    return rocsolver_orgbr_impl<float>(handle, storev, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgbr(rocblas_handle handle,
                                                 const rocsolver_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *A,
                                                 const rocblas_int lda,
                                                 double *ipiv)
{
    return rocsolver_orgbr_impl<double>(handle, storev, m, n, k, A, lda, ipiv);
}

} //extern C

