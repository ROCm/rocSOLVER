/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orgqr.hpp"

template <typename T>
rocblas_status rocsolver_orgqr_impl(rocsolver_handle handle, const rocsolver_int m, const rocsolver_int n, 
                                   const rocsolver_int k, T* A, const rocsolver_int lda, T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (m < 0 || n < 0 || n > m || k < 0 || k > n || lda < m)
        return rocblas_status_invalid_size;
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;

    rocblas_int strideA = 0;
    rocblas_int strideP = 0;
    rocblas_int batch_count=1;

    return rocsolver_orgqr_template<T>(handle,
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

ROCSOLVER_EXPORT rocsolver_status rocsolver_sorgqr(rocsolver_handle handle,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 float *A,
                                                 const rocsolver_int lda,
                                                 float *ipiv)
{
    return rocsolver_orgqr_impl<float>(handle, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocsolver_status rocsolver_dorgqr(rocsolver_handle handle,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 double *A,
                                                 const rocsolver_int lda,
                                                 double *ipiv)
{
    return rocsolver_orgqr_impl<double>(handle, m, n, k, A, lda, ipiv);
}

} //extern C

