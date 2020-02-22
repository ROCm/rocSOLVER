/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

template <typename T>
rocblas_status rocsolver_getrs_impl(rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
                 const rocblas_int nrhs, T *A, const rocblas_int lda,
                 const rocblas_int *ipiv, T *B, const rocblas_int ldb) 
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???    

    if (n < 0 || nrhs < 0 || lda < n || ldb < n) 
        return rocblas_status_invalid_size;

    if (!A || !ipiv || !B)
        return rocblas_status_invalid_pointer;

    rocblas_int strideA = 0;
    rocblas_int strideB = 0;
    rocblas_int strideP = 0;
    rocblas_int batch_count = 1;

    return rocsolver_getrs_template<T>(handle,trans,n,nrhs,
                                        A,0,
                                        lda,strideA,
                                        ipiv,strideP,
                                        B,0,
                                        ldb,strideB,
                                        batch_count);
}



/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_sgetrs(rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
                 const rocblas_int nrhs, float *A, const rocblas_int lda,
                 const rocblas_int *ipiv, float *B, const rocblas_int ldb) 
{
  return rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" ROCSOLVER_EXPORT rocblas_status
rocsolver_dgetrs(rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
                 const rocblas_int nrhs, double *A, const rocblas_int lda,
                 const rocblas_int *ipiv, double *B, const rocblas_int ldb) 
{
  return rocsolver_getrs_impl<double>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" ROCSOLVER_EXPORT rocsolver_status 
rocsolver_cgetrs(
    rocsolver_handle handle, const rocsolver_operation trans, const rocsolver_int n,
    const rocsolver_int nrhs, rocblas_float_complex *A, const rocsolver_int lda,
    const rocsolver_int *ipiv, rocblas_float_complex *B, const rocsolver_int ldb) 
{
  return rocsolver_getrs_impl<rocblas_float_complex>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" ROCSOLVER_EXPORT rocsolver_status 
rocsolver_zgetrs(
    rocsolver_handle handle, const rocsolver_operation trans, const rocsolver_int n,
    const rocsolver_int nrhs, rocblas_double_complex *A, const rocsolver_int lda,
    const rocsolver_int *ipiv, rocblas_double_complex *B, const rocsolver_int ldb)
{
  return rocsolver_getrs_impl<rocblas_double_complex>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

