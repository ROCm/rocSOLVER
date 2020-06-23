/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_bdsqr.hpp"

template <typename T, typename W1, typename W2>
rocblas_status rocsolver_bdsqr_impl(rocblas_handle handle,
                                     const rocblas_fill uplo,
                                     const rocblas_int n, 
                                     const rocblas_int nv, 
                                     const rocblas_int nu, 
                                     const rocblas_int nc,
                                     W1*   D,
                                     W1*   E, 
                                     W2    V,
                                     const rocblas_int ldv,
                                     W2    U,
                                     const rocblas_int ldu,
                                     W2    C,
                                     const rocblas_int ldc,
                                     rocblas_int *info)
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_bdsqr_argCheck(uplo,n,nv,nu,nc,ldv,ldu,ldc,D,E,V,U,C,info);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // memory managment
    size_t size;  //size of workspace
    rocsolver_bdsqr_getMemorySize<W1>(n,nv,nu,nc,batch_count,&size);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *work;
    hipMalloc(&work,size);
    if (size && !work)
        return rocblas_status_memory_error;

    // execution
    rocblas_status status =
           rocsolver_bdsqr_template<T>(handle,uplo,n,nv,nu,nc,
                                         D,strideD,
                                         E,strideE,
                                         V,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                         ldv,strideV,
                                         U,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                         ldu,strideU,
                                         C,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                         ldc,strideC,
                                         info,
                                         batch_count,
                                         (W1*)work);
    hipFree(work);
    
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, 
                                                 const rocblas_int nv, 
                                                 const rocblas_int nu, 
                                                 const rocblas_int nc,
                                                 float* D,
                                                 float* E, 
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* U,
                                                 const rocblas_int ldu,
                                                 float* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int *info)
{
    return rocsolver_bdsqr_impl<float>(handle,uplo,n,nv,nu,nc,D,E,V,ldv,U,ldu,C,ldc,info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 double* D,
                                                 double* E,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* U,
                                                 const rocblas_int ldu,
                                                 double* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int *info)
{
    return rocsolver_bdsqr_impl<double>(handle,uplo,n,nv,nu,nc,D,E,V,ldv,U,ldu,C,ldc,info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_float_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_float_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int *info)
{
    return rocsolver_bdsqr_impl<rocblas_float_complex>(handle,uplo,n,nv,nu,nc,D,E,V,ldv,U,ldu,C,ldc,info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zbdsqr(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int nv,
                                                 const rocblas_int nu,
                                                 const rocblas_int nc,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_double_complex* U,
                                                 const rocblas_int ldu,
                                                 rocblas_double_complex* C,
                                                 const rocblas_int ldc,
                                                 rocblas_int *info)
{
    return rocsolver_bdsqr_impl<rocblas_double_complex>(handle,uplo,n,nv,nu,nc,D,E,V,ldv,U,ldu,C,ldc,info);
}


} //extern C
