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
                                     W1   D,
                                     W1   E, 
                                     W2   V,
                                     const rocblas_int ldv,
                                     W2   U,
                                     const rocblas_int ldu,
                                     W2   C,
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
/*    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    size_t size_4;  //size of cache for norms
    rocsolver_labrd_getMemorySize<T,false>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *norms;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&norms,size_4);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !norms))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));
*/
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
                                         batch_count);
//                                         (T*)scalars,
//                                         (T*)work,
//                                         (T**)workArr,
//                                         (T*)norms);

/*
    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(norms);
*/
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
