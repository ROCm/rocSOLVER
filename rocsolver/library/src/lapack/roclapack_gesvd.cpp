/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvd.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_impl(rocblas_handle handle,
                                    const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    W A,
                                    const rocblas_int lda,
                                    TT* S,
                                    T* U,
                                    const rocblas_int ldu,
                                    T* V,
                                    const rocblas_int ldv,
                                    TT* E,
                                    const bool fast_alg,
                                    rocblas_int *info)
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_gesvd_argCheck(left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,info);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory managment
//    size_t size;  //size of workspace
//    rocsolver_gesvd_getMemorySize<S>(n,nv,nu,nc,batch_count,&size);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
//    void *work;
//    hipMalloc(&work,size);
//    if (size && !work)
//        return rocblas_status_memory_error;

    // execution
    rocblas_status status =
           rocsolver_gesvd_template<T>(handle,left_svect,right_svect,m,n,
                                       A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                       lda,strideA,
                                       S,strideS,
                                       U,  
                                       ldu,strideU,
                                       V,
                                       ldv,strideV,
                                       E,strideE,
                                       fast_alg,             
                                       info,
                                       batch_count);
                                       
//    hipFree(work);
    
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                float* E,
                                const bool fast_alg,
                                rocblas_int *info)
{
    return rocsolver_gesvd_impl<float>(handle,left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,fast_alg,info);
}

rocblas_status rocsolver_dgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* S,
                                double* U,
                                const rocblas_int ldu,
                                double* V,
                                const rocblas_int ldv,
                                double* E,
                                const bool fast_alg,
                                rocblas_int *info)
{
    return rocsolver_gesvd_impl<double>(handle,left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,fast_alg,info);
}

rocblas_status rocsolver_cgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* S,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                float* E,
                                const bool fast_alg,
                                rocblas_int *info)
{
    return rocsolver_gesvd_impl<rocblas_float_complex>(handle,left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,fast_alg,info);
}

rocblas_status rocsolver_zgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* S,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                double* E,
                                const bool fast_alg,
                                rocblas_int *info)
{
    return rocsolver_gesvd_impl<rocblas_double_complex>(handle,left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,fast_alg,info);
}


} //extern C
