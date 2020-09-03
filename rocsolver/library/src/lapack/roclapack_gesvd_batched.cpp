/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvd.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_batched_impl(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        const rocblas_stride strideS,             
                                        T* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        T* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        TT* E,
                                        const rocblas_stride strideE,
                                        const bool fast_alg,
                                        rocblas_int *info,
                                        const rocblas_int batch_count)
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_gesvd_argCheck(left_svect,right_svect,m,n,A,lda,S,U,ldu,V,ldv,E,info,batch_count);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;

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

rocblas_status rocsolver_sgesvd_batched(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            float *const A[],
                                            const rocblas_int lda,
                                            float* S,
                                            const rocblas_stride strideS,
                                            float* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            float* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            float* E,
                                            const rocblas_stride strideE,
                                            const bool fast_alg,
                                            rocblas_int *info,
                                            const rocblas_int batch_count)
{
    return rocsolver_gesvd_batched_impl<float>(handle,left_svect,right_svect,m,n,A,lda,S,strideS,U,ldu,strideU,V,ldv,strideV,E,strideE,fast_alg,info,batch_count);
}

rocblas_status rocsolver_dgesvd_batched(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            double *const A[],
                                            const rocblas_int lda,
                                            double* S,
                                            const rocblas_stride strideS,
                                            double* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            double* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            double* E,
                                            const rocblas_stride strideE,
                                            const bool fast_alg,
                                            rocblas_int *info,
                                            const rocblas_int batch_count)
{
    return rocsolver_gesvd_batched_impl<double>(handle,left_svect,right_svect,m,n,A,lda,S,strideS,U,ldu,strideU,V,ldv,strideV,E,strideE,fast_alg,info,batch_count);
}

rocblas_status rocsolver_cgesvd_batched(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_float_complex *const A[],
                                            const rocblas_int lda,
                                            float* S,
                                            const rocblas_stride strideS,
                                            rocblas_float_complex* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            rocblas_float_complex* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            float* E,
                                            const rocblas_stride strideE,
                                            const bool fast_alg,
                                            rocblas_int *info,
                                            const rocblas_int batch_count)
{
    return rocsolver_gesvd_batched_impl<rocblas_float_complex>(handle,left_svect,right_svect,m,n,A,lda,S,strideS,U,ldu,strideU,V,ldv,strideV,E,strideE,fast_alg,info,batch_count);
}

rocblas_status rocsolver_zgesvd_batched(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_double_complex *const A[],
                                            const rocblas_int lda,
                                            double* S,
                                            const rocblas_stride strideS,
                                            rocblas_double_complex* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            rocblas_double_complex* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            double* E,
                                            const rocblas_stride strideE,
                                            const bool fast_alg,
                                            rocblas_int *info,
                                            const rocblas_int batch_count)
{
    return rocsolver_gesvd_batched_impl<rocblas_double_complex>(handle,left_svect,right_svect,m,n,A,lda,S,strideS,U,ldu,strideU,V,ldv,strideV,E,strideE,fast_alg,info,batch_count);
}


} //extern C
