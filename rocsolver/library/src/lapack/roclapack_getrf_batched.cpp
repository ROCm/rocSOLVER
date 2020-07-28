/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_getrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrf_batched_impl(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, U A, rocblas_int lda, 
                                        rocblas_int *ipiv, const rocblas_stride strideP, rocblas_int* info, rocblas_int batch_count) {
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    

    // argument checking
    rocblas_status st = rocsolver_getf2_getrf_argCheck(m,n,lda,A,ipiv,info,batch_count);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;
    size_t size_3;
    rocsolver_getrf_getMemorySize<T>(m,n,batch_count,&size_1,&size_2,&size_3);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *pivotGPU, *iinfo;
    hipMalloc(&scalars,size_1);
    hipMalloc(&pivotGPU,size_2);
    hipMalloc(&iinfo,size_3);
    if (!scalars || (size_2 && !pivotGPU) || (size_3 && !iinfo))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
           rocsolver_getrf_template<true,false,T>(handle,m,n,
                                                    A,0,    //The matrix is shifted 0 entries (will work on the entire matrix)
                                                    lda,strideA,
                                                    ipiv,0, //the vector is shifted 0 entries (will work on the entire vector)
                                                    strideP,
                                                    info,batch_count,
                                                    (T*)scalars,
                                                    (T*)pivotGPU,
                                                    (rocblas_int*)iinfo);

    hipFree(scalars);
    hipFree(pivotGPU);
    hipFree(iinfo);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 float *const A[], const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count) 
{
    return rocsolver_getrf_batched_impl<float>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 double *const A[], const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count) 
{
//    return rocsolver_getrf_batched_impl<double>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_float_complex *const A[], const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count) 
{
//    return rocsolver_getrf_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_double_complex *const A[], const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count) 
{
//    return rocsolver_getrf_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, strideP, info, batch_count);
}

} //extern C

#undef batched
