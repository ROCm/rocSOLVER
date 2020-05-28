/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gelq2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gelq2_strided_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda, const rocblas_stride strideA,
                                        T* ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_gelq2_gelqf_argCheck(m,n,lda,A,ipiv,batch_count);
    if (st != rocblas_status_continue)
        return st;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    size_t size_4;
    rocsolver_gelq2_getMemorySize<T,false>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *diag;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&diag,size_4);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !diag))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
         rocsolver_gelq2_template<T>(handle,m,n,
                                    A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                    lda,strideA,
                                    ipiv,
                                    stridep,
                                    batch_count,
                                    (T*)scalars,
                                    (T*)work,
                                    (T**)workArr,
                                    (T*)diag);

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(diag);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelq2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, const rocblas_stride strideA, float *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelq2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, const rocblas_stride strideA, double *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelq2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *A,
                 const rocblas_int lda, const rocblas_stride strideA, rocblas_float_complex *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_strided_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelq2_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *A,
                 const rocblas_int lda, const rocblas_stride strideA, rocblas_double_complex *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelq2_strided_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

} //extern C
