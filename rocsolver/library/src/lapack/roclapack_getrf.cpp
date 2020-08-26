/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getrf_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        rocblas_int *ipiv, rocblas_int* info, const int pivot) 
{
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    

    // argument checking
    rocblas_status st = rocsolver_getf2_getrf_argCheck(m,n,lda,A,ipiv,info);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory managment
    using S = decltype(std::real(T{}));
    size_t size_1;  //constants
    size_t size_2;  //pivot values
    size_t size_3;  //pivot indexes
    size_t size_4;  //info values
    size_t size_5;  //workspace
    size_t size_6;  //for TRSM x_temp
    size_t size_7;  //for TRSM x_temp_arr
    size_t size_8;  //for TRSM invA
    size_t size_9;  //for TRSM invA_arr
    rocsolver_getrf_getMemorySize<false,T,S>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4,&size_5,&size_6,&size_7,&size_8,&size_9);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *pivot_val, *pivot_idx, *iinfo, *work, *x_temp, *x_temp_arr, *invA, *invA_arr;
    bool optim_mem = true; //always allocate all required memory for TRSM optimal performance
    
    hipMalloc(&scalars,size_1);
    hipMalloc(&pivot_val,size_2);
    hipMalloc(&pivot_idx,size_3);
    hipMalloc(&iinfo,size_4);
    hipMalloc(&work,size_5);
    hipMalloc(&x_temp,size_6);
    hipMalloc(&x_temp_arr,size_7);
    hipMalloc(&invA,size_8);
    hipMalloc(&invA_arr,size_9);
    if (!scalars || (size_2 && !pivot_val) || (size_3 && !pivot_idx) || (size_4 && !iinfo) || (size_5 && !work) ||
        (size_6 && !x_temp) || (size_7 && !x_temp_arr) || (size_8 && !invA) || (size_9 && !invA_arr))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status = 
           rocsolver_getrf_template<false,false,T,S>(handle,m,n,
                                                    A,0,    //The matrix is shifted 0 entries (will work on the entire matrix)
                                                    lda,strideA,
                                                    ipiv,0, //the vector is shifted 0 entries (will work on the entire vector)
                                                    strideP,
                                                    info,batch_count,pivot,
                                                    (T*)scalars,
                                                    (T*)pivot_val,
                                                    (rocblas_int*)pivot_idx,    
                                                    (rocblas_int*)iinfo,
                                                    (rocblas_index_value_t<S>*)work,
                                                    x_temp,
                                                    x_temp_arr,
                                                    invA,
                                                    invA_arr,
                                                    optim_mem);

    hipFree(scalars);
    hipFree(pivot_val);
    hipFree(pivot_idx);
    hipFree(iinfo);
    hipFree(work);
    hipFree(x_temp);
    hipFree(x_temp_arr);
    hipFree(invA);
    hipFree(invA_arr);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 float *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, 1);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 double *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<double>(handle, m, n, A, lda, ipiv, info, 1);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_float_complex *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info, 1);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_double_complex *A, const rocblas_int lda, rocblas_int *ipiv, rocblas_int* info) 
{
    return rocsolver_getrf_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info, 1);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_npvt(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 float *A, const rocblas_int lda, rocblas_int* info) 
{
    rocblas_int *ipiv;
    return rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, 0);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_npvt(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 double *A, const rocblas_int lda, rocblas_int* info) 
{
    rocblas_int *ipiv;
    return rocsolver_getrf_impl<double>(handle, m, n, A, lda, ipiv, info, 0);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_npvt(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_float_complex *A, const rocblas_int lda, rocblas_int* info) 
{
    rocblas_int *ipiv;
    return rocsolver_getrf_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info, 0);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_npvt(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                 rocblas_double_complex *A, const rocblas_int lda, rocblas_int* info) 
{
    rocblas_int *ipiv;
    return rocsolver_getrf_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info, 0);
}

} //extern C
