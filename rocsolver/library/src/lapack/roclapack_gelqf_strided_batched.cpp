/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gelqf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gelqf_strided_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda, const rocblas_stride strideA,
                                        T* ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;
    size_t size_4;
    size_t size_5;
    rocsolver_gelqf_getMemorySize<T,false>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4,&size_5);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *diag, *trfact;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&diag,size_4);
    hipMalloc(&trfact,size_5);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !diag) || (size_5 && !trfact))
        return rocblas_status_memory_error;

    // scalars constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3)
    std::vector<T> sca(size_1);
    sca[0] = -1;
    sca[1] = 0;
    sca[2] = 1;
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca.data(), sizeof(T)*size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
           rocsolver_gelqf_template<false,true,T>(handle,m,n,
                                                    A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                                    lda,strideA,
                                                    ipiv,
                                                    stridep,
                                                    batch_count,
                                                    (T*)scalars,
                                                    (T*)work,
                                                    (T**)workArr,
                                                    (T*)diag,
                                                    (T*)trfact);

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(diag);
    hipFree(trfact);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
                 const rocblas_int lda, const rocblas_stride strideA, float *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
                 const rocblas_int lda, const rocblas_stride strideA, double *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *A,
                 const rocblas_int lda, const rocblas_stride strideA, rocblas_float_complex *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgelqf_strided_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *A,
                 const rocblas_int lda, const rocblas_stride strideA, rocblas_double_complex *ipiv, const rocblas_stride stridep, const rocblas_int batch_count) 
{
    return rocsolver_gelqf_strided_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

} //extern C
