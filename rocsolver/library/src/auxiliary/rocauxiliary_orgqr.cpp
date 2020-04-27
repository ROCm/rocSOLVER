/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orgqr.hpp"

template <typename T>
rocblas_status rocsolver_orgqr_impl(rocblas_handle handle, const rocblas_int m, const rocblas_int n, 
                                   const rocblas_int k, T* A, const rocblas_int lda, T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking
    if (m < 0 || n < 0 || n > m || k < 0 || k > n || lda < m)
        return rocblas_status_invalid_size;
    if (!A || !ipiv)
        return rocblas_status_invalid_pointer;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count=1;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    size_t size_4;  // size of temporary array for triangular factor
    rocsolver_orgqr_getMemorySize<T,false>(m,n,k,batch_count,&size_1,&size_2,&size_3,&size_4);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *trfact;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&trfact,size_4);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !trfact))
        return rocblas_status_memory_error;
    
    // execution
    rocblas_status status = 
          rocsolver_orgqr_template<false,false,T>(handle,
                                                  m,n,k,
                                                  A,0,    //shifted 0 entries
                                                  lda,
                                                  strideA,
                                                  ipiv,
                                                  strideP,
                                                  batch_count,
                                                  (T*)scalars,
                                                  (T*)work,
                                                  (T**)workArr,
                                                  (T*)trfact);

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(trfact);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sorgqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *A,
                                                 const rocblas_int lda,
                                                 float *ipiv)
{
    return rocsolver_orgqr_impl<float>(handle, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dorgqr(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *A,
                                                 const rocblas_int lda,
                                                 double *ipiv)
{
    return rocsolver_orgqr_impl<double>(handle, m, n, k, A, lda, ipiv);
}

} //extern C

