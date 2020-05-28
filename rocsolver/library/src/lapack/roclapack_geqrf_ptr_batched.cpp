/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_geqrf.hpp"

/*
 * ===========================================================================
 *    geqrf_ptr_batched is not intended for inclusion in the public API. It
 *    exists to provide a geqrf_batched method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipBLAS.
 * ===========================================================================
 */

template <typename T>
__global__ void copy_array_to_ptrs(rocblas_stride n, T *const ptrs[], T* array)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    if (i < n)
        ptrs[b][i] = array[i + b*n];
}


template <typename T, typename U>
rocblas_status rocsolver_geqrf_ptr_batched_impl(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int lda,
                                        U tau, const rocblas_int batch_count) 
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_geqr2_geqrf_argCheck(m,n,lda,A,tau,batch_count);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = min(m, n);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;
    size_t size_4;
    size_t size_5;
    size_t size_6 = sizeof(T) * strideP * batch_count;
    rocsolver_geqrf_getMemorySize<T,true>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4,&size_5);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *diag, *trfact, *ipiv;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&diag,size_4);
    hipMalloc(&trfact,size_5);
    hipMalloc(&ipiv,size_6);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !diag) || (size_5 && !trfact) || (size_6 && !ipiv))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
           rocsolver_geqrf_template<true,false,T>(handle,m,n,
                                                  A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                                  lda,strideA,
                                                  (T*)ipiv,
                                                  strideP,
                                                  batch_count,
                                                  (T*)scalars,
                                                  (T*)work,
                                                  (T**)workArr,
                                                  (T*)diag,
                                                  (T*)trfact);
    
    // copy ipiv into tau
    if (size_6 > 0)
    {
        rocblas_int blocks = (strideP - 1)/32 + 1;
        hipLaunchKernelGGL(copy_array_to_ptrs, dim3(blocks,batch_count), dim3(32,1), 0, stream,
                        strideP, tau, (T*)ipiv);
    }

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(diag);
    hipFree(trfact);
    hipFree(ipiv);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_ptr_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *const A[],
                 const rocblas_int lda, float *const ipiv[], const rocblas_int batch_count) 
{
    return rocsolver_geqrf_ptr_batched_impl<float>(handle, m, n, A, lda, ipiv, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_ptr_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *const A[],
                 const rocblas_int lda, double *const ipiv[], const rocblas_int batch_count) 
{
    return rocsolver_geqrf_ptr_batched_impl<double>(handle, m, n, A, lda, ipiv, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqrf_ptr_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_float_complex *const A[],
                 const rocblas_int lda, rocblas_float_complex *const ipiv[], const rocblas_int batch_count) 
{
    return rocsolver_geqrf_ptr_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqrf_ptr_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *const A[],
                 const rocblas_int lda, rocblas_double_complex *const ipiv[], const rocblas_int batch_count) 
{
    return rocsolver_geqrf_ptr_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, batch_count);
}

} //extern C

#undef batched
