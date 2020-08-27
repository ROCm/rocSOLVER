/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#define batched
#include "roclapack_getri.hpp"

/*
 * ===========================================================================
 *    getri_outofplace_batched is not intended for inclusion in the public API.
 *    It exists to provide a getri_batched method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipBLAS.
 * ===========================================================================
 */

template <typename T, typename U, typename V>
__global__ void copy_batch(const rocblas_int m, const rocblas_int n,
                           U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                           V W, const rocblas_int shiftw, const rocblas_int ldw, const rocblas_stride stridew)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    T* a = load_ptr_batch<T>(A,b,shifta,stridea);
    T* w = load_ptr_batch<T>(W,b,shiftw,stridew);

    if (i < m && j < n)
        w[i + j*ldw] = a[i + j*lda];
}

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_batched_impl(rocblas_handle handle, const rocblas_int n, U A,
                                        const rocblas_int lda, rocblas_int* ipiv, const rocblas_stride strideP,
                                        U C, const rocblas_int ldc, rocblas_int *info, const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_getri_argCheck(n,lda,ldc,A,C,ipiv,info,batch_count);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideC = 0;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // copy A into C for out-of-place inversion
    rocblas_int blocks = (n - 1)/32 + 1;
    hipLaunchKernelGGL(copy_batch<T>, dim3(batch_count,blocks,blocks), dim3(1,32,32), 0, stream,
                       n, n, A, 0, lda, 0, C, 0, ldc, 0);

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    rocsolver_getri_getMemorySize<true,T>(n,batch_count,&size_1,&size_2,&size_3);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
           rocsolver_getri_template<true,false,T>(handle,n,
                                                  C,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                                  ldc, strideC,
                                                  ipiv,0, //the vector is shifted 0 entries (will work on the entire vector)
                                                  strideP,
                                                  info,
                                                  batch_count,
                                                  (T*)scalars,
                                                  (T*)work,
                                                  (T**)workArr);

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle, const rocblas_int n, float *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_stride strideP, float *const C[], const rocblas_int ldc,
                 rocblas_int *info, const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<float>(handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle, const rocblas_int n, double *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_stride strideP, double *const C[], const rocblas_int ldc,
                 rocblas_int *info, const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<double>(handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle, const rocblas_int n, rocblas_float_complex *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_stride strideP, rocblas_float_complex *const C[], const rocblas_int ldc,
                 rocblas_int *info, const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle, const rocblas_int n, rocblas_double_complex *const A[],
                 const rocblas_int lda, rocblas_int* ipiv, const rocblas_stride strideP, rocblas_double_complex *const C[], const rocblas_int ldc,
                 rocblas_int *info, const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

} //extern C

#undef batched
