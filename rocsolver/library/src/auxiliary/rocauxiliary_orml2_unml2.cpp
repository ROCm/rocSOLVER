/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orml2_unml2.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_orml2_unml2_impl(rocblas_handle handle, const rocblas_side side, const rocblas_operation trans, 
                                          const rocblas_int m, const rocblas_int n, 
                                          const rocblas_int k, T* A, const rocblas_int lda, T* ipiv, T *C, const rocblas_int ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking
    if (m < 0 || n < 0 ||  k < 0 || ldc < m || lda < k)
        return rocblas_status_invalid_size;
    if (side == rocblas_side_left && k > m)
        return rocblas_status_invalid_size;
    if (side == rocblas_side_right && k > n)
        return rocblas_status_invalid_size;
    if (!A || !ipiv || !C)
        return rocblas_status_invalid_pointer;
    if (COMPLEX && trans == rocblas_operation_transpose)
        return rocblas_status_invalid_value;
    if (!COMPLEX && trans == rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count=1;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    size_t size_4;  //size of temporary array for diagonal elemements
    rocsolver_orml2_unml2_getMemorySize<T,false>(side,m,n,batch_count,&size_1,&size_2,&size_3,&size_4);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *diag;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&diag,size_4);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !diag))
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
           rocsolver_orml2_unml2_template<T>(handle,side,trans,
                                             m,n,k,
                                             A,0,    //shifted 0 entries
                                             lda,
                                             strideA,
                                             ipiv,
                                             strideP,
                                             C,0,  
                                             ldc,
                                             strideC,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sorml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *A,
                                                 const rocblas_int lda,
                                                 float *ipiv,
                                                 float *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_orml2_unml2_impl<float>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dorml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *A,
                                                 const rocblas_int lda,
                                                 double *ipiv,
                                                 double *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_orml2_unml2_impl<double>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cunml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex *A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex *ipiv,
                                                 rocblas_float_complex *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_orml2_unml2_impl<rocblas_float_complex>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zunml2(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex *A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex *ipiv,
                                                 rocblas_double_complex *C,
                                                 const rocblas_int ldc)
{
    return rocsolver_orml2_unml2_impl<rocblas_double_complex>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

} //extern C

