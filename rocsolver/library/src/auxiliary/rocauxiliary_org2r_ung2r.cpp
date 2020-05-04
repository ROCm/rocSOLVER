/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_org2r_ung2r.hpp"

template <typename T>
rocblas_status rocsolver_org2r_ung2r_impl(rocblas_handle handle, const rocblas_int m, const rocblas_int n, 
                                   const rocblas_int k, T* A, const rocblas_int lda, T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking    
    rocblas_status st = rocsolver_org2r_orgqr_argCheck(m,n,k,lda,A,ipiv);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count=1;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    rocsolver_org2r_ung2r_getMemorySize<T,false>(m,n,batch_count,&size_1,&size_2,&size_3);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr))
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
          rocsolver_org2r_ung2r_template<T>(handle,
                                            m,n,k,
                                            A,0,    //shifted 0 entries
                                            lda,
                                            strideA,
                                            ipiv,
                                            strideP,
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

ROCSOLVER_EXPORT rocblas_status rocsolver_sorg2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *A,
                                                 const rocblas_int lda,
                                                 float *ipiv)
{
    return rocsolver_org2r_ung2r_impl<float>(handle, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dorg2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *A,
                                                 const rocblas_int lda,
                                                 double *ipiv)
{
    return rocsolver_org2r_ung2r_impl<double>(handle, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cung2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_float_complex *A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex *ipiv)
{
    return rocsolver_org2r_ung2r_impl<rocblas_float_complex>(handle, m, n, k, A, lda, ipiv);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zung2r(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 rocblas_double_complex *A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex *ipiv)
{
    return rocsolver_org2r_ung2r_impl<rocblas_double_complex>(handle, m, n, k, A, lda, ipiv);
}

} //extern C

