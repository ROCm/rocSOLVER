/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larft.hpp"

template <typename T>
rocblas_status rocsolver_larft_impl(rocblas_handle handle, const rocblas_direct direct, 
                                   const rocblas_storev storev, const rocblas_int n, 
                                   const rocblas_int k, T* V, const rocblas_int ldv, T* tau,
                                   T* F, const rocblas_int ldf)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking
    if (n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if (ldv < n && storev == rocblas_column_wise)
        return rocblas_status_invalid_size;
    if (ldv < k && storev == rocblas_row_wise)
        return rocblas_status_invalid_size;
    if (!V || !tau || !F)
        return rocblas_status_invalid_pointer;

    rocblas_stride stridev = 0;
    rocblas_stride stridet = 0;
    rocblas_stride stridef = 0;
    rocblas_int batch_count=1;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    rocsolver_larft_getMemorySize<T,false>(k,batch_count,&size_1,&size_2,&size_3);

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
           rocsolver_larft_template<T>(handle,direct,storev,
                                      n,k,
                                      V,0,    //shifted 0 entries
                                      ldv,
                                      stridev,
                                      tau,
                                      stridet,
                                      F,
                                      ldf,
                                      stridef, 
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

ROCSOLVER_EXPORT rocblas_status rocsolver_slarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *V,
                                                 const rocblas_int ldv,
                                                 float *tau,
                                                 float *T,
                                                 const rocblas_int ldt)
{
    return rocsolver_larft_impl<float>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarft(rocblas_handle handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *V,
                                                 const rocblas_int ldv,
                                                 double *tau,
                                                 double *T,
                                                 const rocblas_int ldt)
{
    return rocsolver_larft_impl<double>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

} //extern C

