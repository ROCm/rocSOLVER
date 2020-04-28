/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfb.hpp"

template <typename T>
rocblas_status rocsolver_larfb_impl(rocblas_handle handle, const rocblas_side side, 
                                    const rocblas_operation trans, const rocblas_direct direct, 
                                    const rocblas_storev storev,
                                    const rocblas_int m, const rocblas_int n, 
                                    const rocblas_int k, T* V, const rocblas_int ldv, T* F, const rocblas_int ldf,
                                    T* A, const rocblas_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    // argument checking
    if (m < 0 || n < 0 || k < 1 || lda < m || ldf < k)
        return rocblas_status_invalid_size;
    if (storev == rocblas_row_wise) {
        if (ldv < k)
            return rocblas_status_invalid_size;
    } else {    
        if ((side == rocblas_side_left && ldv < m) || (side == rocblas_side_right && ldv < n))
            return rocblas_status_invalid_size;
    }
    if (!V || !A || !F)
        return rocblas_status_invalid_pointer;

    rocblas_stride stridev = 0;
    rocblas_stride stridea = 0;
    rocblas_stride stridef = 0;
    rocblas_int batch_count=1;

    // memory managment
    size_t size_1;  //size of workspace
    size_t size_2;  //size of array of pointers to workspace
    rocsolver_larfb_getMemorySize<T,false>(side,m,n,k,batch_count,&size_1,&size_2);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *work, *workArr;
    hipMalloc(&work,size_1);
    hipMalloc(&workArr,size_2);
    if ((size_1 && !work) || (size_2 && !workArr))
        return rocblas_status_memory_error;

    //  execution
    rocblas_status status = 
            rocsolver_larfb_template<false,false,T>(handle,side,trans,direct,storev, 
                                                  m,n,k,
                                                  V,0,      //shifted 0 entries
                                                  ldv,
                                                  stridev,
                                                  F,0,      //shifted 0 entries
                                                  ldf,
                                                  stridef,
                                                  A,0,      //shifted 0 entries
                                                  lda,
                                                  stridea, 
                                                  batch_count,
                                                  (T*)work,
                                                  (T**)workArr);

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

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 float *V,
                                                 const rocblas_int ldv,
                                                 float *T,
                                                 const rocblas_int ldt,
                                                 float *A,
                                                 const rocblas_int lda)
{
    return rocsolver_larfb_impl<float>(handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfb(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct direct,
                                                 const rocblas_storev storev,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 double *V,
                                                 const rocblas_int ldv,
                                                 double *T,
                                                 const rocblas_int ldt,
                                                 double *A,
                                                 const rocblas_int lda)
{
    return rocsolver_larfb_impl<double>(handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}


} //extern C

