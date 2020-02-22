/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfb.hpp"

template <typename T>
rocblas_status rocsolver_larfb_impl(rocsolver_handle handle, const rocsolver_side side, 
                                    const rocsolver_operation trans, const rocsolver_direct direct, 
                                    const rocsolver_storev storev,
                                    const rocsolver_int m, const rocsolver_int n, 
                                    const rocsolver_int k, T* V, const rocsolver_int ldv, T* F, const rocsolver_int ldf,
                                    T* A, const rocsolver_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (m < 0 || n < 0 || k < 1 || lda < m || ldf < k)
        return rocblas_status_invalid_size;
    if (storev == rocsolver_row_wise) {
        if (ldv < k)
            return rocblas_status_invalid_size;
    } else {    
        if ((side == rocblas_side_left && ldv < m) || (side == rocblas_side_right && ldv < n))
            return rocblas_status_invalid_size;
    }
    if (!V || !A || !F)
        return rocblas_status_invalid_pointer;

    rocblas_int stridev = 0;
    rocblas_int stridea = 0;
    rocblas_int stridef = 0;
    rocblas_int batch_count=1;

    return rocsolver_larfb_template<T>(handle,side,trans,direct,storev, 
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
                                      batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocsolver_status rocsolver_slarfb(rocsolver_handle handle,
                                                 const rocsolver_side side,
                                                 const rocsolver_operation trans,
                                                 const rocsolver_direct direct,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 float *V,
                                                 const rocsolver_int ldv,
                                                 float *T,
                                                 const rocsolver_int ldt,
                                                 float *A,
                                                 const rocsolver_int lda)
{
    return rocsolver_larfb_impl<float>(handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}

ROCSOLVER_EXPORT rocsolver_status rocsolver_dlarfb(rocsolver_handle handle,
                                                 const rocsolver_side side,
                                                 const rocsolver_operation trans,
                                                 const rocsolver_direct direct,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 double *V,
                                                 const rocsolver_int ldv,
                                                 double *T,
                                                 const rocsolver_int ldt,
                                                 double *A,
                                                 const rocsolver_int lda)
{
    return rocsolver_larfb_impl<double>(handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt, A, lda);
}


} //extern C

