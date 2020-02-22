/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larft.hpp"

template <typename T>
rocblas_status rocsolver_larft_impl(rocsolver_handle handle, const rocsolver_direct direct, 
                                   const rocsolver_storev storev, const rocsolver_int n, 
                                   const rocsolver_int k, T* V, const rocsolver_int ldv, T* tau,
                                   T* F, const rocsolver_int ldf)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if (ldv < n && storev == rocsolver_column_wise)
        return rocblas_status_invalid_size;
    if (ldv < k && storev == rocsolver_row_wise)
        return rocblas_status_invalid_size;
    if (!V || !tau || !F)
        return rocblas_status_invalid_pointer;

    rocblas_int stridev = 0;
    rocblas_int stridet = 0;
    rocblas_int stridef = 0;
    rocblas_int batch_count=1;

    return rocsolver_larft_template<T>(handle,direct,storev,
                                      n,k,
                                      V,0,    //shifted 0 entries
                                      ldv,
                                      stridev,
                                      tau,
                                      stridet,
                                      F,
                                      ldf,
                                      stridef, 
                                      batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocsolver_status rocsolver_slarft(rocsolver_handle handle,
                                                 const rocsolver_direct direct,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 float *V,
                                                 const rocsolver_int ldv,
                                                 float *tau,
                                                 float *T,
                                                 const rocsolver_int ldt)
{
    return rocsolver_larft_impl<float>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

ROCSOLVER_EXPORT rocsolver_status rocsolver_dlarft(rocsolver_handle handle,
                                                 const rocsolver_direct direct,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 double *V,
                                                 const rocsolver_int ldv,
                                                 double *tau,
                                                 double *T,
                                                 const rocsolver_int ldt)
{
    return rocsolver_larft_impl<double>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

} //extern C

