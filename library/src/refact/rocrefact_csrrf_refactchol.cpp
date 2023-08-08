/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifdef HAVE_ROCSPARSE
#include "rocrefact_csrrf_refactchol.hpp"
#endif

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactchol_impl(rocblas_handle handle,
                                               const rocblas_int n,
                                               const rocblas_int nnzA,
                                               rocblas_int* ptrA,
                                               rocblas_int* indA,
                                               U valA,
                                               const rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               U valT,
                                               rocblas_int* pivQ,
                                               rocsolver_rfinfo rfinfo)
{
    bool const use_lu = false;
    if(rfinfo != nullptr)
    {
        rfinfo->use_lu = use_lu;
    };

    rocblas_int* pivP = pivQ;
    return (rocsolver_csrrf_refact_impl<T, U>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT,
                                              valT, pivP, pivQ, rfinfo, use_lu));
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_refactchol(rocblas_handle handle,
                                           const rocblas_int n,
                                           const rocblas_int nnzA,
                                           rocblas_int* ptrA,
                                           rocblas_int* indA,
                                           float* valA,
                                           const rocblas_int nnzT,
                                           rocblas_int* ptrT,
                                           rocblas_int* indT,
                                           float* valT,
                                           rocblas_int* pivQ,
                                           rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_refactchol_impl<float>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                  indT, valT, pivQ, rfinfo);
}

rocblas_status rocsolver_dcsrrf_refactchol(rocblas_handle handle,
                                           const rocblas_int n,
                                           const rocblas_int nnzA,
                                           rocblas_int* ptrA,
                                           rocblas_int* indA,
                                           double* valA,
                                           const rocblas_int nnzT,
                                           rocblas_int* ptrT,
                                           rocblas_int* indT,
                                           double* valT,
                                           rocblas_int* pivQ,
                                           rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_refactchol_impl<double>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, pivQ, rfinfo);
}

} // extern C