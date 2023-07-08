/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_rfinfo.hpp"
#include "rocsparse.hpp"

#include "rocrefact_csrrf_refact.hpp"

template <typename T>
rocblas_status rocsolver_csrrf_refactlu_argCheck(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 T valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 T valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo)
{
    return (rocsolver_refact_argCheck(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT,
                                      pivP, pivQ, rfinfo));
}

template <typename T, typename U>
void rocsolver_csrrf_refactlu_getMemorySize(const rocblas_int n,
                                            const rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            U valT,
                                            rocsolver_rfinfo rfinfo,
                                            size_t* size_work)
{
    rocsolver_csrrf_refact_getMemorySize<T, U>(n, nnzT, ptrT, indT, valT, rfinfo, size_work);
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactlu_template(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 U valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 U valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo,
                                                 void* work)
{
    bool const use_lu = true;
    rfinfo->use_lu = use_lu;
    return (rocsolver_csrrf_refact_template<T, U>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                  indT, valT, pivP, pivQ, rfinfo, work, use_lu));
}
