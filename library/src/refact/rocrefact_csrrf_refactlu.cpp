/* ************************************************************************
  * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution
 * ************************************************************************ */

#ifdef HAVE_ROCSPARSE
#include "rocrefact_csrrf_refactlu.hpp"
#endif

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactlu_impl(rocblas_handle handle,
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
                                             rocsolver_rfinfo rfinfo)
{
    ROCSOLVER_ENTER_TOP("csrrf_refactlu", "-n", n, "--nnzA", nnzA, "--nnzT", nnzT);

#ifdef HAVE_ROCSPARSE
    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_refactlu_argCheck(handle, n, nnzA, ptrA, indA, valA, nnzT,
                                                          ptrT, indT, valT, pivP, pivQ, rfinfo);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size for temp buffer in refactlu calls
    size_t size_work = 0;

    rocsolver_csrrf_refactlu_getMemorySize<T>(n, nnzT, ptrT, indT, valT, rfinfo, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_refactlu_template<T>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT,
                                                valT, pivP, pivQ, rfinfo, work);
#else
    return rocblas_status_not_implemented;
#endif
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_refactlu(rocblas_handle handle,
                                         const rocblas_int n,
                                         const rocblas_int nnzA,
                                         rocblas_int* ptrA,
                                         rocblas_int* indA,
                                         float* valA,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         float* valT,
                                         rocblas_int* pivP,
                                         rocblas_int* pivQ,
                                         rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_refactlu_impl<float>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT,
                                                valT, pivP, pivQ, rfinfo);
}

rocblas_status rocsolver_dcsrrf_refactlu(rocblas_handle handle,
                                         const rocblas_int n,
                                         const rocblas_int nnzA,
                                         rocblas_int* ptrA,
                                         rocblas_int* indA,
                                         double* valA,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         double* valT,
                                         rocblas_int* pivP,
                                         rocblas_int* pivQ,
                                         rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_refactlu_impl<double>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                 indT, valT, pivP, pivQ, rfinfo);
}

} // extern C
