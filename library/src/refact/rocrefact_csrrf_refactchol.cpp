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
    ROCSOLVER_ENTER_TOP("csrrf_refactchol", "-n", n, "--nnzA", nnzA, "--nnzT", nnzT);

#ifdef HAVE_ROCSPARSE
    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_refactchol_argCheck(handle, n, nnzA, ptrA, indA, valA, nnzT,
                                                            ptrT, indT, valT, pivQ, rfinfo);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size for temp buffer in refactlu calls
    size_t size_work = 0;

    rocsolver_csrrf_refactchol_getMemorySize<T>(n, nnzT, ptrT, indT, valT, rfinfo, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_refactchol_template<T>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                  indT, valT, pivQ, rfinfo, work);
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
