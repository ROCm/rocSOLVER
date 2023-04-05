/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifdef HAVE_ROCSPARSE
#include "rocrefact_csrrf_analysis.hpp"
#endif

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T, typename U>
rocblas_status rocsolver_csrrf_analysis_impl(rocblas_handle handle,
                                             const rocblas_int n,
                                             const rocblas_int nrhs,
                                             const rocblas_int nnzM,
                                             rocblas_int* ptrM,
                                             rocblas_int* indM,
                                             U valM,
                                             const rocblas_int nnzT,
                                             rocblas_int* ptrT,
                                             rocblas_int* indT,
                                             U valT,
                                             rocblas_int* pivP,
                                             rocblas_int* pivQ,
                                             U B,
                                             const rocblas_int ldb,
                                             rocsolver_rfinfo rfinfo)
{
    ROCSOLVER_ENTER_TOP("csrrf_analysis", "-n", n, "--nnzM", nnzM, "--nnzT", nnzT, "--nrhs", nrhs,
                        "--ldb", ldb);

#ifdef HAVE_ROCSPARSE
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_analysis_argCheck(
        handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb, rfinfo);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size for temp buffer in analysis calls
    size_t size_work = 0;

    rocsolver_csrrf_analysis_getMemorySize<T>(n, nrhs, nnzT, ptrT, indT, valT, B, ldb, rfinfo,
                                              &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_analysis_template<T>(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                indT, valT, pivP, pivQ, B, ldb, rfinfo, work);
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

rocblas_status rocsolver_scsrrf_analysis(rocblas_handle handle,
                                         const rocblas_int n,
                                         const rocblas_int nrhs,
                                         const rocblas_int nnzM,
                                         rocblas_int* ptrM,
                                         rocblas_int* indM,
                                         float* valM,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         float* valT,
                                         rocblas_int* pivP,
                                         rocblas_int* pivQ,
                                         float* B,
                                         const rocblas_int ldb,
                                         rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_analysis_impl<float>(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                indT, valT, pivP, pivQ, B, ldb, rfinfo);
}

rocblas_status rocsolver_dcsrrf_analysis(rocblas_handle handle,
                                         const rocblas_int n,
                                         const rocblas_int nrhs,
                                         const rocblas_int nnzM,
                                         rocblas_int* ptrM,
                                         rocblas_int* indM,
                                         double* valM,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         double* valT,
                                         rocblas_int* pivP,
                                         rocblas_int* pivQ,
                                         double* B,
                                         const rocblas_int ldb,
                                         rocsolver_rfinfo rfinfo)
{
    return rocsolver_csrrf_analysis_impl<double>(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT,
                                                 ptrT, indT, valT, pivP, pivQ, B, ldb, rfinfo);
}

} // extern C
