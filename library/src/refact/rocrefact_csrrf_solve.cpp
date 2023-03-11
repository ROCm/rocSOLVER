/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocrefact_csrrf_solve.hpp"

template <typename T, typename U>
rocblas_status rocsolver_csrrf_solve_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          const rocblas_int nnzT,
                                          rocblas_int* ptrT,
                                          rocblas_int* indT,
                                          U valT,
                                          rocblas_int* pivP,
                                          rocblas_int* pivQ,
                                          rocsolver_rfinfo rfinfo,
                                          U B,
                                          const rocblas_int ldb)
{
    ROCSOLVER_ENTER_TOP("csrrf_solve", "-n", n, "--nrhs", nrhs, "--nnzT", nnzT, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_solve_argCheck(handle, n, nrhs, nnzT, ptrT, indT, valT,
                                                       pivP, pivQ, rfinfo, B, ldb);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add bacthed versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size for temp buffer in solve calls
    size_t size_work, size_temp;

    rocsolver_csrrf_solve_getMemorySize<T>(n, nrhs, nnzT, ptrT, indT, valT, rfinfo, ldb, &size_work,
                                           &size_temp);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_temp);

    // memory workspace allocation
    void *work, *temp;
    rocblas_device_malloc mem(handle, size_work, size_temp);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    temp = mem[1];

    // execution
    return rocsolver_csrrf_solve_template<T>(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ,
                                             rfinfo, B, ldb, work, (T*)temp);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_solve(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nrhs,
                                      const rocblas_int nnzT,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      float* valT,
                                      rocblas_int* pivP,
                                      rocblas_int* pivQ,
                                      rocsolver_rfinfo rfinfo,
                                      float* B,
                                      const rocblas_int ldb)
{
    return rocsolver_csrrf_solve_impl<float>(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ,
                                             rfinfo, B, ldb);
}

rocblas_status rocsolver_dcsrrf_solve(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nrhs,
                                      const rocblas_int nnzT,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      double* valT,
                                      rocblas_int* pivP,
                                      rocblas_int* pivQ,
                                      rocsolver_rfinfo rfinfo,
                                      double* B,
                                      const rocblas_int ldb)
{
    return rocsolver_csrrf_solve_impl<double>(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ,
                                              rfinfo, B, ldb);
}

} // extern C
