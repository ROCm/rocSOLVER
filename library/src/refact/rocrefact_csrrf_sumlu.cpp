/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifdef ROCSOLVER_WITH_ROCSPARSE
#include "rocrefact_csrrf_sumlu.hpp"
#else
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#endif

template <typename T, typename U>
rocblas_status rocsolver_csrrf_sumlu_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nnzL,
                                          rocblas_int* ptrL,
                                          rocblas_int* indL,
                                          U valL,
                                          const rocblas_int nnzU,
                                          rocblas_int* ptrU,
                                          rocblas_int* indU,
                                          U valU,
                                          rocblas_int* ptrT,
                                          rocblas_int* indT,
                                          U valT)
{
    ROCSOLVER_ENTER_TOP("csrrf_sumlu", "-n", n, "--nnzL", nnzL, "--nnzU", nnzU);

#ifdef ROCSOLVER_WITH_ROCSPARSE
    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_sumlu_argCheck(handle, n, nnzL, ptrL, indL, valL, nnzU,
                                                       ptrU, indU, valU, ptrT, indT, valT);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // this function does not requiere memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_csrrf_sumlu_template<T>(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                             valU, ptrT, indT, valT);
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

rocblas_status rocsolver_scsrrf_sumlu(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nnzL,
                                      rocblas_int* ptrL,
                                      rocblas_int* indL,
                                      float* valL,
                                      const rocblas_int nnzU,
                                      rocblas_int* ptrU,
                                      rocblas_int* indU,
                                      float* valU,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      float* valT)
{
    return rocsolver_csrrf_sumlu_impl<float>(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                             valU, ptrT, indT, valT);
}

rocblas_status rocsolver_dcsrrf_sumlu(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nnzL,
                                      rocblas_int* ptrL,
                                      rocblas_int* indL,
                                      double* valL,
                                      const rocblas_int nnzU,
                                      rocblas_int* ptrU,
                                      rocblas_int* indU,
                                      double* valU,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      double* valT)
{
    return rocsolver_csrrf_sumlu_impl<double>(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                              valU, ptrT, indT, valT);
}

} // extern C
