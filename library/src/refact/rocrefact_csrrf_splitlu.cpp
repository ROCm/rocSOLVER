/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifdef HAVE_ROCSPARSE
#include "rocrefact_csrrf_splitlu.hpp"
#endif

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T, typename U>
rocblas_status rocsolver_csrrf_splitlu_impl(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            U valT,
                                            rocblas_int* ptrL,
                                            rocblas_int* indL,
                                            U valL,
                                            rocblas_int* ptrU,
                                            rocblas_int* indU,
                                            U valU)
{
    ROCSOLVER_ENTER_TOP("csrrf_splitlu", "-n", n, "--nnzT", nnzT);

#ifdef HAVE_ROCSPARSE
    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_splitlu_argCheck(handle, n, nnzT, ptrT, indT, valT, ptrL,
                                                         indL, valL, ptrU, indU, valU);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size to store number of non-zeros per row
    size_t size_work = 0;

    rocsolver_csrrf_splitlu_getMemorySize<T>(n, nnzT, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_splitlu_template<T>(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL,
                                               ptrU, indU, valU, (rocblas_int*)work);
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

rocblas_status rocsolver_scsrrf_splitlu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nnzT,
                                        rocblas_int* ptrT,
                                        rocblas_int* indT,
                                        float* valT,
                                        rocblas_int* ptrL,
                                        rocblas_int* indL,
                                        float* valL,
                                        rocblas_int* ptrU,
                                        rocblas_int* indU,
                                        float* valU)
{
    return rocsolver_csrrf_splitlu_impl<float>(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL,
                                               ptrU, indU, valU);
}

rocblas_status rocsolver_dcsrrf_splitlu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nnzT,
                                        rocblas_int* ptrT,
                                        rocblas_int* indT,
                                        double* valT,
                                        rocblas_int* ptrL,
                                        rocblas_int* indL,
                                        double* valL,
                                        rocblas_int* ptrU,
                                        rocblas_int* indU,
                                        double* valU)
{
    return rocsolver_csrrf_splitlu_impl<double>(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL,
                                                ptrU, indU, valU);
}

} // extern C
