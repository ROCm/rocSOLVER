/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

template <typename T>
rocblas_status rocsolver_csrrf_solve_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              const rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              T valT,
                                              rocblas_int* pivP,
                                              rocblas_int* pivQ,
                                              rocsolver_rfinfo rfinfo,
                                              T B,
                                              const rocblas_int ldb)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nrhs < 0 || nnzT < 0 || ldb < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!rfinfo || !ptrT || (n && (!pivP || !pivQ)) || (nnzT && (!indT || !valT)) || (nrhs * n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_solve_getMemorySize(const rocblas_int n,
                                         const rocblas_int nrhs,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         U valT,
                                         rocsolver_rfinfo rfinfo,
                                         const rocblas_int ldb,
                                         size_t* size_work,
                                         size_t* size_temp)
{
    // if quick return, no need of workspace
    if(n == 0)
    {
        *size_work = 0;
        *size_temp = 0;
        return;
    }

    // requirements for csrsv_solve
    rocsparseCall_csrsv_buffer_size(rfinfo->sphandle, rocsparse_operation_none, n, nnzT,
                                    rfinfo->descrT, valT, ptrT, indT, rfinfo->infoT, size_work);

    // temp storage for solution vector
    *size_temp = sizeof(T) * ldb * nrhs;
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_solve_template(rocblas_handle handle,
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
                                              const rocblas_int ldb,
                                              void* work,
                                              T* temp)
{
    ROCSOLVER_ENTER("csrrf_solve", "n:", n, "nrhs:", nrhs, "nnzT:", nnzT, "ldb:", ldb);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    /* TODO:

    // ---- main loop (rocsparse does not support multiple rhs) ----
    // -------------------------------------------------------------
    // For each right-hand-side brhs:
    // solve A * x = brhs
    //   P A Q * (inv(Q) x) = P brhs
    //
    //   (LU) xhat = bhat,  xhat = inv(Q) x, or Q xhat = x,
    //                      bhat = P brhs
    // -------------------------------------------------------------
    for(int irhs = 0; irhs < nrhs; irhs++)
    {
        rf_pqrlusolve(handle, n, nnzT, pivP, pivQ, Rs, ptrT, indT, valT,
                        B + ldb * irhs, temp + ldb * irhs);
    }

    */

    return rocblas_status_success;
}
