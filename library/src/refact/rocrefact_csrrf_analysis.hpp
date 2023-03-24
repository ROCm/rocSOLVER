/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "rocsparse_check.h"

template <typename T>
rocblas_status rocsolver_csrrf_analysis_argCheck(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nrhs,
                                                 const rocblas_int nnzM,
                                                 rocblas_int* ptrM,
                                                 rocblas_int* indM,
                                                 T valM,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 T valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 T B,
                                                 const rocblas_int ldb,
                                                 rocsolver_rfinfo rfinfo)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nnzM < 0 || nnzT < 0 || nrhs < 0 || ldb < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(rfinfo == nullptr)
        return rocblas_status_invalid_pointer;
    if(!rfinfo || !ptrM || !ptrT || (n && (!pivP || !pivQ)) || (nnzM && (!indM || !valM))
       || (nnzT && (!indT || !valT)) || (n * nrhs && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_analysis_getMemorySize(const rocblas_int n,
                                            const rocblas_int nrhs,
                                            const rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            U valT,
                                            U B,
                                            const rocblas_int ldb,
                                            rocsolver_rfinfo rfinfo,
                                            size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0)
    {
        *size_work = 0;
        return;
    }

    // requirements for solve with L and U, and for incomplete factorization
    // (buffer size is the same for all routines if the sparsity pattern does not change)
    size_t csrilu0_buffer_size = 0;
    size_t csrsm_L_buffer_size = 0;
    size_t csrsm_U_buffer_size = 0;
    size_t csrsm_buffer_size = 0;

    THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrilu0_buffer_size(rfinfo->sphandle, n, nnzT,
                                                               rfinfo->descrT, valT, ptrT, indT,
                                                               rfinfo->infoT, &csrilu0_buffer_size));

    rocsparse_operation const trans = rocsparse_operation_none;
    T alpha = 1.0;
    rocsparse_solve_policy const solve_policy = rfinfo->solve_policy;

    if(nrhs > 0)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrsm_buffer_size(
            rfinfo->sphandle, trans, trans, n, nrhs, nnzT, &alpha, rfinfo->descrL, valT, ptrT, indT,
            B, ldb, rfinfo->infoL, solve_policy, &csrsm_L_buffer_size));

        THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrsm_buffer_size(
            rfinfo->sphandle, trans, trans, n, nrhs, nnzT, &alpha, rfinfo->descrU, valT, ptrT, indT,
            B, ldb, rfinfo->infoU, solve_policy, &csrsm_U_buffer_size));
    }

    csrsm_buffer_size = std::max(csrsm_L_buffer_size, csrsm_U_buffer_size);

    *size_work = std::max(csrilu0_buffer_size, csrsm_buffer_size);
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_analysis_template(rocblas_handle handle,
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
                                                 rocsolver_rfinfo rfinfo,
                                                 void* work)
{
    ROCSOLVER_ENTER("csrrf_analysis", "n:", n, "nnzM:", nnzM, "nnzT:", nnzT, "nrhs:", nrhs,
                    "ldb:", ldb);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    rocsparse_operation const trans = rocsparse_operation_none;

    rocsparse_solve_policy const solve = rfinfo->solve_policy;
    rocsparse_analysis_policy const analysis = rfinfo->analysis_policy;

    try
    {
        // analysis for incomplete factorization
        THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrilu0_analysis(
            rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT, rfinfo->infoT,
            rocsparse_analysis_policy_force, solve, work));

        if(nrhs > 0)
        {
            // analysis for solve with L
            T alpha = 1.0;
            THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrsm_analysis(
                rfinfo->sphandle, trans, trans, n, nrhs, nnzT, &alpha, rfinfo->descrL, valT, ptrT,
                indT, B, ldb, rfinfo->infoL, analysis, solve, work));

            // analysis for solve with U
            THROW_IF_ROCSPARSE_ERROR(rocsparseCall_csrsm_analysis(
                rfinfo->sphandle, trans, trans, n, nrhs, nnzT, &alpha, rfinfo->descrU, valT, ptrT,
                indT, B, ldb, rfinfo->infoU, analysis, solve, work));
        }
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    };

    return rocblas_status_success;
}
