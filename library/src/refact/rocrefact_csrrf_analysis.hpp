/* **************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_rfinfo.hpp"
#include "rocsparse.hpp"

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

    rocsparseCall_csrilu0_buffer_size(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                      rfinfo->infoT, &csrilu0_buffer_size);

    if(nrhs > 0)
    {
        T alpha = 1.0;

        rocsparseCall_csrsm_buffer_size(rfinfo->sphandle, rocsparse_operation_none,
                                        rocsparse_operation_none, n, nrhs, nnzT, &alpha,
                                        rfinfo->descrL, valT, ptrT, indT, B, ldb, rfinfo->infoL,
                                        rfinfo->solve_policy, &csrsm_L_buffer_size);

        rocsparseCall_csrsm_buffer_size(rfinfo->sphandle, rocsparse_operation_none,
                                        rocsparse_operation_none, n, nrhs, nnzT, &alpha,
                                        rfinfo->descrU, valT, ptrT, indT, B, ldb, rfinfo->infoU,
                                        rfinfo->solve_policy, &csrsm_U_buffer_size);
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

    // analysis for incomplete factorization
    ROCSPARSE_CHECK(rocsparseCall_csrilu0_analysis(
        rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT, rfinfo->infoT,
        rocsparse_analysis_policy_force, rfinfo->solve_policy, work));

    if(nrhs > 0)
    {
        T alpha = 1.0;

        // analysis for solve with L
        ROCSPARSE_CHECK(rocsparseCall_csrsm_analysis(
            rfinfo->sphandle, rocsparse_operation_none, rocsparse_operation_none, n, nrhs, nnzT,
            &alpha, rfinfo->descrL, valT, ptrT, indT, B, ldb, rfinfo->infoL,
            rfinfo->analysis_policy, rfinfo->solve_policy, work));

        // analysis for solve with U
        ROCSPARSE_CHECK(rocsparseCall_csrsm_analysis(
            rfinfo->sphandle, rocsparse_operation_none, rocsparse_operation_none, n, nrhs, nnzT,
            &alpha, rfinfo->descrU, valT, ptrT, indT, B, ldb, rfinfo->infoU,
            rfinfo->analysis_policy, rfinfo->solve_policy, work));
    }

    return rocblas_status_success;
}
