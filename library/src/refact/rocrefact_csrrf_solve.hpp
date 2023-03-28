/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_rfinfo.hpp"
#include "rocsparse.hpp"

/**************** Solver Kernels and methods *********************/
// ---------------------
// gather operation
// temp[i,*] = src[P[i],*]
// src <-- temp
// ---------------------
template <typename T>
ROCSOLVER_KERNEL void rf_gather_kernel(const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int* P,
                                       T* src,
                                       const rocblas_int lds,
                                       T* temp)
{
    rocblas_int tid = hipThreadIdx_x;

    // execute permutations
    for(size_t i = tid; i < n; i += hipBlockDim_x)
    {
        const rocblas_int ip = P[i];
        const bool is_valid = (0 <= ip) && (ip < n);
        if(is_valid)
        {
            for(size_t j = 0; j < nrhs; ++j)
                temp[i + j * n] = src[ip + j * lds];
        }
    }
    __syncthreads();

    // overwrite results
    for(size_t i = tid; i < n; i += hipBlockDim_x)
    {
        for(size_t j = 0; j < nrhs; ++j)
            src[i + j * lds] = temp[i + j * n];
    }
}

// ---------------------
// scatter operation
// temp[P[i],*] = src[i,*]
// src <-- temp
// ---------------------
template <typename T>
ROCSOLVER_KERNEL void rf_scatter_kernel(const rocblas_int n,
                                        const rocblas_int nrhs,
                                        const rocblas_int* P,
                                        T* src,
                                        const rocblas_int lds,
                                        T* temp)
{
    rocblas_int tid = hipThreadIdx_x;

    // execute permutations
    for(size_t i = tid; i < n; i += hipBlockDim_x)
    {
        const rocblas_int ip = P[i];
        const bool is_valid = (0 <= ip) && (ip < n);
        if(is_valid)
        {
            for(size_t j = 0; j < nrhs; ++j)
                temp[ip + j * n] = src[i + j * lds];
        }
    }
    __syncthreads();

    // overwrite results
    for(size_t i = tid; i < n; i += hipBlockDim_x)
    {
        for(size_t j = 0; j < nrhs; ++j)
            src[i + j * lds] = temp[i + j * n];
    }
}

// -------------------------------------------
// If A = L*U
// Solve A * X = (L*U) * X = B as
// (1)   solve L * Y = B,   L unit diagonal lower triangular
// (2)   solve U * X = Y,   U non-unit diagonal upper triangular
// -------------------------------------------
template <typename T>
rocblas_status rf_lusolve(rocsolver_rfinfo rfinfo,
                          const rocblas_int n,
                          const rocblas_int nnzLU,
                          const rocblas_int nrhs,
                          rocblas_int* d_LUp,
                          rocblas_int* d_LUi,
                          T* d_LUx,
                          T* B,
                          const rocblas_int ldb,
                          void* buffer)
{
    T alpha = 1.0;

    // ----------------------
    // step (1) solve L * Y = B
    // B <-- Y
    // ----------------------
    ROCSPARSE_CHECK(rocsparseCall_csrsm_solve(
        rfinfo->sphandle, rocsparse_operation_none, rocsparse_operation_none, n, nrhs, nnzLU, &alpha,
        rfinfo->descrL, d_LUx, d_LUp, d_LUi, B, ldb, rfinfo->infoL, rfinfo->solve_policy, buffer));
    // ----------------------
    // step (2) solve U * X = Y
    // B <-- X
    // ----------------------
    ROCSPARSE_CHECK(rocsparseCall_csrsm_solve(
        rfinfo->sphandle, rocsparse_operation_none, rocsparse_operation_none, n, nrhs, nnzLU, &alpha,
        rfinfo->descrU, d_LUx, d_LUp, d_LUi, B, ldb, rfinfo->infoU, rfinfo->solve_policy, buffer));

    return rocblas_status_success;
}
/************************************************************************/

/************** Argument checking and buffer size auxiliaries *************/
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
                                              T B,
                                              const rocblas_int ldb,
                                              rocsolver_rfinfo rfinfo)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    };

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
                                         U B,
                                         const rocblas_int ldb,
                                         rocsolver_rfinfo rfinfo,
                                         size_t* size_work,
                                         size_t* size_temp)
{
    // if quick return, no need of workspace
    if(n == 0 || nrhs == 0)
    {
        *size_work = 0;
        *size_temp = 0;
        return;
    }

    // temp storage for performing permutations
    *size_temp = sizeof(T) * n * nrhs;

    // requirements for solve with L and U
    // (buffer size is the same for all routines if the sparsity pattern does not change)
    size_t csrsm_L_buffer_size = 0;
    size_t csrsm_U_buffer_size = 0;
    T alpha = 1.0;

    rocsparseCall_csrsm_buffer_size(rfinfo->sphandle, rocsparse_operation_none,
                                    rocsparse_operation_none, n, nrhs, nnzT, &alpha, rfinfo->descrL,
                                    valT, ptrT, indT, B, ldb, rfinfo->infoL, rfinfo->solve_policy,
                                    &csrsm_L_buffer_size);

    rocsparseCall_csrsm_buffer_size(rfinfo->sphandle, rocsparse_operation_none,
                                    rocsparse_operation_none, n, nrhs, nnzT, &alpha, rfinfo->descrU,
                                    valT, ptrT, indT, B, ldb, rfinfo->infoU, rfinfo->solve_policy,
                                    &csrsm_U_buffer_size);

    *size_work = std::max(csrsm_L_buffer_size, csrsm_U_buffer_size);
}
/****************************************************************************/

/**************** Template function ****************************/
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
                                              U B,
                                              const rocblas_int ldb,
                                              rocsolver_rfinfo rfinfo,
                                              void* work,
                                              T* temp)
{
    ROCSOLVER_ENTER("csrrf_solve", "n:", n, "nrhs:", nrhs, "nnzT:", nnzT, "ldb:", ldb);

    // quick return
    if(n == 0 || nrhs == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    // -------------------------------------------------------------
    // solve A * X = B
    //   (P * A * Q) * (inv(Q) * X) = P * B
    //
    //   (L * U) * Xhat = Bhat,  Xhat = inv(Q) * X, or Q * Xhat = X,
    //                      Bhat = P * B
    // -------------------------------------------------------------

    // compute Bhat (reordering of B)
    ROCSOLVER_LAUNCH_KERNEL(rf_gather_kernel<T>, dim3(1), dim3(BS1), 0, stream, n, nrhs, pivP, B,
                            ldb, temp);

    // solve (L * U) * Xhat = Bhat
    ROCBLAS_CHECK(rf_lusolve(rfinfo, n, nnzT, nrhs, ptrT, indT, valT, B, ldb, work));

    // Compute X (reordering of Xhat)
    ROCSOLVER_LAUNCH_KERNEL(rf_scatter_kernel<T>, dim3(1), dim3(BS1), 0, stream, n, nrhs, pivQ, B,
                            ldb, temp);

    return rocblas_status_success;
}
/**************************************************************************/
