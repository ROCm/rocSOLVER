/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_rfinfo.hpp"
#include "rocsparse.hpp"

#include "refact_helpers.hpp"

// -------------------------------------------
// Compute B = beta * B + alpha * (Q * A * Q') as
// (1) B = beta * B
// (2) B += alpha * (Q * A * Q')
// where sparsity pattern of reordered A is a proper subset
// of sparsity pattern of B.
// Further assume for each row, the column indices are
// in increasing sorted order
// -------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void rf_add_QAQ_kernel(const rocblas_int n,
                                        rocblas_int* pivQ,
                                        rocblas_int* inv_pivQ,
                                        const T alpha,
                                        rocblas_int* Ap,
                                        rocblas_int* Ai,
                                        T* Ax,
                                        const T beta,
                                        rocblas_int* LUp,
                                        rocblas_int* LUi,
                                        T* LUx)
{
    rocblas_int tix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;

    // -------------------------------------------
    // If Q is NULL, then treat as identity permutation
    // -------------------------------------------
    if(tix < n)
    {
        rocblas_int irow = tix;
        rocblas_int istart = LUp[irow];
        rocblas_int iend = LUp[irow + 1];
        rocblas_int i, icol;

        rocblas_int irow_old = (pivQ ? pivQ[irow] : irow);
        rocblas_int istart_old = Ap[irow_old];
        rocblas_int iend_old = Ap[irow_old + 1];
        rocblas_int i_old, icol_old;

        T aij;

        // ----------------
        // scale B by beta
        // ----------------
        for(i = istart + tiy; i < iend; i += hipBlockDim_y)
        {
            // only access lower triangle of B
            if(irow < LUi[i])
                break;
            LUx[i] *= beta;
        }
        __syncthreads();

        // ------------------------------
        // scale A by alpha and add to B
        // ------------------------------
        for(i_old = istart_old + tiy; i_old < iend_old; i_old += hipBlockDim_y)
        {
            icol_old = Ai[i_old];
            icol = (inv_pivQ ? inv_pivQ[icol_old] : icol_old);

            // only access lower triangle of A
            if(irow_old < icol_old)
                break;

            // upper part of QAQ' is conjugate transpose of lower part
            if(irow < icol)
            {
                aij = conj(Ax[i_old]);
                istart = LUp[icol];
                iend = LUp[icol + 1];
                icol = irow;
            }
            else
            {
                aij = Ax[i_old];
                istart = LUp[irow];
                iend = LUp[irow + 1];
            }

            i = rf_search<T>(LUi, istart, iend, icol);
            if(i != -1)
            {
                LUx[i] += alpha * aij;
            }
        }
    }
}

template <typename T>
rocblas_status rocsolver_csrrf_refactchol_argCheck(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   const rocblas_int nnzA,
                                                   rocblas_int* ptrA,
                                                   rocblas_int* indA,
                                                   T valA,
                                                   const rocblas_int nnzT,
                                                   rocblas_int* ptrT,
                                                   rocblas_int* indT,
                                                   T valT,
                                                   rocblas_int* pivQ,
                                                   rocsolver_rfinfo rfinfo)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nnzA < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(rfinfo == nullptr)
        return rocblas_status_invalid_pointer;
    if(!rfinfo || !ptrA || !ptrT || (n && !pivQ) || (nnzA && (!indA || !valA))
       || (nnzT && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_refactchol_getMemorySize(const rocblas_int n,
                                              const rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              U valT,
                                              rocsolver_rfinfo rfinfo,
                                              size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0)
    {
        *size_work = 0;
        return;
    }

    // requirements for incomplete factorization
    rocsparseCall_csric0_buffer_size(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                     rfinfo->infoT, size_work);

    // need at least size n integers to generate inverse permutation inv_pivQ
    *size_work = std::max(*size_work, sizeof(rocblas_int) * n);
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactchol_template(rocblas_handle handle,
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
                                                   rocsolver_rfinfo rfinfo,
                                                   void* work)
{
    ROCSOLVER_ENTER("csrrf_refactchol", "n:", n, "nnzA:", nnzA, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // check state of rfinfo
    if(!rfinfo->analyzed || rfinfo->mode != rocsolver_rfinfo_mode_cholesky)
        return rocblas_status_internal_error;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    rocblas_int nblocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_ipvec_kernel<T>, dim3(nblocks), dim3(BS2), 0, stream, n, pivQ,
                            (rocblas_int*)work);

    // --------------------------------------------------------------
    // copy Q'*A*Q into T
    //
    // Note: assume A and B are symmetric and ONLY the LOWER triangular parts of A and T are touched
    // --------------------------------------------------------------
    T const alpha = static_cast<T>(1);
    T const beta = static_cast<T>(0);
    ROCSOLVER_LAUNCH_KERNEL(rf_add_QAQ_kernel<T>, dim3(nblocks, 1), dim3(BS2, BS2), 0, stream, n,
                            pivQ, (rocblas_int*)work, alpha, ptrA, indA, valA, beta, ptrT, indT,
                            valT);

    // perform incomplete factorization of T
    ROCSPARSE_CHECK(rocsparseCall_csric0(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT,
                                         indT, rfinfo->infoT, rocsparse_solve_policy_auto, work));

    return rocblas_status_success;
}
