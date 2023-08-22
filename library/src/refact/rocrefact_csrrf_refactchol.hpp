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
//
// NOTE: access ONLY the  LOWER triangular part matrix A and matrix B
//
// Compute B += alpha * (Q * A * Q')
// where sparsity pattern of reordered (Q * A * Q') is a proper subset
// of sparsity pattern of B.
// Further assume for each row, the column indices are
// in increasing sorted order
// -------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void rf_add_QAQ_kernel(const rocblas_int n,
                                        rocblas_int* Qold2new,
                                        const T alpha,
                                        rocblas_int* Ap,
                                        rocblas_int* Ai,
                                        T* Ax,
                                        rocblas_int* Bp,
                                        rocblas_int* Bi,
                                        T* Bx)
{
    // ------------------------------------------------------
    // Note: access to ONLY lower triangular part of matrix A
    // to update ONLY lower triangular part of matrix B
    // ------------------------------------------------------

    rocblas_int const tix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int const tiy = hipThreadIdx_y;

    // ----------------------------------------------
    // each row is processed by one "wave" of threads
    // ----------------------------------------------
    if(tix < n)
    {
        rocblas_int const irow_old = tix;
        rocblas_int const istart_old = Ap[irow_old];
        rocblas_int const iend_old = Ap[irow_old + 1];

        for(rocblas_int i = istart_old + tiy; i < iend_old; i += hipBlockDim_y)
        {
            // ---------------------------------------------------
            // Note: access only lower triangular part of matrix A
            // ---------------------------------------------------

            rocblas_int const jcol_old = Ai[i];
            bool const is_strictly_upper_A = (irow_old < jcol_old);
            if(is_strictly_upper_A)
            {
                break;
            }

            T aij = Ax[i];

            // --------------------------------------------
            // Note: access only lower triangular part of B
            // --------------------------------------------

            rocblas_int irow_new = Qold2new[irow_old];
            rocblas_int jcol_new = Qold2new[jcol_old];
            bool const is_strictly_upper_B = (irow_new < jcol_new);

            if(is_strictly_upper_B)
            {
                // ------------------------------------------
                // take the conj transpose to access ONLY
                // the LOWER triangular part of B
                // ------------------------------------------
                aij = conj(aij);

                // --------------------------
                // swap( irow_new, jcol_new )
                // --------------------------
                rocblas_int const iswap = irow_new;
                irow_new = jcol_new;
                jcol_new = iswap;
            }

            // ----------------------------
            // search for entry B(irow_new, jcol_new)
            // ----------------------------
            rocblas_int const istart_new = Bp[irow_new];
            rocblas_int const iend_new = Bp[irow_new + 1];
            rocblas_int const ipos = rf_search<T>(Bi, istart_new, iend_new, jcol_new);
            bool const is_found = (istart_new <= ipos) && (ipos < iend_new);
            if(is_found)
            {
                Bx[ipos] += alpha * aij;
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
    if(!rfinfo->analyzed || rfinfo->mode != rocsolver_rfinfo_mode_symmetric)
        return rocblas_status_internal_error;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    rocblas_int nblocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_ipvec_kernel<T>, dim3(nblocks), dim3(BS2), 0, stream, n, pivQ,
                            (rocblas_int*)work);

    // --------------------------------------------------
    // set valT[] to zero  before numerical factorization
    // --------------------------------------------------
    if(hipMemsetAsync((void*)valT, 0, sizeof(T) * nnzT, stream) != hipSuccess)
        return rocblas_status_internal_error;

    // --------------------------------------------------------------
    // copy Q'*A*Q into T
    //
    // Note: assume A and T are symmetric and ONLY the LOWER triangular parts of A, and T are touched
    // --------------------------------------------------------------
    T const alpha = static_cast<T>(1);
    ROCSOLVER_LAUNCH_KERNEL(rf_add_QAQ_kernel<T>, dim3(nblocks, 1), dim3(BS2, BS2), 0, stream, n,
                            pivQ, alpha, ptrA, indA, valA, ptrT, indT, valT);

    // perform incomplete factorization of T
    ROCSPARSE_CHECK(rocsparseCall_csric0(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT,
                                         indT, rfinfo->infoT, rocsparse_solve_policy_auto, work));

    return rocblas_status_success;
}
