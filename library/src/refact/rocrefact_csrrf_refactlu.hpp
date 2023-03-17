/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "rocsparse_check.h"

#define ADD_PAQ_MAX_THDS 256

// -------------------------------------------------
// function to perform search in array
// -------------------------------------------------
// search array  arr[0], ..., arr[ len-1]
// for matching value "key"
//
// return the index value of matching position
// ---------------------------------------
static 
__device__ rocblas_int rf_search(const rocblas_int len, const rocblas_int* const arr, const rocblas_int key)
{
    rocblas_int constexpr small_len = 8;
    rocblas_int ipos = len;
    if((len <= 0) || (arr == nullptr))
    {
        return (ipos = len);
    }

    // -----------------
    // use binary search
    // -----------------
    rocblas_int lo = 0;
    rocblas_int hi = len;

    for(int i = 0; i < 32; i++)
    {
        rocblas_int const len_remain = hi - lo;
        if(len_remain <= small_len)
        {
            // ------------------------
            // use simple linear search
            // ------------------------
            for(int k = 0; k < len; k++)
            {
                bool const is_found = (arr[k] == key);
                if(is_found)
                {
                    return (ipos = k);
                }
            }
        }
        else
        {
            rocblas_int const mid = lo + ((hi - lo)/2);
            bool const is_found = (arr[mid] == key);
            if(is_found)
            {
                return (ipos = mid);
            };

            if(arr[mid] < key)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
    }
    return (ipos);
}


// -------------------------------------------
// Compute B = beta * B + alpha * (P * A * Q') as
// (1) B = beta * B
// (2) B += alpha * (P * A * Q')
// where sparsity pattern of reordered A is a proper subset
// of sparsity pattern of B.
// Further assume for each row, the column indices are
// in increasing sorted order
// -------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(ADD_PAQ_MAX_THDS)
    rf_add_PAQ_kernel(const rocblas_int nrow,
                        const rocblas_int ncol,
                        rocblas_int* P_new2old,
                        rocblas_int* Q_old2new,
                        const T alpha,
                        rocblas_int* Ap,
                        rocblas_int* Ai,
                        T* Ax,
                        const T beta,
                        rocblas_int* LUp,
                        rocblas_int* LUi,
                        T* LUx)
{
    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

    // -------------------------------------------
    // If P_new2old, or Q_old2new is NULL, then treat as identity permutation
    // -------------------------------------------
    bool const has_P = (P_new2old != nullptr);
    bool const has_Q = (Q_old2new != nullptr);
    rocblas_int const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    rocblas_int const irow_inc = blockDim.x * gridDim.x;

    for(rocblas_int irow = irow_start; irow < nrow; irow += irow_inc)
    {
        rocblas_int const kstart_LU = LUp[irow];
        rocblas_int const kend_LU = LUp[irow + 1];
        rocblas_int const nz_LU = kend_LU - kstart_LU;

        // -------------------
        // scale row by beta
        // -------------------
        for(rocblas_int k = 0; k < nz_LU; k++)
        {
            rocblas_int const k_lu = kstart_LU + k;
            T const LUij = LUx[k_lu];
            LUx[k_lu] = (is_beta_zero) ? zero : beta * LUij;
        }

        // -------------------------------
        // check column indices are sorted
        // -------------------------------
        for(rocblas_int k = 0; k < (nz_LU - 1); k++)
        {
            rocblas_int const k_lu = kstart_LU + k;
            rocblas_int const kcol = LUi[k_lu];
            rocblas_int const kcol_next = LUi[k_lu + 1];
            bool const is_sorted = (kcol < kcol_next);
            assert(is_sorted);
        }

        rocblas_int const irow_old = (has_P) ? P_new2old[irow] : irow;
        rocblas_int const kstart_A = Ap[irow_old];
        rocblas_int const kend_A = Ap[irow_old + 1];
        rocblas_int const nz_A = kend_A - kstart_A;

        for(rocblas_int k = 0; k < nz_A; k++)
        {
            rocblas_int const ka = kstart_A + k;

            rocblas_int const jcol_old = Ai[ka];
            rocblas_int const jcol = (has_Q) ? Q_old2new[jcol_old] : jcol_old;

            rocblas_int const len = nz_LU;
            rocblas_int ipos = len;
            {
                rocblas_int const* const arr = &(LUi[kstart_LU]);
                rocblas_int const key = jcol;

                ipos = rf_search(len, arr, key);
                bool const is_found = (0 <= ipos) && (ipos < len) && (arr[ipos] == key);
                assert(is_found);
            }

            rocblas_int const k_lu = kstart_LU + ipos;

            T const aij = Ax[ka];
            LUx[k_lu] += alpha * aij;
        }
    }
}

template <typename T>
rocblas_status rocsolver_csrrf_refactlu_argCheck(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 T valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 T valT,
                                                 rocblas_int* pivP,
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
    if(!rfinfo || !ptrA || !ptrT || (n && (!pivP || !pivQ)) || (nnzA && (!indA || !valA))
       || (nnzT && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_refactlu_getMemorySize(const rocblas_int n,
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
    THROW_IF_ROCSPARSE_ERROR( 
      rocsparseCall_csrilu0_buffer_size(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                       rfinfo->infoT, size_work)  );
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactlu_template(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 U valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 U valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo,
                                                 void* work)
{
    ROCSOLVER_ENTER("csrrf_refactlu", "n:", n, "nnzA:", nnzA, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCSPARSE_CHECK( 
       rocsparse_get_stream(rfinfo->sphandle, &stream),
       rocblas_status_internal_error );


    rocblas_int nthreads = ADD_PAQ_MAX_THDS;
    rocblas_int nblocks = (n + (nthreads - 1)) / nthreads;

    // ---------------------------------------------------------------------
    // copy P*A*Q into T
    // Note: the sparsity pattern of A is a subset of T, and since the re-orderings
    // P and Q are applied, the incomplete factorization of P*A*Q (factorization without fill-in),
    // yields the complete factorization of A.
    // ---------------------------------------------------------------------

    ROCSOLVER_LAUNCH_KERNEL(rf_add_PAQ_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream,
                            n, n, pivP, pivQ, 1, ptrA, indA, valA, 0, ptrT, indT, valT);

    // perform incomplete factorization of T
    ROCSPARSE_CHECK( 
      rocsparseCall_csrilu0(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                            rfinfo->infoT, rocsparse_solve_policy_auto, work),
      rocblas_status_internal_error );


    return rocblas_status_success;
}
