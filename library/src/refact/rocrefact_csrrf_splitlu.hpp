/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "rocblas_check.h"
#include "rocsparse_check.h"

template <typename T>
ROCSOLVER_KERNEL void rf_splitLU_kernel(const rocblas_int n,
                                        const rocblas_int nnzM,
                                        rocblas_int* Mp,
                                        rocblas_int* Mi,
                                        T* Mx,
                                        rocblas_int* Lp,
                                        rocblas_int* Li,
                                        T* Lx,
                                        rocblas_int* Up,
                                        rocblas_int* Ui,
                                        T* Ux,
                                        rocblas_int* work)
{
    // ---------------------------------
    // use a single block for simplicity
    // ---------------------------------
    bool const is_root_block = (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0);
    if(!is_root_block)
    {
        return;
    };

    rocblas_int* const nzLp = work;
    rocblas_int* const nzUp = work + n;

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // -------------------------------------------------

    rocblas_int const nthreads = blockDim.x;
    rocblas_int const my_thread = threadIdx.x;
    rocblas_int const i_start = my_thread;
    rocblas_int const i_inc = nthreads;
    bool const is_root_thread = (my_thread == 0);

    __syncthreads();

    for(rocblas_int i = i_start; i < n; i += i_inc)
    {
        nzLp[i] = 0;
        nzUp[i] = 0;
    };
    __syncthreads();

    rocblas_int const nb = (n + (nthreads - 1)) / nthreads;
    rocblas_int const irow_start = my_thread * nb;
    rocblas_int const irow_end = min(n, irow_start + nb);

    rocblas_int nnzL = 0;
    rocblas_int nnzU = 0;

    rocblas_int constexpr MAX_THREADS = 1024 * 2;
    __shared__ rocblas_int isum_nnzL[MAX_THREADS];
    __shared__ rocblas_int isum_nnzU[MAX_THREADS];

    __syncthreads();

    bool const isok = (0 <= my_thread) && (my_thread < MAX_THREADS) && (nthreads <= MAX_THREADS);
    assert(isok);

    isum_nnzL[my_thread] = 0;
    isum_nnzU[my_thread] = 0;

    __syncthreads();

    for(rocblas_int irow = irow_start; irow < irow_end; irow++)
    {
        rocblas_int const istart = Mp[irow];
        rocblas_int const iend = Mp[irow + 1];
        rocblas_int const nz = (iend - istart);

        rocblas_int nzU = 0;
        for(rocblas_int k = istart; k < iend; k++)
        {
            rocblas_int const kcol = Mi[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                nzU++;
            }
        }
        int const nzL = nz - nzU;

        nzLp[irow] = (nzL + 1); // add 1 for unit diagonal
        nzUp[irow] = nzU;

        isum_nnzL[my_thread] += (nzL + 1);
        isum_nnzU[my_thread] += nzU;
    }; // end for irow

    __syncthreads();

    nnzL = 0;
    nnzU = 0;

    // -----------------------------------
    // use a single thread for simplicity
    // -----------------------------------
    if(is_root_thread)
    {
        for(rocblas_int ithread = 0; ithread < nthreads; ithread++)
        {
            nnzL += isum_nnzL[ithread];
            nnzU += isum_nnzU[ithread];
        };

        Up[n] = nnzU;
        Lp[n] = nnzL;
    };

    __syncthreads();

    // ------------------------------------
    // prefix sum scan to setup Lp and Up
    // ------------------------------------

    if(is_root_thread)
    {
        rocblas_int iposL = 0;
        rocblas_int iposU = 0;

        for(rocblas_int ithread = 0; ithread < nthreads; ithread++)
        {
            rocblas_int const nzL = isum_nnzL[ithread];
            rocblas_int const nzU = isum_nnzU[ithread];

            isum_nnzL[ithread] = iposL;
            isum_nnzU[ithread] = iposU;

            iposL += nzL;
            iposU += nzU;
        };
    };

    __syncthreads();

    // --------------------------------------------------------------
    // isum_nnzL[ my_thread ] now contains value for Lp[ irow_start ]
    // isum_nnzU[ my_thread ] now contains value for Up[ irow_start ]
    // --------------------------------------------------------------

    rocblas_int iL = isum_nnzL[my_thread];
    rocblas_int iU = isum_nnzU[my_thread];

    // ------------------------------------------------
    // setup Lp[ irow ], irow=irow_start .. (irow_end-1)
    // setup Up[ irow ], irow=irow_start .. (irow_end-1)
    // ------------------------------------------------
    __syncthreads();

    for(rocblas_int irow = irow_start; irow < irow_end; irow++)
    {
        rocblas_int const nzL = nzLp[irow];
        rocblas_int const nzU = nzUp[irow];
        Lp[irow] = iL;
        iL += nzL;

        Up[irow] = iU;
        iU += nzU;
    }
    __syncthreads();

    // ---------------------------------------------------
    // second pass to populate  Li[], Lx[], Ui[], Ux[]
    // ---------------------------------------------------

    // -----------------------------------------------
    // reuse array nzLp[] as pointers into  Lx[], Li[]
    // reuse array nzUp[] as pointers into  Ux[], Ui[]
    // -----------------------------------------------

    __syncthreads();

    for(rocblas_int i = i_start; i < n; i += i_inc)
    {
        rocblas_int const irow = i;

        nzLp[irow] = Lp[irow];
        nzUp[irow] = Up[irow];
    }

    __syncthreads();

    T const one = 1;

    for(rocblas_int irow = irow_start; irow < irow_end; irow++)
    {
        rocblas_int const istart = Mp[irow];
        rocblas_int const iend = Mp[irow + 1];

        for(rocblas_int k = istart; k < iend; k++)
        {
            rocblas_int const kcol = Mi[k];
            T const mij = Mx[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                rocblas_int const ip = nzUp[irow];
                nzUp[irow]++;

                Ui[ip] = kcol;
                Ux[ip] = mij;
            }
            else
            {
                rocblas_int const ip = nzLp[irow];
                nzLp[irow]++;

                Li[ip] = kcol;
                Lx[ip] = mij;
            }
        }
    };

    __syncthreads();

    // ------------------------
    // set unit diagonal entry in L
    // ------------------------

    for(rocblas_int irow = irow_start; irow < irow_end; irow++)
    {
        rocblas_int const kend = Lp[irow + 1];
        rocblas_int const ip = kend - 1;
        Li[ip] = irow;
        Lx[ip] = one;
    };
    __syncthreads();

    bool const perform_extra_check = true;
    if(perform_extra_check)
    {
        // -----------------------------
        // check upper triangular matrix
        // -----------------------------
        __syncthreads();

        for(rocblas_int i = i_start; i < n; i += i_inc)
        {
            rocblas_int const irow = i;
            rocblas_int istart = Up[irow];
            rocblas_int iend = Up[irow + 1];

            // -----------------------------------------
            // check column indices are upper triangular
            // -----------------------------------------
            for(rocblas_int k = istart; k < iend; k++)
            {
                rocblas_int const kcol = Ui[k];
                bool const is_upper = (irow <= kcol);
                assert(is_upper);
            };

            // -------------------------------
            // check column indices are sorted
            // -------------------------------
            for(rocblas_int k = istart; k < (iend - 1); k++)
            {
                rocblas_int const kcol_k = Ui[k];
                rocblas_int const kcol_kp1 = Ui[k + 1];
                bool const is_sorted = (kcol_k < kcol_kp1);
                assert(is_sorted);
            };
        };

        // -----------------------------
        // check lower triangular matrix
        // -----------------------------
        __syncthreads();

        for(rocblas_int i = i_start; i < n; i += i_inc)
        {
            rocblas_int const irow = i;
            rocblas_int const istart = Lp[irow];
            rocblas_int const iend = Lp[irow + 1];

            // -----------------------------------------
            // check column indices are lower triangular
            // -----------------------------------------
            for(rocblas_int k = istart; k < iend; k++)
            {
                rocblas_int const kcol = Li[k];
                bool const is_lower = (irow >= kcol);
                assert(is_lower);
            };

            // -----------------------------------------
            // check column indices are sorted
            // -----------------------------------------
            for(rocblas_int k = istart; k < (iend - 1); k++)
            {
                rocblas_int const kcol_k = Li[k];
                rocblas_int const kcol_kp1 = Li[k + 1];
                bool const is_sorted = (kcol_k < kcol_kp1);
                assert(is_sorted);
            };

            // -------------------
            // check unit diagonal
            // -------------------
            rocblas_int const k = (iend - 1);
            rocblas_int const kcol = Li[k];
            bool const is_unit_diagonal = (kcol == irow) && (Lx[k] == one);
            assert(is_unit_diagonal);
        };

    }; // end if (perform_extra_check)
}

template <typename T>
void rocsolver_csrrf_splitlu_getMemorySize(const rocblas_int n,
                                           const rocblas_int nnzT,
                                           size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0 || nnzT == 0)
    {
        *size_work = 0;
        return;
    }

    // space to store the number of non-zeros per row in L and U
    *size_work = sizeof(rocblas_int) * 2 * n;
}

template <typename T>
rocblas_status rocsolver_csrrf_splitlu_argCheck(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nnzT,
                                                rocblas_int* ptrT,
                                                rocblas_int* indT,
                                                T valT,
                                                rocblas_int* ptrL,
                                                rocblas_int* indL,
                                                T valL,
                                                rocblas_int* ptrU,
                                                rocblas_int* indU,
                                                T valU)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    };

    // 2. invalid size
    if(n < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzT && (!indT || !valT || !indU || !valU))
       || ((n || nnzT) && (!indL || !valL)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_splitlu_template(rocblas_handle handle,
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
                                                U valU,
                                                rocblas_int* work)
{
    ROCSOLVER_ENTER("csrrf_splitlu", "n:", n, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream), rocblas_status_internal_error);

    // quick return with matrix zero
    if(nnzT == 0)
    {
        // set ptrU = 0
        rocblas_int blocks = n / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrU, n + 1, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrL, n + 1, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, indL, n, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, valL, n, 1);

        return rocblas_status_success;
    }

    rocblas_int const nthreads = 1024;
    rocblas_int const nblocks = 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, n, nnzT,
                            ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);

    return rocblas_status_success;
}
