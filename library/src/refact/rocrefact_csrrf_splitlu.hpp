/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

/*
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
    rocblas_int* nzLp = work;
    rocblas_int* nzUp = work + n;

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // -------------------------------------------------
    for(int i = 0; i < n; i++)
    {
        nzLp[i] = 0;
        nzUp[i] = 0;
    };

    int nnzL = 0;
    int nnzU = 0;
    for(int irow = 0; irow < n; irow++)
    {
        int const istart = Mp[irow];
        int const iend = Mp[irow + 1];
        int const nz = (iend - istart);

        int nzU = 0;
        for(int k = istart; k < iend; k++)
        {
            int const kcol = Mi[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                nzU++;
            }
        }
        int const nzL = nz - nzU;

        nzLp[irow] = (nzL + 1); // add 1 for unit diagonal
        nzUp[irow] = nzU;

        nnzL += (nzL + 1);
        nnzU += nzU;
    }

    // ------------------------------------
    // prefix sum scan to setup Lp and Up
    // ------------------------------------
    int iL = 0;
    int iU = 0;
    for(int irow = 0; irow < n; irow++)
    {
        int const nzL = nzLp[irow];
        int const nzU = nzUp[irow];
        Lp[irow] = iL;
        iL += nzL;

        Up[irow] = iU;
        iU += nzU;
    }
    Up[n] = nnzU;
    Lp[n] = nnzL;

    // ---------------------------------------------------
    // second pass to populate  Li[], Lx[], Ui[], Ux[]
    // ---------------------------------------------------

    for(int irow = 0; irow < n; irow++)
    {
        nzLp[irow] = Lp[irow];
        nzUp[irow] = Up[irow];
    }

    double const one = 1;

    for(int irow = 0; irow < n; irow++)
    {
        int const istart = Mp[irow];
        int const iend = Mp[irow + 1];
        for(int k = istart; k < iend; k++)
        {
            int const kcol = Mi[k];
            double const mij = Mx[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                int const ip = nzUp[irow];
                nzUp[irow]++;

                Ui[ip] = kcol;
                Ux[ip] = mij;
            }
            else
            {
                int const ip = nzLp[irow];
                nzLp[irow]++;

                Li[ip] = kcol;
                Lx[ip] = mij;
            }
        }
    }

    // ------------------------
    // set unit diagonal entry in L
    // ------------------------
    for(int irow = 0; irow < n; irow++)
    {
        int const kend = Lp[irow + 1];
        int const ip = kend - 1;
        Li[ip] = irow;
        Lx[ip] = one;
    }
}
*/

template <typename T>
void rocsolver_csrrf_splitlu_getMemorySize(const rocblas_int n, size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0)
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

    // 2. invalid size
    if(n < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzT && (!indT || !valT || !indU || !valU || !indL || valL)))
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
    rocblas_get_stream(handle, &stream);

    /* TODO:

    ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(1), dim3(1), 0, stream,
                            n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);

    */

    return rocblas_status_success;
}
