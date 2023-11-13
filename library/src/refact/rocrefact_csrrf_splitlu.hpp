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

#include <iostream>

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

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
    const rocblas_int avg_nnzM = max(1, nnzM / n);

    const rocblas_int waveSize = (avg_nnzM > warpSize) ? warpSize
        : (avg_nnzM > warpSize / 2)                    ? warpSize / 2
        : (avg_nnzM > warpSize / 4)                    ? warpSize / 4
        : (avg_nnzM > warpSize / 8)                    ? warpSize / 8
        : (avg_nnzM > warpSize / 16)                   ? warpSize / 16
                                                       : 1;

    const rocblas_int nthreads = blockDim.x;
    const rocblas_int nwaves = nthreads / waveSize;

    const rocblas_int tid = threadIdx.x;
    const rocblas_int lid = tid % waveSize;
    const rocblas_int wid = tid / waveSize;

    rocblas_int* const diagpos = work;

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // and set up Lp and Up
    // -------------------------------------------------

    auto time_diagpos = -clock64();

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const rocblas_int istart = Mp[irow];
        const rocblas_int iend = Mp[irow + 1];

#pragma unroll
        for(auto i = istart + lid; i < iend; i += waveSize)
        {
            const auto icol = Mi[i];
            if(icol == irow)
            {
                diagpos[irow] = i;
            };
        }
    }
    __syncthreads();
    time_diagpos += clock64();

    // ---------------------------------
    // prefix sum to setup Lp[] and Up[]
    // ---------------------------------
    auto time_sum = -clock64();
    if(tid == 0)
    {
        rocblas_int nnzL = 0;
        rocblas_int nnzU = 0;

        for(auto irow = 0 * n; irow < n; irow++)
        {
            Lp[irow] = nnzL;
            Up[irow] = nnzU;

            const auto istart = Mp[irow];
            const auto iend = Mp[irow + 1];
            const auto idiag = diagpos[irow];

            const auto nzUp_i = iend - idiag;
            const auto nzLp_i = idiag - istart + 1; // add 1 for unit diagonal

            nnzL += nzLp_i;
            nnzU += nzUp_i;
        };

        Lp[n] = nnzL;
        Up[n] = nnzU;
    };
    __syncthreads();
    time_sum += clock64();

    // ------------------------------------
    // 2nd pass to populate Li, Lx, Ui, Ux
    // ------------------------------------
    auto time_copy = -clock64();
    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto istart = Mp[irow];
        const auto iend = Mp[irow + 1];

        const auto idiag = diagpos[irow];

        // -----------
        // copy into L
        // -----------
        {
            const auto nzLp_i = idiag - istart + 1; // add 1 for unit diagonal
            for(auto k = lid; k < nzLp_i; k += waveSize)
            {
                const auto ip = Lp[irow] + k;
                const auto icol = Mi[istart + k];
                const auto aij = Mx[istart + k];

                Li[ip] = icol;
                Lx[ip] = aij;
            }
        }

        // -----------
        // copy into U
        // -----------
        {
            const auto nzUp_i = iend - idiag;
            for(auto k = lid; k < nzUp_i; k += waveSize)
            {
                const auto ip = Up[irow] + k;
                const auto icol = Mi[idiag + k];
                const auto aij = Mx[idiag + k];

                Ui[ip] = icol;
                Ux[ip] = aij;
            }
        }
    }

    // -----------------------------
    // set unit diagonal entry in L
    // -----------------------------
    for(auto irow = tid; irow < n; irow += nthreads)
    {
        const auto j = Lp[irow + 1] - 1;
        Li[j] = irow;
        Lx[j] = static_cast<T>(1);
    };

    __syncthreads();
    time_copy += clock64();
    if(tid == 0)
    {
        printf("time_diagpos=%le, time_sum=%le, time_copy=%le\n", static_cast<double>(time_diagpos),
               static_cast<double>(time_sum), static_cast<double>(time_copy));
    }
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
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

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

    rocblas_int const nthreads = BS1;
    rocblas_int const nblocks = 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, n, nnzT,
                            ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);

    return rocblas_status_success;
}
