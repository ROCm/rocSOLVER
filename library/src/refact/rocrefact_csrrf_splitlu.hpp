/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprim/rocprim.hpp>

#include <iostream>

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#ifndef SPLITLU_SWITCH_SIZE
#define SPLITLU_SWITCH_SIZE 64
#endif

template <typename I>
__device__ static I cal_wave_size(I avg_nnzM)
{
    const auto ifactor = 2;

    const auto wave_size = ((avg_nnzM >= ifactor * warpSize)              ? warpSize
                                : (avg_nnzM >= ifactor * (warpSize / 2))  ? (warpSize / 2)
                                : (avg_nnzM >= ifactor * (warpSize / 4))  ? (warpSize / 4)
                                : (avg_nnzM >= ifactor * (warpSize / 8))  ? (warpSize / 8)
                                : (avg_nnzM >= ifactor * (warpSize / 16)) ? (warpSize / 16)
                                                                          : 1);
    return (wave_size);
}

template <typename T>
ROCSOLVER_KERNEL __launch_bounds__(BS1) void rf_splitLU_gen_nzLU_kernel(const rocblas_int n,
                                                                        const rocblas_int nnzM,
                                                                        rocblas_int* Mp,
                                                                        rocblas_int* Mi,
                                                                        rocblas_int* nzLarray,
                                                                        rocblas_int* nzUarray)
{
    const auto avg_nnzM = max(1, nnzM / n);

    const auto waveSize = cal_wave_size(avg_nnzM);

    const auto nthreads = gridDim.x * blockDim.x;
    const auto nwaves = nthreads / waveSize;

    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto lid = tid % waveSize;
    const auto wid = tid / waveSize;

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto kstart = Mp[irow];
        const auto kend = Mp[irow + 1];
        // -------------------
        // find diagonal entry
        // -------------------
        for(auto k = kstart + lid; (k < kend); k += waveSize)
        {
            const auto icol = Mi[k];
            if(icol == irow)
            {
                const auto kdiag = k;
                nzUarray[irow] = kend - kdiag;
                nzLarray[irow] = (kdiag - kstart);
                nzLarray[irow] += 1; // add 1 for unit diagonal
            };
        };
    };
}

template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(BS1)
    rf_splitLU_copy_kernel(const rocblas_int n,
                           const rocblas_int nnzM,
                           rocblas_int const* const __restrict__ Mp,
                           rocblas_int const* const __restrict__ Mi,
                           T const* const __restrict__ Mx,
                           rocblas_int const* const __restrict__ Lp,
                           rocblas_int* const __restrict__ Li,
                           T* const __restrict__ Lx,
                           rocblas_int const* const __restrict__ Up,
                           rocblas_int* const __restrict__ Ui,
                           T* const __restrict__ Ux)
{
    const auto avg_nnzM = max(1, nnzM / n);

    const auto waveSize = cal_wave_size(avg_nnzM);

    const auto nthreads = gridDim.x * blockDim.x;
    const auto nwaves = nthreads / waveSize;

    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto lid = tid % waveSize;
    const auto wid = tid / waveSize;

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto kstart = Mp[irow];
        const auto kend = Mp[irow + 1];
        const auto nz = (kend - kstart);

        const auto nzU = (Up[irow + 1] - Up[irow]);
        const auto nzL = (kend - kstart) - nzU;
        const auto kdiag = kstart + nzL;

        for(auto k = kstart + lid; k < kend; k += waveSize)
        {
            const auto icol = Mi[k];
            const auto aij = Mx[k];
            const bool is_lower = (icol < irow);
            if(is_lower)
            {
                const auto ip = Lp[irow] + (k - kstart);
                Li[ip] = icol;
                Lx[ip] = aij;
            }
            else
            {
                const auto ip = Up[irow] + (k - kdiag);
                Ui[ip] = icol;
                Ux[ip] = aij;
            };
        };

        // -------------------
        // unit diagonal entry of L
        // -------------------
        const auto ip = Lp[irow + 1] - 1;
        if(lid == 0)
        {
            Li[ip] = irow;
            Lx[ip] = static_cast<T>(1);
        };
    };
}

// ----------------------------------------------
// Note: intended for execution on a single block
// ----------------------------------------------
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
    if(blockIdx.x != 0)
    {
        return;
    };

    const rocblas_int avg_nnzM = max(1, nnzM / n);

    const auto waveSize = cal_wave_size(avg_nnzM);

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

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const rocblas_int istart = Mp[irow];
        const rocblas_int iend = Mp[irow + 1];

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

    // ---------------------------------
    // prefix sum to setup Lp[] and Up[]
    // ---------------------------------
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

    // ------------------------------------
    // 2nd pass to populate Li, Lx, Ui, Ux
    // ------------------------------------
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

    __syncthreads();
    // -----------------------------
    // set unit diagonal entry in L
    // -----------------------------
    for(auto irow = tid; irow < n; irow += nthreads)
    {
        const auto j = Lp[irow + 1] - 1;
        Li[j] = irow;
        Lx[j] = static_cast<T>(1);
    };
}

template <typename T>
rocblas_status rocsolver_csrrf_splitlu_getMemorySize(const rocblas_int n,
                                                     const rocblas_int nnzT,
                                                     rocblas_int* ptrT,
                                                     size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0 || nnzT == 0)
    {
        *size_work = 0;
        return (rocblas_status_success);
    }

    // space to store the number of non-zeros per row in L and U
    const size_t size_work_LU = sizeof(rocblas_int) * 2 * n;

    // ------------------------------------------
    // query amount of temporary storage required
    // ------------------------------------------
    size_t rocprim_size_bytes = 0;
    void* temp_ptr = nullptr;

    const hipError_t istat = rocprim::inclusive_scan(temp_ptr, rocprim_size_bytes, ptrT, ptrT, n,
                                                     rocprim::plus<rocblas_int>());
    assert(istat == hipSuccess);
    if(istat != hipSuccess)
    {
        return (rocblas_status_internal_error);
    };

    *size_work = max(rocprim_size_bytes, size_work_LU);

    return (rocblas_status_success);
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
                                                rocblas_int* work,
                                                size_t size_work)
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
        const rocblas_int blocks = n / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrU, n + 1, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrL, n + 1, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, indL, n, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, valL, n, 1);

        return rocblas_status_success;
    }

    if(n <= SPLITLU_SWITCH_SIZE)
    {
        // --------------------------------
        // note using a single thread block
        // --------------------------------
        const rocblas_int nthreads = BS1;
        const rocblas_int nblocks = 1;
        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, n,
                                nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);
    }
    else
    {
        rocblas_int const nthreads = BS1;
        rocblas_int const nblocks = max(1, min(1024, (n - 1) / nthreads + 1));

        rocblas_int* const Lp = ptrL;
        rocblas_int* const Up = ptrU;

        // ------------------------------------------------
        // setup number of nonzeros in each row of L and U
        // note: reuse arrays Lp[] and Up[]
        // ------------------------------------------------
        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_gen_nzLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0,
                                stream, n, nnzT, ptrT, indT, Lp + 1, Up + 1);

        // -------------------------------------
        // generate prefix sum for Lp[] and Up[]
        // note: in-place prefix sum
        // -------------------------
        {
            // ------------------------------------------
            // query amount of temporary storage required
            // ------------------------------------------
            size_t storage_size_bytes_Lp = 0;
            size_t storage_size_bytes_Up = 0;

            void* temp_ptr = nullptr;

            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes_Lp, (Lp + 1), (Lp + 1),
                                              n, rocprim::plus<rocblas_int>(), stream));

            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes_Up, (Up + 1), (Up + 1),
                                              n, rocprim::plus<rocblas_int>(), stream));

            size_t storage_size_bytes = max(storage_size_bytes_Lp, storage_size_bytes_Up);
            assert(size_work >= storage_size_bytes);

            if(size_work < storage_size_bytes)
            {
                return (rocblas_status_internal_error);
            };

            temp_ptr = (void*)work;
            storage_size_bytes = size_work;

            // ----------------------------------------
            // perform inclusive scan for Lp[] and Up[]
            // ----------------------------------------
            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes, (Lp + 1), (Lp + 1), n,
                                              rocprim::plus<rocblas_int>(), stream));

            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes, (Up + 1), (Up + 1), n,
                                              rocprim::plus<rocblas_int>(), stream));
        }

        // ------------------------
        // set Lp[0] = 0, Up[0] = 0
        // ------------------------
        {
            const rocblas_int ival = static_cast<rocblas_int>(0);

            HIP_CHECK(hipMemcpyAsync(Lp, &ival, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));

            HIP_CHECK(hipMemcpyAsync(Up, &ival, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        }

        // -----------------
        // copy into L and U
        // -----------------
        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_copy_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream,
                                n, nnzT, ptrT, indT, valT, Lp, indL, valL, Up, indU, valU);
    }

    return rocblas_status_success;
}
