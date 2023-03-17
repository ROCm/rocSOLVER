/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "rocblas_check.h"

// ----------------------------------------------------
// device (serial) code to perform shell sort by a single thread
// key is  iarr[]
// data is darr[]
// ----------------------------------------------------
template <typename T>
static __device__ void rf_shellsort(rocblas_int* iarr, T* darr, rocblas_int num)
{
    for(rocblas_int i = num / 2; i > 0; i = i / 2)
    {
        for(rocblas_int j = i; j < num; j++)
        {
            for(rocblas_int k = j - i; k >= 0; k = k - i)
            {
                if(iarr[k + i] >= iarr[k])
                {
                    break;
                }
                else
                {
                    // swap entries
                    rocblas_int const itmp = iarr[k];
                    iarr[k] = iarr[k + i];
                    iarr[k + i] = itmp;

                    T const dtmp = darr[k];
                    darr[k] = darr[k + i];
                    darr[k + i] = dtmp;
                }
            }
        }
    }

    bool const perform_check = true;
    if(perform_check)
    {
        for(rocblas_int i = 0; i < (num - 1); i++)
        {
            bool const is_sorted = (iarr[i] <= iarr[i + 1]);
            assert(is_sorted);
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void
    rf_setupLUp_kernel(const rocblas_int nrow, rocblas_int* Lp, rocblas_int* Up, rocblas_int* LUp)
{
    // ---------------------------------------------
    // just use a single thread block for simplicity
    // ---------------------------------------------
    bool const is_root = (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0);
    if(!is_root)
    {
        return;
    }

    // -----------------------------------------
    // compute number of non-zeros per row in LU
    // -----------------------------------------
    rocblas_int const nthreads = blockDim.x;
    rocblas_int const i_start = threadIdx.x;
    rocblas_int const i_inc = nthreads;
    for(rocblas_int i = i_start; i < nrow; i += i_inc)
    {
        rocblas_int const irow = i;
        rocblas_int const nnz_L = Lp[irow + 1] - Lp[irow];
        rocblas_int const nnz_U = Up[irow + 1] - Up[irow];

        // -----------------------------------------
        // note: assume L has explicit unit diagonal
        // so use (nnzL - 1)
        // -----------------------------------------
        rocblas_int const nnz_LU = (nnz_L - 1) + nnz_U;

        LUp[irow] = nnz_LU;
    }
    __syncthreads();

    // ---------------------------------------
    // prepare for prefix sum in shared memory
    // ---------------------------------------
    rocblas_int constexpr MAX_THREADS = 1024;
    __shared__ rocblas_int isum[MAX_THREADS];

    for(rocblas_int i = i_start; i < nthreads; i += i_inc)
    {
        // ---------------------------------------------
        // the i-th thread computes
        //
        //    isum[ i ] = sum( LUp[ istart..(iend-1) ] )
        // ---------------------------------------------
        rocblas_int const nb = (nrow + (nthreads - 1)) / nthreads;
        rocblas_int const irow_start = i * nb;
        rocblas_int const irow_end = min(nrow, irow_start + nb);

        rocblas_int presum = 0;
        for(rocblas_int irow = irow_start; irow < irow_end; irow++)
        {
            rocblas_int const nnz_LU = LUp[irow];
            presum += nnz_LU;
        }
        isum[i] = presum;
    }
    __syncthreads();

    // ------------------
    // compute prefix sum
    // use single thread for simplicity
    // ------------------
    rocblas_int offset = 0;
    rocblas_int nnz_LU = 0;
    bool const is_root_thread = (threadIdx.x == 0);
    if(is_root_thread)
    {
        for(rocblas_int i = 0; i < nthreads; i++)
        {
            rocblas_int const isum_i = isum[i];
            isum[i] = offset;
            offset += isum_i;
        }

        nnz_LU = offset;
        LUp[nrow] = nnz_LU;
    }
    __syncthreads();

    // ----------
    // update LUp
    // ----------
    for(rocblas_int i = i_start; i < nthreads; i += i_inc)
    {
        rocblas_int const nb = (nrow + (nthreads - 1)) / nthreads;
        rocblas_int const irow_start = i * nb;
        rocblas_int const irow_end = min(nrow, irow_start + nb);

        rocblas_int ipos = isum[i];
        for(rocblas_int irow = irow_start; irow < irow_end; irow++)
        {
            rocblas_int const nz = LUp[irow];
            LUp[irow] = ipos;
            ipos += nz;
        }
    }
    __syncthreads();

    bool constexpr perform_check = true;
    if(perform_check)
    {
        // ------------
        // check
        // ------------
        __syncthreads();

        for(rocblas_int i = i_start; i < nrow; i += i_inc)
        {
            rocblas_int const irow = i;
            rocblas_int const nnz_L = Lp[irow + 1] - Lp[irow];
            rocblas_int const nnz_U = Up[irow + 1] - Up[irow];

            // ----------------------------------------------
            // note assume explicit unit diagonal in matrix L
            // so use (nnzL - 1)
            // ----------------------------------------------
            rocblas_int const nnz_LU = (nnz_L - 1) + nnz_U;

            rocblas_int const nz_irow = (LUp[irow + 1] - LUp[irow]);
            bool const isok = (nz_irow == nnz_LU);

            assert(isok);
        }
        __syncthreads();

        rocblas_int const nnzL = (Lp[nrow] - Lp[0]);
        rocblas_int const nnzU = (Up[nrow] - Up[0]);
        rocblas_int const nnzLU = (LUp[nrow] - LUp[0]);
        bool const isok = (((nnzL - nrow) + (nnzU)) == nnzLU);
        assert(isok);
    }
    __syncthreads();
}

template <typename T>
ROCSOLVER_KERNEL void rf_sumLU_kernel(const rocblas_int nrow,
                                      const rocblas_int ncol,
                                      rocblas_int* Lp,
                                      rocblas_int* Li,
                                      T* Lx,
                                      rocblas_int* Up,
                                      rocblas_int* Ui,
                                      T* Ux,
                                      rocblas_int* LUp,
                                      rocblas_int* LUi,
                                      T* LUx)
{
    rocblas_int const nnzL = Lp[nrow] - Lp[0];
    rocblas_int const nnzU = Up[nrow] - Up[0];
    rocblas_int const nnzLU = LUp[nrow] - LUp[0];

    bool const isok = (nnzLU == (nnzL + nnzU - nrow));
    assert(isok);

    rocblas_int const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    rocblas_int const irow_inc = blockDim.x * gridDim.x;

    for(rocblas_int irow = irow_start; irow < nrow; irow += irow_inc)
    {
        rocblas_int const kstart_L = Lp[irow];
        rocblas_int const kend_L = Lp[irow + 1];
        rocblas_int const nz_L = (kend_L - kstart_L);

        rocblas_int const kstart_U = Up[irow];
        rocblas_int const kend_U = Up[irow + 1];
        rocblas_int const nz_U = (kend_U - kstart_U);

        rocblas_int const kstart_LU = LUp[irow];
        rocblas_int const kend_LU = LUp[irow + 1];
        rocblas_int const nz_LU = (kend_LU - kstart_LU);

        // --------------
        // zero out array LUi[], LUx[]
        // --------------
        T const zero = 0.0;
        for(rocblas_int k = 0; k < nz_LU; k++)
        {
            rocblas_int const ip = kstart_LU + k;
            LUi[ip] = 0;
            LUx[ip] = zero;
        }

        // --------------
        // copy L into LU
        // --------------
        rocblas_int ip = kstart_LU;
        for(rocblas_int k = 0; k < nz_L; k++)
        {
            rocblas_int const jp = kstart_L + k;
            rocblas_int const jcol = Li[jp];
            T const Lij = Lx[jp];
            bool const is_strictly_lower = (irow > jcol);
            if(is_strictly_lower)
            {
                LUi[ip] = jcol;
                LUx[ip] = Lij;
                ip++;
            }
        }

        // --------------
        // copy U into LU
        // --------------
        for(rocblas_int k = 0; k < nz_U; k++)
        {
            rocblas_int const jp = kstart_U + k;
            rocblas_int const jcol = Ui[jp];
            T const Uij = Ux[jp];
            bool const is_upper = (irow <= jcol);
            if(is_upper)
            {
                LUi[ip] = jcol;
                LUx[ip] = Uij;
                ip++;
            }
        }

        bool const is_filled = (ip == kend_LU);
        assert(is_filled);

        // ----------------------------------------
        // check column indices are in sorted order
        // ----------------------------------------
        bool is_sorted = true;
        for(rocblas_int k = 0; k < (nz_LU - 1); k++)
        {
            rocblas_int const ip = kstart_LU + k;
            is_sorted = (LUi[ip] < LUi[ip + 1]);
            if(!is_sorted)
            {
                break;
            }
        }

        if(!is_sorted)
        {
            // ----------------------------------
            // sort row in LU using shellsort algorithm
            // ----------------------------------
            rocblas_int* iarr = &(LUi[kstart_LU]);
            T* darr = &(LUx[kstart_LU]);
            rocblas_int const num = nz_L;
            rf_shellsort(iarr, darr, num);
        }
    } // end for irow

    __syncthreads();
}

template <typename T>
rocblas_status rocsolver_csrrf_sumlu_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nnzL,
                                              rocblas_int* ptrL,
                                              rocblas_int* indL,
                                              T valL,
                                              const rocblas_int nnzU,
                                              rocblas_int* ptrU,
                                              rocblas_int* indU,
                                              T valU,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              T valT)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    };

    // 2. invalid size
    if(n < 0 || nnzL < 0 || nnzU < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzL && (!indL || !valL)) || (nnzU && (!indU || !valU))
       || (nnzL * nnzU && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_sumlu_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nnzL,
                                              rocblas_int* ptrL,
                                              rocblas_int* indL,
                                              U valL,
                                              const rocblas_int nnzU,
                                              rocblas_int* ptrU,
                                              rocblas_int* indU,
                                              U valU,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              U valT)
{
    ROCSOLVER_ENTER("csrrf_sumlu", "n:", n, "nnzL:", nnzL, "nnzU:", nnzU);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream), rocblas_status_internal_error);

    // Step 1: setup row pointer ptrT
    rocblas_int nthreads = 1024;
    rocblas_int nblocks = 1; // special case use only a single block

    ROCSOLVER_LAUNCH_KERNEL(rf_setupLUp_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, n,
                            ptrL, ptrU, ptrT);

    // Step 2: copy entries of indL, valL, indU, valU into indT, valT
    nthreads = 128;
    nblocks = (n + (nthreads - 1)) / nthreads;

    {
        rocblas_int nrow = n;
        rocblas_int ncol = n;
        ROCSOLVER_LAUNCH_KERNEL(rf_sumLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, nrow,
                                ncol, ptrL, indL, valL, ptrU, indU, valU, ptrT, indT, valT);
    };

    return rocblas_status_success;
}
