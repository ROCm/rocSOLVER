
/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver/hipsolver.h"
#include "hipsparse/hipsparse.h"

#ifndef ISSYM_NTHREADS
#define ISSYM_NTHREADS 256
#endif

template <int waveSize>
__global__ void __launch_bounds__(ISSYM_NTHREADS) check_issym_kernel(const int m,
                                                                     const int* csrRowPtrA,
                                                                     const int* csrEndPtrA,
                                                                     const int* csrColIndA,
                                                                     int* d_issym)
{
    auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto const iwave = tid / waveSize;
    auto const nwave = (gridDim.x * blockDim.x) / waveSize;
    auto const lid = tid % waveSize;

#include "rf_search.hpp"

    // ----------------------------
    // assume *d_issym = 1 on entry
    // ----------------------------

    for(auto irow = iwave; irow < m; irow += nwave)
    {
        if(*d_issym == 0)
        {
            // -------------------------------------------------------------------
            // sparse matrix is already known to be non-symmetric, so quick return
            // -------------------------------------------------------------------
            break;
        };

        for(auto k = csrRowPtrA[irow] + lid; k < csrRowPtrA[irow + 1]; k += waveSize)
        {
            auto const kcol = csrColIndA[k];
            // entry (irow, kcol)

            auto const krow = kcol;
            bool has_found = false;

            bool const use_rf_search = true;
            if(use_rf_search)
            {
                // -----------------------------------------
                // use binary search if csrColIndA is sorted
                // -----------------------------------------

                auto const key = irow;
                auto const jstart = csrRowPtrA[krow];
                auto const jend = csrEndPtrA[krow];
                auto const ipos = rf_search(csrColIndA, jstart, jend, key);
                has_found = (ipos >= jstart);
            }
            else
            {
                // ------------------------------------
                // use simpler but slower linear search
                // ------------------------------------

                for(auto j = csrRowPtrA[krow]; j < csrEndPtrA[krow]; j++)
                {
                    auto const jcol = csrColIndA[j];

                    // ---------------------------------------------------
                    // is entry (krow, jcol) the transpose of  (irow,kcol) ?
                    // ---------------------------------------------------
                    has_found = (irow == jcol);
                    if(has_found)
                    {
                        break;
                    };
                };
            };

            if(!has_found)
            {
                // -------------------------------------
                // detect non-symmetric sparsity pattern
                // -------------------------------------

                *d_issym = 0;
                __threadfence();
            };
        };
    };
}

extern "C" {

hipsolverStatus_t hipsolverSpXcsrissym(
    // TODO: check whether hipsolverSpHandle_t or hipsolverHandle_t

    hipsolverHandle_t handle,
    int m,
    int nnzA,
    const hipsparseMatDescr_t descrA,
    const int* csrRowPtrA,
    const int* csrEndPtrA,
    const int* csrColIndA,
    int* p_issym)
{
#ifdef HAVE_ROCSPARSE

    // ----------------
    // check arguments
    // ----------------

    if(handle == nullptr)
    {
        return (HIPSOLVER_STATUS_INVALID_VALUE);
    };

    {
        bool const isok = (m >= 0) && (nnzA >= 0) && (csrRowPtrA != nullptr)
            && (csrEndPtrA != nullptr) && (csrColIndA != nullptr) && (p_isymm != nullptr);

        if(!isok)
        {
            return (HIPSOLVER_STATUS_INVALID_VALUE);
        };
    };

    *p_issym = 1;

    int* d_issym = nullptr;

    hipError_t istat_hip = hipMalloc(d_issym, sizeof(int));
    bool isok = (istat_hip == hipSuccess) && (d_issym != nullptr);
    if(!isok)
    {
        return (HIPSOLVER_STATUS_EXECUTION_FAILED);
    };

    istat_hip = hipMemcpy(d_issym, p_issym, sizeof(int), hipMemcpyHostToDevice);
    if(istat_hip != hipSuccess)
    {
        return (HIPSOLVER_STATUS_EXECUTION_FAILED);
    };

    {
        int const nthreads = ISSYM_NTHREADS;
        int const nblocks = std::max(1, std::min(m / nthreads, 1024));
        int const avgnnz = std::max(1, nnzA / m);

        if(avgnnz >= ISSYM_NTHREADS)
        {
            check_issym_kernel<ISSYM_NTHREADS><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(
                m, csrRowPtrA, csrEndPtrA, csrColIndA, d_isymm);
        }
        else if(avgnnz >= 128)
        {
            check_issym_kernel<128><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(
                m, csrRowPtrA, csrEndPtrA, csrColIndA, d_isymm);
        }
        else if(avgnnz >= 64)
        {
            check_issym_kernel<64><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(
                m, csrRowPtrA, csrEndPtrA, csrColIndA, d_isymm);
        }
        else if(avgnnz >= 32)
        {
            check_issym_kernel<32><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(
                m, csrRowPtrA, csrEndPtrA, csrColIndA, d_isymm);
        }
        else if(avgnnz >= 16)
        {
            check_issym_kernel<16><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(
                m, csrRowPtrA, csrEndPtrA, csrColIndA, d_isymm);
        }
        else if(avgnnz >= 8)
        {
            check_issym_kernel<8><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(m, csrRowPtrA, csrEndPtrA,
                                                                           csrColIndA, d_isymm);
        }
        else
        {
            check_issym_kernel<4><<<dim3(nblocks), dim3(nthreads), 0, 0>>>(m, csrRowPtrA, csrEndPtrA,
                                                                           csrColIndA, d_isymm);
        };
    }

    istat_hip = hipMemcpy(p_issym, d_issym, sizeof(int), hipMemcpyDeviceToHost);
    if(istat_hip != hipSuccess)
    {
        return (HIPSOLVER_STATUS_EXECUTION_FAILED);
    };

    istat_hip = hipFree(d_issym);
    if(istat_hip != hipSuccess)
    {
        return (HIPSOLVER_STATUS_EXECUTION_FAILED);
    };

    return (HIPSOLVER_STATUS_SUCCESS);
#else
    return (HIPSOLVER_STATUS_SUCCESS);
#endif
}
}
