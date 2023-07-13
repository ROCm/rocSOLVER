
/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver/hipsolver.h"
#include "hipsparse/hipsparse.h"

__global__ void check_issym_kernel(const int m,
                                   const int* csrRowPtrA,
                                   const int* csrEndPtrA,
                                   const int* csrColIndA,
                                   int* d_issym)
{
    auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto const iwave = tid / warpSize;
    auto const nwave = (gridDim.x * blockDim.x) / warpSize;
    auto const lid = tid % warpSize;

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

        for(auto k = csrRowPtrA[irow] + lid; k < csrEndPtrA[irow]; k += warpSize)
        {
            auto const kcol = csrColIndA[k];
            // entry (irow, kcol)

            auto const krow = kcol;
            bool has_found = false;
            for(auto j = csrRowPtrA[krow]; j < csrEndPtrA[krow]; j++)
            {
                auto const jcol = csrColIndA[j];

                // entry (krow, jcol)
                has_found = (irow == jcol);
                if(has_found)
                {
                    break;
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
        int const nthreads = 256;
        int const nblocks = std::max(1, std::min(m / nthreads, 1024));
        check_issym_kernel<<<dim3(nblocks), dim3(nthreads), 0, 0>>>(m, csrRowPtrA, csrEndPtrA,
                                                                    csrColIndA, d_isymm);
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
