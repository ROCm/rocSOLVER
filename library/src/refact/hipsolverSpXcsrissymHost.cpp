
/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver/hipsolver.h"
#include "hipsparse/hipsparse.h"

extern "C" {

hipsolverStatus_t hipsolverSpXcsrissymHost(
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

    {
        bool const isok = (m >= 0) && (nnzA >= 0) && (csrRowPtrA != nullptr)
            && (csrEndPtrA != nullptr) && (csrColIndA != nullptr) && (p_isymm != nullptr);

        if(!isok)
        {
            return (HIPSOLVER_STATUS_INVALID_VALUE);
        };
    };

    int issym = 1;

#pragma omp parallel for schedule(dynamic)
    for(int irow = 0; irow < m; irow++)
    {
        // ----------------------------------------------------
        // no need to check transpose entries if
        // matrix is already known to be not symmetric
        // ----------------------------------------------------
        if(issym == 0)
        {
            break;
        };

        for(int k = csrRowPtrA[irow]; k < csrEndPtrA[irow]; k++)
        {
            int const kcol = csrColIndA[k];
            int const krow = kcol;

            // ----------------------------
            // search entries in row "krow"
            // ----------------------------
            bool has_found = false;
            for(int j = csrRowPtrA[krow]; j < csrEndPtrA[krow]; j++)
            {
                int const jcol = csrColIndA[j];
                int const jrow = jcol;

                has_found = (jrow == irow);
                if(has_found)
                {
                    break;
                };
            };

            if(!has_found)
            {
// ----------------------------------------------------
// transpose entry not found so matrix is not symmetric
// ----------------------------------------------------
#pragma omp atomic
                {
                    issym = 0;
                };
#pragma omp flush

                break;
            };
        };

#pragma omp barrier
        *p_isymm = isymm;
        return (HIPSOLVER_STATUS_SUCCESS);
#else
    return (HIPSOLVER_STATUS_SUCCESS);
#endif
    }
}
