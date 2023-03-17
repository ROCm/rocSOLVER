/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

template <typename T>
rocblas_status rocsolver_csrrf_analysis_argCheck(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzM,
                                                 rocblas_int* ptrM,
                                                 rocblas_int* indM,
                                                 T valM,
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
    if (handle == nullptr) {
        return rocblas_status_invalid_handle;
        };
    if (rfinfo == nullptr) {
        return rocblas_status_invalid_pointer;
        };

    // 2. invalid size
    if(n < 0 || nnzM < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!rfinfo || !ptrM || !ptrT || (n && (!pivP || !pivQ)) || (nnzM && (!indM || !valM))
       || (nnzT && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_analysis_getMemorySize(const rocblas_int n,
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

    // requirements for solve with L and U, and for incomplete factorization
    // (buffer size is the same for all routines if the sparsity pattern does not change)
    size_t csrilu0_buffer_size = 0;
    size_t csrsv_L_buffer_size = 0;
    size_t csrsv_U_buffer_size = 0;
    size_t csrsv_buffer_size = 0;

    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrilu0_buffer_size(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                          rfinfo->infoT, &csrilu0_buffer_size) );
                 

    rocsparse_operation const trans = rocsparse_operation_none;

    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrsv_buffer_size(rfinfo->sphandle, 
                                    trans,
                                    n,
                                    nnzT,
                                    rfinfo->descrL,
                                    valT,
                                    ptrT,
                                    indT,
                                    rfinfo->infoL,
                                    &csrsv_L_buffer_size)  );
           
    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrsv_buffer_size(rfinfo->sphandle, 
                                    trans,
                                    n,
                                    nnzT,
                                    rfinfo->descrU,
                                    valT,
                                    ptrT,
                                    indT,
                                    rfinfo->infoU,
                                    &csrsv_U_buffer_size)  );

    csrsv_buffer_size = std::max(csrsv_L_buffer_size, csrsv_U_buffer_size );

    *size_work = std::max( csrilu0_buffer_size, csrsv_buffer_size );
      
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_analysis_template(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzM,
                                                 rocblas_int* ptrM,
                                                 rocblas_int* indM,
                                                 U valM,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 U valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo,
                                                 void* work)
{
    ROCSOLVER_ENTER("csrrf_analysis", "n:", n, "nnzM:", nnzM, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    

    rocsparse_operation const trans = rocsparse_operation_none;
    // rocsparse_solve_policy solve = rocsparse_solve_policy_auto;
    // rocsparse_analysis_policy analysis = rocsparse_analysis_policy_reuse;

    rocsparse_solve_policy const solve = rfinfo->solve_policy;
    rocsparse_analysis_policy const analysis = rfinfo->analysis_policy;

   try {

    // analysis for incomplete factorization
    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrilu0_analysis(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                    rfinfo->infoT, rocsparse_analysis_policy_force, solve, work) );

    // analysis for solve with L
    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrsv_analysis(rfinfo->sphandle, trans, n, nnzT, rfinfo->descrL, valT, ptrT, indT,
                                    rfinfo->infoL, analysis, solve, work) );

    // analysis for solve with U
    THROW_IF_ROCSPARSE_ERROR(
    rocsparseCall_csrsv_analysis(rfinfo->sphandle, trans, n, nnzT, rfinfo->descrU, valT, ptrT, indT,
                                    rfinfo->infoU, analysis, solve, work) );
    }
   catch( ... )
   {  
    return rocblas_status_internal_error;
   };


    return rocblas_status_success;
}
