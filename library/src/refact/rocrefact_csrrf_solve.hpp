/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#if NDEBUG
#define RF_ASSERT( tcond )
#else
#include <stdexcept>
#define RF_ASSERT( tcond ) { if (!(tcond)) { throw std::runtime_error(__FILE__); } }
#endif


template <typename T>
rocblas_status rocsolver_csrrf_solve_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              const rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              T valT,
                                              rocblas_int* pivP,
                                              rocblas_int* pivQ,
                                              rocsolver_rfinfo rfinfo,
                                              T B,
                                              const rocblas_int ldb)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if (handle == nullptr) {
       return rocblas_status_invalid_handle;
       };

    // 2. invalid size
    if(n < 0 || nrhs < 0 || nnzT < 0 || ldb < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!rfinfo || !ptrT || (n && (!pivP || !pivQ)) || (nnzT && (!indT || !valT)) || (nrhs * n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_solve_getMemorySize(const rocblas_int n,
                                         const rocblas_int nrhs,
                                         const rocblas_int nnzT,
                                         rocblas_int* ptrT,
                                         rocblas_int* indT,
                                         U valT,
                                         rocsolver_rfinfo rfinfo,
                                         const rocblas_int ldb,
                                         size_t* size_work,
                                         size_t* size_temp)
{
    // if quick return, no need of workspace
    if(n == 0 || nrhs == 0)
    {
        *size_work = 0;
        *size_temp = 0;
        return;
    }

    // ----------------------------
    // requirements for csrsv_solve
    // ----------------------------
    THROW_IF_ROCSPARSE_ERROR(	
        rocsparseCall_csrsv_buffer_size(rfinfo->sphandle, rocsparse_operation_none, n, nnzT,
                                        rfinfo->descrT, valT, ptrT, indT, rfinfo->infoT, size_work) );

    // temp storage for solution vector
    *size_temp = sizeof(T) * ldb * nrhs;
}





template <typename Iint, typename Ilong, typename T>
static rocsolverStatus_t rf_pqrlusolve(rocsolverRfHandle_t handle,
                                       Iint const n,
                                       Ilong const nnzLU,
                                       Iint* const P_new2old,
                                       Iint* const Q_new2old,
                                       Ilong* const LUp,
                                       Iint* const LUi,
                                       T* const LUx, /* LUp,LUi,LUx  are in CSR format */
                                       T* const brhs,
                                       T* Temp)
{
    int const idebug = 1;
    /*
    -------------------------------------------------
    (P * A * Q) = LU
    solve A * x = b
       P A Q * (inv(Q) x) = P b
       { (P A Q) } * (inv(Q) x) = (P b)
       
       (LU) xhat = bhat,  xhat = inv(Q) x, or Q xhat = x,
                          bhat = (P b)
    -------------------------------------------------
*/

    {
        bool const isok_arg = (n >= 0) && (nnzLU >= 0) && (LUp != nullptr) && (LUi != nullptr)
            && (LUx != nullptr) && (brhs != nullptr) && (Temp != nullptr) &&
            (P_new2old != nullptr) && (Q_new2old != nullptr); 
        if(!isok_arg)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };

        bool const has_work = (n >= 1) && (nnzLU >= 1);
        if(!has_work)
        {
            return (ROCSOLVER_STATUS_SUCCESS);
        };
    };


    rocblas_status istat_return = rocblas_status_success;
    try
    {

        hipStream_t stream = handle->streamId.data();

        T* const d_brhs = brhs;
        T* const d_bhat = Temp;
        T* const d_Rs = Rs;


        {
            // ------------------------------
            // bhat[k] = brhs[ P_new2old[k] ]
            // ------------------------------

            rf_gather(stream, n, P_new2old, d_brhs, d_bhat);

        }



        // -----------------------------------------------
        // prepare to call triangular solvers rf_lusolve()
        // -----------------------------------------------

        {
            Ilong const nnz = nnzLU;

            // ---------------------------------------
            // allocate device memory and copy LU data
            // ---------------------------------------

            Ilong* const d_LUp = LUp;
            Iint* const d_LUi = LUi;
            T* const d_LUx = LUx;
            T* const d_Temp = Temp;

            rocblas_status const istat_lusolve
                = rf_lusolve(handle, n, nnz, d_LUp, d_LUi, d_LUx, d_bhat, d_Temp);
            bool const isok_lusolve = (istat_lusolve == rocblas_status_success);
            RF_ASSERT(isok_lusolve);
        };

        {
            // -------------------------------
            // brhs[ Q_new2old[i] ] = bhat[i]
            // -------------------------------
            rf_scatter(stream, n, Q_new2old, d_bhat, d_brhs);
        }

    }
    catch(const std::bad_alloc& e)
    {
        istat_return = ROCSOLVER_STATUS_ALLOC_FAILED;
    }
    catch(const std::runtime_error& e)
    {
        istat_return = ROCSOLVER_STATUS_EXECUTION_FAILED;
    }
    catch(...)
    {
        istat_return = ROCSOLVER_STATUS_INTERNAL_ERROR;
    };

    if(idebug >= 1)
    {
        printf("%s line %d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
        fflush(stdout);
    };
    return (istat_return);
}

#endif

template <typename T, typename U>
rocblas_status rocsolver_csrrf_solve_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              const rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              U valT,
                                              rocblas_int* pivP,
                                              rocblas_int* pivQ,
                                              rocsolver_rfinfo rfinfo,
                                              U B,
                                              const rocblas_int ldb,
                                              void* work,
                                              T* temp)
{
    ROCSOLVER_ENTER("csrrf_solve", "n:", n, "nrhs:", nrhs, "nnzT:", nnzT, "ldb:", ldb);

    // quick return
    if(n == 0 || nrhs == 0)
        return rocblas_status_success;

    hipStream_t stream;
    CHECK_ROCBLAS( 
       rocblas_get_stream(handle, &stream),
       rocblas_status_internal_error );


    // ---- main loop (rocsparse does not support multiple rhs) ----
    // -------------------------------------------------------------
    // For each right-hand-side brhs:
    // solve A * x = brhs
    //   P A Q * (inv(Q) x) = P brhs
    //
    //   (LU) xhat = bhat,  xhat = inv(Q) x, or Q xhat = x,
    //                      bhat = P brhs
    // -------------------------------------------------------------
    for(int irhs = 0; irhs < nrhs; irhs++)
    {
        rf_pqrlusolve(handle, n, nnzT, pivP, pivQ,  ptrT, indT, valT,
                        B + ldb * irhs, temp + ldb * irhs);
    }


    return rocblas_status_success;
}
