/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rfinfo.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "rocblas_check.h"
#include "rocsparse_check.h"

#ifdef NDEBUG
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

    // temp storage for solution vector
    *size_temp = sizeof(T) * ldb * nrhs;


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





// ---------------------
// gather operation
// dest[i] = src[ P[i] ]
// ---------------------
template <typename Iint, typename T>
static __global__ void
    rf_gather_kernel(Iint const n, Iint const* const P, T const* const src, T* const dest)
{
    Iint const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const i_inc = blockDim.x * gridDim.x;

    for(Iint i = i_start; i < n; i += i_inc)
    {
        Iint const ip = P[i];
        bool const is_valid = (0 <= ip) && (ip < n);
        if(is_valid)
        {
            dest[i] = src[ip];
        };
    };
}

template <typename Iint, typename T>
static void
    rf_gather(hipStream_t streamId, Iint const n, Iint const* const P, T const* const src, T* const dest)
{
    int const nthreads = 128;
    int const nblocks = (n + (nthreads - 1)) / nthreads;

     rf_gather_kernel<<<dim3(nblocks), dim3(nthreads), 0, streamId>>>(n, P, src, dest);
                             
}






template <typename Iint, typename Ilong, typename T>
rocblas_status rf_lusolve(rocsolver_rfinfo  rfinfo,
                             Iint const n,
                             Ilong const nnz,
                             Ilong* const d_LUp,
                             Iint* const d_LUi,
                             T* const d_LUx,
                             T* const d_b,
                             T* const d_Temp,
                             void* work)
{

    Ilong const nnzLU = nnz;
    {
        bool isok = (rfinfo != nullptr);
        if(!isok)
        {
            return (rocblas_status_internal_error);
        };
    }

    Iint const m = n;
    {
        bool const isok_scalar = (n >= 0) && (nnz >= 0);
        if (!isok_scalar) { 
            return( rocblas_status_invalid_value );
            };

        bool const isok_arg = (d_LUp != nullptr) && (d_LUi != nullptr) && (d_LUx != nullptr)
            && (d_b != nullptr) && (d_Temp != nullptr) && (work != nullptr);

        if (!isok_arg) 
        {
            return( rocblas_status_invalid_pointer); 
        };
    };


    rocblas_status istat_return = rocblas_status_success;
    try
    {
        rocsparse_handle const  sphandle = rfinfo->sphandle;

        rocsparse_mat_descr const descrL = rfinfo->descrL;
        rocsparse_mat_descr const descrU = rfinfo->descrU;

        rocsparse_mat_info const infoL = rfinfo->infoL;
        rocsparse_mat_info const infoU = rfinfo->infoU;
         
        void* const buffer = work;


        rocsparse_solve_policy const solve_policy = rfinfo->solve_policy;
        rocsparse_analysis_policy const analysis_policy = rfinfo->analysis_policy;

        rocsparse_operation const transL = rocsparse_operation_none;
        rocsparse_operation const transU = rocsparse_operation_none;


        T* const d_y = d_Temp;

        T* const d_x = d_b;

        // -------------------------------------------
        // If A = LU
        // Solve A x = (LU) x b as
        // (1)   solve L y = b,   L unit diagonal
        // (2)   solve U x = y,   U non-unit diagonal
        // -------------------------------------------

        rocblas_int nnzL = nnzLU;
        rocblas_int nnzU = nnzLU;


        TRACE();

        THROW_IF_ROCSPARSE_ERROR( 
        rocsparseCall_csrsv_analysis( 
                   sphandle,
                   transL, 
                   n, 
                   nnzL,
                   descrL,
                   d_LUx,
                   d_LUp,
                   d_LUi,
                   infoL,
                   analysis_policy,
                   solve_policy, 
                   buffer )  );
                   
        TRACE();

        THROW_IF_ROCSPARSE_ERROR( 
        rocsparseCall_csrsv_analysis( 
                   sphandle,
                   transU, 
                   n, 
                   nnzU,
                   descrU,
                   d_LUx,
                   d_LUp,
                   d_LUi,
                   infoU,
                   analysis_policy,
                   solve_policy, 
                   buffer )   );

        TRACE();

        // ----------------------
        // step (1) solve L y = b
        // ----------------------

        T alpha = 1.0;


        THROW_IF_ROCSPARSE_ERROR(
        rocsparseCall_csrsv_solve( sphandle,
                                   transL,
                                   n,
                                   nnzL,
                                   &alpha,
                                   descrL,
                                   d_LUx,
                                   d_LUp,
                                   d_LUi,
                                   infoL,
                                   d_b,
                                   d_y,
                                   solve_policy,
                                   buffer ) );




        // ----------------------
        // step (2) solve U x = y
        // ----------------------


        TRACE();
 
   
        THROW_IF_ROCSPARSE_ERROR(
        rocsparseCall_csrsv_solve( sphandle,
                                   transU,
                                   n,
                                   nnzU,
                                   &alpha,
                                   descrU,
                                   d_LUx,
                                   d_LUp,
                                   d_LUi,
                                   infoU,
                                   d_y,
                                   d_x,
                                   solve_policy,
                                   buffer ) );




        TRACE();



    }
    catch(const std::bad_alloc& e)
    {
        istat_return = rocblas_status_memory_error;
    }
    catch(const std::runtime_error& e)
    {
        istat_return = rocblas_status_internal_error;
    }
    catch(...)
    {
        istat_return = rocblas_status_internal_error;
    };



        TRACE();

    return (istat_return);
}


template <typename Iint, typename Ilong, typename T>
static rocblas_status rf_pqrlusolve(rocsolver_rfinfo rfinfo,
                                       Iint const n,
                                       Ilong const nnzLU,
                                       Iint* const P_new2old,
                                       Iint* const Q_old2new,
                                       Ilong* const LUp,
                                       Iint* const LUi,
                                       T* const LUx, /* LUp,LUi,LUx  are in CSR format */
                                       T* const brhs,
                                       T* Temp, 
                                       void *work)
{
    int constexpr idebug = 1;
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
        bool const isok_scalar = (n >= 0) && (nnzLU >= 0);
        if (!isok_scalar) {
            return( rocblas_status_invalid_value );
            };
        bool const isok_arg = (LUp != nullptr) && (LUi != nullptr)
            && (LUx != nullptr) && (brhs != nullptr) && (Temp != nullptr) && (work != nullptr) &&
            (P_new2old != nullptr) && (Q_old2new != nullptr); 
        if(!isok_arg)
        {
            return (rocblas_status_invalid_pointer);
        };

        bool const has_work = (n >= 1) && (nnzLU >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        };
    };


    rocblas_status istat_return = rocblas_status_success;
    try
    {

        hipStream_t stream;

        THROW_IF_ROCSPARSE_ERROR(
        rocsparse_get_stream( rfinfo->sphandle, &stream ) );


        T* const d_brhs = brhs;
        T* const d_bhat = Temp;

    TRACE();

        {
            // ------------------------------
            // bhat[k] = brhs[ P_new2old[k] ]
            // ------------------------------

            rf_gather(stream, n, P_new2old, d_brhs, d_bhat);

        }

    TRACE();


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
                = rf_lusolve(rfinfo, n, nnz, d_LUp, d_LUi, d_LUx, d_bhat, d_Temp,work);
            bool const isok_lusolve = (istat_lusolve == rocblas_status_success);
            RF_ASSERT(isok_lusolve);
        };

    TRACE();
        {
            // -------------------------------
            // brhs[ Q_new2old[i] ] = bhat[i]
            // or
            // brhs[ i ] = bhat[ Q_old2new[i] ]
            // -------------------------------
            rf_gather(stream, n, Q_old2new, d_bhat, d_brhs);
        }

    TRACE();
    }
    catch(const std::bad_alloc& e)
    {
        istat_return = rocblas_status_memory_error;
    }
    catch(...)
    {
        istat_return = rocblas_status_internal_error;
    };

    TRACE();

    return (istat_return);
}


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
    ROCBLAS_CHECK( 
       rocblas_get_stream(handle, &stream),
       rocblas_status_internal_error );

  rocblas_status istat = rocblas_status_success;
  try {
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
        THROW_IF_ROCBLAS_ERROR( 
        rf_pqrlusolve(rfinfo, n, nnzT, pivP, pivQ,  ptrT, indT, valT,
                        B + ldb * irhs, temp + ldb * irhs, work) );
    }

  }
  catch(...)
  {
   istat = rocblas_status_internal_error;
  };

    return istat;
}
