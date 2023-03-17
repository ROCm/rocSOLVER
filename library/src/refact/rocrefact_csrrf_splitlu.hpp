/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

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
    


    // ---------------------------------
    // use a single block for simplicity
    // ---------------------------------
    bool const is_root_block = (blockIdx.x == 0) &&
                               (blockIdx.y == 0) &&
                               (blockIdx.z == 0);
    if (!is_root_block) { return; };
    
    rocblas_int* const nzLp = work;
    rocblas_int* const nzUp = work + n;

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // -------------------------------------------------



    rocblas_int const nthreads = blockDim.x;
    rocblas_int const my_thread = threadId.x;
    rocblas_int const i_start = my_thread;
    rocblas_int const i_inc = nthreads ;

    for(int i = i_start; i < n; i += i_inc)
    {
        nzLp[i] = 0;
        nzUp[i] = 0;
    };
    __syncthreads();

    rocblas_int const nb = (n + (nthreads-1))/nthreads;
    rocblas_int const irow_start = my_thread * nb;
    rocblas_int const irow_end = min( n, irow_start + nb );

    rocblas_int nnzL = 0;
    rocblas_int nnzU = 0;
    rocblas_int constexpr MAX_THREADS = 1024;
    __shared__ isum_nnzL[ MAX_THREADS ];
    __shared__ isum_nnzU[ MAX_THREADS ];

    for(rocblas_int ithread=0; ithread < nthreads; ithread++) 
    {
      for(rocblas_int irow=irow_start; irow < irow_end; irow++) {

        rocblas_int const istart = Mp[irow];
        rocblas_int const iend = Mp[irow + 1];
        rocblas_int const nz = (iend - istart);

        rocblas_int nzU = 0;
        for(int k = istart; k < iend; k++)
        {
            rocblas_int const kcol = Mi[k];
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
        }; // end for irow
    }; // end for ithread

    isum_nnzL[ ithread ] = nnzL;
    isum_nnzU[ ithread ] = nnzU;

    

    __syncthreads();

    nnzL = 0;
    nnzU = 0;
    if (is_root_thread) {
      for(rocblas_int ithread=0; ithread < nthreads; ithread++) {
        nnzL += isum_nnzL[ ithread ];
        nnzU += isum_nnzU[ ithread ];
       };

      Up[n] = nnzU;
      Lp[n] = nnzL;
      };

    __syncthreads();

    // ------------------------------------
    // prefix sum scan to setup Lp and Up
    // ------------------------------------
   
    if (is_root_thread) {
      
      rocblas_int ipos = 0;
      
      for(rocblas_int ithread=0; ithread < nthreads; ithread++) {
         rocblas_int const nz = isum_nnzL[ ithread ];
         isum_nnzL[ ithread ] = ipos;
         ipos += nz;
         };

      };
       
    __syncthreads();


    rocblas_int iL = isum_nnzL[ mythread ];
    rocblas_int iU = isum_nnzU[ mythread ];

    for(rocblas_int irow = irow_start; irow < irow_end; irow++)
    {
        rocblas_int const nzL = nzLp[irow];
        rocblas_int const nzU = nzUp[irow];
        Lp[irow] = iL;
        iL += nzL;

        Up[irow] = iU;
        iU += nzU;
    }
    __syncthreads();



    // ---------------------------------------------------
    // second pass to populate  Li[], Lx[], Ui[], Ux[]
    // ---------------------------------------------------



    // -----------------------------------------------
    // reuse array nzLp[] as pointers into  Lx[], Li[]
    // reuse array nzUp[] as pointers into  Ux[], Ui[]
    // -----------------------------------------------

    for(rocblas_int i = i_start; i < n; i += i_inc)
    {
        rocblas_int const irow = i;

        nzLp[irow] = Lp[irow];
        nzUp[irow] = Up[irow];
    }

    __syncthreads();

    T const one = 1;

    for(rocblas_int i=i_start; i < n; i += i_inc )
    {
        rocblas_int const irow = i;
        rocblas_int const istart = Mp[irow];
        rocblas_int const iend = Mp[irow + 1];

        for(rocblas_int k = istart; k < iend; k++)
        {
            rocblas_int const kcol = Mi[k];
            T const mij = Mx[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                rocblas_int const ip = nzUp[irow];
                nzUp[irow]++;

                Ui[ip] = kcol;
                Ux[ip] = mij;
            }
            else
            {
                rocblas_int const ip = nzLp[irow];
                nzLp[irow]++;

                Li[ip] = kcol;
                Lx[ip] = mij;
            }
        }
    };

    __syncthreads();

    // ------------------------
    // set unit diagonal entry in L
    // ------------------------
 
    for(rocblas_int i=i_start; i < n; i += i_inc) 
    {
        rocblas_int const irow = i;
        rocblas_int const kend = Lp[irow + 1];
        rocblas_int const ip = kend - 1;
        Li[ip] = irow;
        Lx[ip] = one;
    };
    __syncthreads();
}

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
    if (handle == nullptr) {
       return rocblas_status_invalid_handle;
       };

    // 2. invalid size
    if(n < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzT && (!indT || !valT || !indU || !valU || !indL || !valL)))
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
    ROCBLAS_CHECK( 
       rocblas_get_stream(handle, &stream),
       rocblas_status_internal_error);


    rocblas_int const nthreads = 1024;
    rocblas_int const nblocks = 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream,
                            n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);


    return rocblas_status_success;
}
