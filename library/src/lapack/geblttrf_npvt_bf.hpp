/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef GBTRF_NPVT_BF_HPP
#define GBTRF_NPVT_BF_HPP

#include "geblt_common.h"

/*
! ------------------------------------------------------
!     Perform LU factorization without pivoting
!     of block tridiagonal matrix
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A2, B2, C2     ] = [ A2 D2      ] * [    I  U2    ]
! % [    A3, B3, C3 ]   [    A3 D3   ]   [       I  U3 ]
! % [        A4, B4 ]   [       A4 D4]   [          I4 ]
! ------------------------------------------------------
*/
#include "gemm_nn_bf.hpp"
#include "getrf_npvt_bf.hpp"
#include "getrs_npvt_bf.hpp"

template <typename T, typename I>
GLOBAL_FUNCTION void geblttrf_npvt_bf_kernel(I const nb,
                                             I const nblocks,
                                             T* A_,
                                             I const lda,
                                             T* B_,
                                             I const ldb,
                                             T* C_,
                                             I const ldc,
                                             I devinfo_array[],
                                             I batch_count)
{
#define A(iv, ia, ja, k) A_[indx4f(iv, ia, ja, k, batch_count, lda, nb)]
#define B(iv, ib, jb, k) B_[indx4f(iv, ib, jb, k, batch_count, ldb, nb)]
#define C(iv, ic, jc, k) C_[indx4f(iv, ic, jc, k, batch_count, ldc, nb)]

    /*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/

#define D(iv, i, j, k) B(iv, i, j, k)
#define U(iv, i, j, k) C(iv, i, j, k)
    I const ldu = ldc;
    I const ldd = ldb;
    /*
! 
! % B1 = D1
! % D1 * U1 = C1 => U1 = D1 \ C1
! % D2 + A2*U1 = B2 => D2 = B2 - A2*U1
! %
! % D2*U2 = C2 => U2 = D2 \ C2
! % D3 + A3*U2 = B3 => D3 = B3 - A3*U2
! %
! % D3*U3 = C3 => U3 = D3 \ C3
! % D4 + A4*U3 = B4 => D4 = B4 - A4*U3
! idebug = 1;
! 
! % ----------------------------------
! % in actual code, overwrite B with D
! % overwrite C with U
! % ----------------------------------
! D = zeros(nb,nb,nblocks);
! U = zeros(nb,nb,nblocks);
! 
! use_getrf_npvt = 1;
! 
! k = 1;
! D(:,1:nb,1:nb,k) = B(:,1:nb,1:nb,k);
! if (use_getrf_npvt),
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
! end;
*/

    {
        I const iv = 1;
        I const k = 1;
        I const mm = nb;
        I const nn = nb;
        T* Ap = &(D(iv, 1, 1, k));
        /*
!   ----------------------------------------------
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
!   ----------------------------------------------
*/
        getrf_npvt_bf_device<T>(batch_count, mm, nn, Ap, ldd, devinfo_array);
        SYNCTHREADS;
    };

    /*
! 
! 
! 
! for k=1:(nblocks-1),
!    
!    if (use_getrf_npvt),
!     if (idebug >= 2),
!       Ck = C(1:nb,1:nb,k);
!       disp(sprintf('k=%d,size(Ck)=%d,%d ',k,size(Ck,1),size(Ck,2)));
!     end;
! 
!     U(:,1:nb,1:nb,k) = getrs_npvt( D(:,1:nb,1:nb,k), C(1:nb,1:nb,k) );
!    else
!     U(:,1:nb,1:nb,k) = D(:,1:nb,1:nb,k) \ C(1:nb,1:nb,k);
!    end;
! 
!    D(:,1:nb,1:nb,k+1) = B(:,1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(:,1:nb,1:nb,k);
!    if (use_getrf_npvt),
!      D(:,1:nb,1:nb,k+1) = getrf_npvt( D(:,1:nb,1:nb,k+1) );
!    end;
! end;
*/

    for(I k = 1; k <= (nblocks - 1); k++)
    {
        /*
!     --------------------------------------------------------------     
!     U(:,1:nb,1:nb,k) = getrs_npvt( D(:,1:nb,1:nb,k), C(1:nb,1:nb,k) );
!     --------------------------------------------------------------     
*/
        {
            I const nn = nb;
            I const nrhs = nb;
            I const iv = 1;

            T const* const Ap = &(D(iv, 1, 1, k));
            T* Bp = &(C(iv, 1, 1, k));
            getrs_npvt_bf<T>(batch_count, nn, nrhs, Ap, ldd, Bp, ldc, devinfo_array);
            SYNCTHREADS;
        };

        /*
!    ------------------------------------------------------------------------
!    D(:,1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(:,1:nb,1:nb,k);
!    ------------------------------------------------------------------------
*/
        {
            I const iv = 1;
            I const mm = nb;
            I const nn = nb;
            I const kk = nb;
            T const alpha = -1;
            T const beta = 1;
            I const ld1 = lda;
            I const ld2 = ldu;
            I const ld3 = ldd;

            T* Ap = &(A(iv, 1, 1, k + 1));
            T* Bp = &(U(iv, 1, 1, k));
            T* Cp = &(D(iv, 1, 1, k + 1));
            gemm_nn_bf_device<T>(batch_count, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
            SYNCTHREADS;
        };

        /*
!      --------------------------------------------------
!      D(:,1:nb,1:nb,k+1) = getrf_npvt( D(:,1:nb,1:nb,k+1) );
!      --------------------------------------------------
*/

        {
            I const iv = 1;
            I const mm = nb;
            I const nn = nb;
            T* Ap = &(D(iv, 1, 1, k + 1));

            getrf_npvt_bf_device<T>(batch_count, mm, nn, Ap, ldd, devinfo_array);

            SYNCTHREADS;
        };

    }; // end for k
}
#undef D
#undef U

#undef A
#undef B
#undef C

template <typename T, typename I>
rocblas_status geblttrf_npvt_bf_template(rocblas_handle handle,
                                         I const nb,
                                         I const nblocks,
                                         T* A_,
                                         I const lda,
                                         T* B_,
                                         I const ldb,
                                         T* C_,
                                         I const ldc,
                                         I* devinfo_array,
                                         I batch_count)
{
#ifdef USE_GPU
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    {
        void* dst = (void*)&(devinfo_array[0]);
        int value = 0;
        size_t sizeBytes = sizeof(I) * batch_count;

        HIP_CHECK(hipMemsetAsync(dst, value, sizeBytes, stream), rocblas_status_internal_error);
    };

    auto const block_dim = GEBLT_BLOCK_DIM;
    auto const grid_dim = (batch_count + (block_dim - 1)) / block_dim;
    hipLaunchKernelGGL((geblttrf_npvt_bf_kernel<T, I>), dim3(grid_dim), dim3(block_dim), 0, stream,

                       nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batch_count);
#else

    for(I i = 0; i < batch_count; i++)
    {
        devinfo_array[i] = 0;
    };

    geblttrf_npvt_bf_kernel<T, I>(nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batch_count);

#endif

    return (rocblas_status_success);
}

#endif
