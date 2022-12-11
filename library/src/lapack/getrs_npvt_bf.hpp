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
#ifndef GETRS_NPVT_BF_HPP
#define GETRS_NPVT_BF_HPP

#include "geblt_common.h"

template <typename T>
DEVICE_FUNCTION void getrs_npvt_bf(rocblas_int const batchCount,
                                   rocblas_int const n,
                                   rocblas_int const nrhs,
                                   T const* const A_,
                                   rocblas_int const lda,
                                   T* B_,
                                   rocblas_int const ldb,
                                   rocblas_int* pinfo)
{
    /*
!     ---------------------------------------------------
!     Perform forward and backward solve without pivoting
!     ---------------------------------------------------
*/

#define A(iv, ia, ja) A_[indx3f(iv, ia, ja, batchCount, lda)]
#define B(iv, ib, irhs) B_[indx3f(iv, ib, irhs, batchCount, ldb)]

#ifdef USE_GPU
    rocblas_int const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = (gridDim.x * blockDim.x);
#else
    rocblas_int const iv_start = 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = 1;
#endif

    T const one = 1;
    T const zero = 0;

    rocblas_int info = 0;
    /*
! 
! % ------------------------
! % L * (U * X) = B
! % step 1: solve L * Y = B
! % step 2: solve U * X = Y
! % ------------------------
! 
! 
! % ------------------------------
! % [I         ] [ Y1 ]   [ B1 ]
! % [L21 I     ] [ Y2 ] = [ B2 ]
! % [L31 L21 I ] [ Y3 ]   [ B3 ]
! % ------------------------------
! 
! 
! % ------------
! % special case
! % ------------
*/

    /*
! 
! for i=1:n,
!   for j=1:(i-1),
!     B(i,1:nrhs) = B(i,1:nrhs) - LU(i,j) * B(j,1:nrhs);
!   end;
! end;
*/

    for(rocblas_int i = 1; i <= n; i++)
    {
        for(rocblas_int j = 1; j <= (i - 1); j++)
        {
            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    B(iv, i, k) = B(iv, i, k) - A(iv, i, j) * B(iv, j, k);
                };
            };

            SYNCTHREADS;
        };
    };
    /*
! 
! % ------------------------------
! % [U11 U12 U13 ] [ X1 ] = [ Y1 ]
! % [    U22 U23 ]*[ X2 ] = [ Y2 ]
! % [        U33 ]*[ X3 ] = [ Y3 ]
! % ------------------------------
! 
! for ir=1:n,
!   i = n - ir + 1;
!   for j=(i+1):n,
!     B(i,1:nrhs) = B(i,1:nrhs) - LU( i,j) * B(j,1:nrhs);
!   end;
!   B(i,1:nrhs) = B(i,1:nrhs) / LU(i,i);
! end;
! 
*/
    for(rocblas_int ir = 1; ir <= n; ir++)
    {
        rocblas_int i = n - ir + 1;
        for(rocblas_int j = (i + 1); j <= n; j++)
        {
            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    B(iv, i, k) = B(iv, i, k) - A(iv, i, j) * B(iv, j, k);
                };
            };
            SYNCTHREADS;

        }; // end for j

        for(rocblas_int iv = 1; iv <= iv_end; iv += iv_inc)
        {
            T const A_iv_i_i = A(iv, i, i);
            bool const is_diag_zero = (std::abs(A_iv_i_i) == zero);
            info = (is_diag_zero && (info == 0)) ? i : info;

            T const inv_Uii_iv = (is_diag_zero) ? one : one / A_iv_i_i;

            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                B(iv, i, k) = B(iv, i, k) * inv_Uii_iv;
            };

        }; // end for iv

        SYNCTHREADS;

    }; // end for ir

    *pinfo = info;
}

#undef A
#undef B

#endif
