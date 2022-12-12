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
#ifndef GEBLTTRS_NPVT_BF_HPP
#define GEBLTTRS_NPVT_BF_HPP
/*
! % ------------------------------------------------
! % Perform forward and backward solve
! %
! %
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A1, B2, C2     ] = [ A1 D2      ] * [    I  U2    ]
! % [    A2, B3, C3 ]   [    A2 D3   ]   [       I  U3 ]
! % [        A3, B4 ]   [       A3 D4]   [          I4 ]
! %
! % ----------------------
! % Solve L * U * x = brhs
! % (1) Solve L * y = brhs,
! % (2) Solve U * x = y
! % ----------------------
*/

#include "geblt_common.h"

#include "gemm_nn_bf.hpp"
#include "getrf_npvt_bf.hpp"
#include "getrs_npvt_bf.hpp"

template <typename T>
GLOBAL_FUNCTION void geblttrs_npvt_bf_kernel(
                                             const rocblas_int  nb,
                                             const rocblas_int  nblocks,
                                             const rocblas_int  nrhs,
                                             T * A_,
                                             const rocblas_int  lda,
                                             T *  D_,
                                             const rocblas_int  ldd,
                                             T *  U_,
                                             const rocblas_int  ldu,
                                             T* brhs_,
                                             const rocblas_int  ldbrhs, 
                                             const rocblas_int  batch_count
                                             )
{
// note adjust indexing for array A
#define A(iv, ia, ja, iblock) A_[indx4f(iv, ia, ja, ((iblock)-1), batch_count, lda, nb)]

#define D(iv, id, jd, iblock) D_[indx4f(iv, id, jd, iblock, batch_count, ldd, nb)]
#define U(iv, iu, ju, iblock) U_[indx4f(iv, iu, ju, iblock, batch_count, ldu, nb)]
#define brhs(iv, ib, iblock, irhs) brhs_[indx4f(iv, ib, iblock, irhs, batch_count, ldbrhs, nblocks)]

#define x(iv, i, j, k) brhs(iv, i, j, k)
#define y(iv, i, j, k) brhs(iv, i, j, k)

    rocblas_int info = 0;
    rocblas_int const ldx = ldbrhs;
    rocblas_int const ldy = ldbrhs;
    /*
! 
! % forward solve
! % --------------------------------
! % [ D1          ]   [ y1 ]   [ b1 ]
! % [ A2 D2       ] * [ y2 ] = [ b2 ]
! % [    A3 D3    ]   [ y3 ]   [ b3 ]
! % [       A4 D4 ]   [ y4 ]   [ b4 ]
! % --------------------------------
! %
! % ------------------
! % y1 = D1 \ b1
! % y2 = D2 \ (b2 - A2 * y1)
! % y3 = D3 \ (b3 - A3 * y2)
! % y4 = D4 \ (b4 - A4 * y3)
! % ------------------
! 
! 
*/

    /*
! for k=1:nblocks,
!     if ((k-1) >= 1),
!       y(:,1:nb,k,:) = y(:,1:nb,k,:) - A(:,1:nb,1:nb,k) * y(:,1:nb,k-1,:);
!     end;
!     if (use_getrf_npvt),
!      LU = D(1:nb,1:nb,k);
!      y(:,1:nb,k,:) = getrs_npvt( LU, y(:,1:nb,k,:) );
!     else
!      y(:,1:nb,k,:) = D(1:nb,1:nb,k) \ y(:,1:nb,k,:);
!     end;
! end;
*/

    for(rocblas_int k = 1; k <= nblocks; k++)
    {
        if((k - 1) >= 1)
        {
            /*
!         ----------------------------------------------------
!         y(:,1:nb,k,:) = y(:,1:nb,k,:) - A(1:nb,1:nb,k) * y(:,1:nb,k-1,:);
!         ----------------------------------------------------
*/
            rocblas_int const iv = 1;
            rocblas_int const mm = nb;
            rocblas_int const nn = nrhs;
            rocblas_int const kk = nb;
            T const alpha = -1;
            T const beta = 1;
            rocblas_int const ld1 = lda;
            rocblas_int const ld2 = ldy * nblocks;
            rocblas_int const ld3 = ldy * nblocks;

            T* Ap = &(A(iv, 1, 1, k));
            T* Bp = &(y(iv, 1, k - 1, 1));
            T* Cp = &(y(iv, 1, k, 1));
            gemm_nn_bf_device<T>(batch_count, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
        };

        /*
!      ----------------------------------------------------
!      y(:,1:nb,k,:) = getrs_npvt( D(:,1:nb,1:nb,k), y(:,1:nb,k,:) );
!      ----------------------------------------------------
*/
        {
            rocblas_int const iv = 1;
            rocblas_int const nn = nb;
            rocblas_int const ld1 = ldd;
            rocblas_int const ld2 = ldbrhs * nblocks;

            T* Ap = &(D(iv, 1, 1, k));
            T* Bp = &(y(iv, 1, k, 1));
            rocblas_int linfo = 0;
            getrs_npvt_bf<T>(batch_count, nn, nrhs, Ap, ld1, Bp, ld2, &linfo);

            info = (linfo != 0) && (info == 0) ? (k - 1) * nb + linfo : info;
        };

    }; // end for  k

    SYNCTHREADS;
    /*
! 
! % backward solve
! % ---------------------------------
! % [ I  U1       ]   [ x1 ]   [ y1 ]
! % [    I  U2    ] * [ x2 ] = [ y2 ]
! % [       I  U3 ]   [ x3 ]   [ y3 ]
! % [          I  ]   [ x4 ]   [ y4 ]
! % ---------------------------------
! % 
! % x4 = y4
! % x3 = y3 - U3 * y4
! % x2 = y2 - U2 * y3
! % x1 = y1 - U1 * y2
! %
! 
! x = zeros(nb,nblocks);
! for kr=1:nblocks,
!   k = nblocks - kr+1;
!   if (k+1 <= nblocks),
!     y(:,1:nb,k,:) = y(:,1:nb,k,:) - U(1:nb,1:nb,k) * x(:,1:nb,k+1,:);
!   end;
!   x(:,1:nb,k,:) = y(:,1:nb,k,:);
! end;
! 
*/
    for(rocblas_int kr = 1; kr <= nblocks; kr++)
    {
        rocblas_int const k = nblocks - kr + 1;
        if(k + 1 <= nblocks)
        {
            /*
!     ----------------------------------------------------------
!     y(:,1:nb,k,:) = y(:,1:nb,k,:) - U(:,1:nb,1:nb,k) * x(:,1:nb,k+1,:);
!     ----------------------------------------------------------
*/

            rocblas_int const iv = 1;
            rocblas_int const mm = nb;
            rocblas_int const nn = nrhs;
            rocblas_int const kk = nb;
            T const alpha = -1;
            T const beta = 1;
            rocblas_int const ld1 = ldu;
            rocblas_int const ld2 = ldx * nblocks;
            rocblas_int const ld3 = ldy * nblocks;

            T* Ap = &(U(iv, 1, 1, k));
            T* Bp = &(x(iv, 1, k + 1, 1));
            T* Cp = &(y(iv, 1, k, 1));

            gemm_nn_bf_device<T>(batch_count, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
        };
    };

}
#undef x
#undef y

#undef A
#undef D
#undef U
#undef brhs

template <typename T>
rocblas_status geblttrs_npvt_bf_template(
                               rocblas_handle handle,
                               const rocblas_int  nb,
                               const rocblas_int  nblocks,
                               const rocblas_int  nrhs,
                               T * A_,
                               const rocblas_int  lda,
                               T * D_,
                               const rocblas_int  ldd,
                               T * U_,
                               const rocblas_int  ldu,
                               T* brhs_,
                               const rocblas_int  ldbrhs,
                               const rocblas_int batch_count
                               )
{
    hipStream_t stream;
    rocblas_get_stream( handle, &stream );

    int block_dim = GEBLT_BLOCK_DIM;
    int grid_dim = (batch_count + (block_dim - 1)) / block_dim;
    hipLaunchKernelGGL((geblttrs_npvt_bf_kernel<T>), dim3(grid_dim), dim3(block_dim), 0, stream,

                       nb, nblocks, nrhs, A_, lda, D_, ldd, U_, ldu, brhs_, ldbrhs, batch_count
                       );


  return( rocblas_status_success );
}

#endif
