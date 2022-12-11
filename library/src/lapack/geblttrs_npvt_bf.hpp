/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#pragma once
#ifndef GEBLTTRS_NPVT_BF_HPP
#define GEBLTTRS_NPVT_BF_HPP
/*
! % ------------------------------------------------
! % Perform forward and backward solve
! %
! %
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A2, B2, C2     ] = [ A2 D2      ] * [    I  U2    ]
! % [    A3, B3, C3 ]   [    A3 D3   ]   [       I  U3 ]
! % [        A4, B4 ]   [       A4 D4]   [          I4 ]
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
GLOBAL_FUNCTION void geblttrs_npvt_bf_kernel(rocblas_int const nb,
                                             rocblas_int const nblocks,
                                             rocblas_int const batchCount,
                                             rocblas_int const nrhs,
                                             T const* const A_,
                                             rocblas_int const lda,
                                             T const* const D_,
                                             rocblas_int const ldd,
                                             T const* const U_,
                                             rocblas_int const ldu,
                                             T* brhs_,
                                             rocblas_int const ldbrhs,
                                             rocblas_int* pinfo)
{
#define A(iv, ia, ja, iblock) A_[indx4f(iv, ia, ja, iblock, batchCount, lda, nb)]
#define D(iv, id, jd, iblock) D_[indx4f(iv, id, jd, iblock, batchCount, ldd, nb)]
#define U(iv, iu, ju, iblock) U_[indx4f(iv, iu, ju, iblock, batchCount, ldu, nb)]
#define brhs(iv, ib, iblock, irhs) brhs_[indx4f(iv, ib, iblock, irhs, batchCount, ldbrhs, nblocks)]

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

            T const* const Ap = &(A(iv, 1, 1, k));
            T const* const Bp = &(y(iv, 1, k - 1, 1));
            T* Cp = &(y(iv, 1, k, 1));
            gemm_nn_bf_device<T>(batchCount, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
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

            T const* const Ap = &(D(iv, 1, 1, k));
            T* Bp = &(y(iv, 1, k, 1));
            rocblas_int linfo = 0;
            getrs_npvt_bf<T>(batchCount, nn, nrhs, Ap, ld1, Bp, ld2, &linfo);

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

            T const* const Ap = &(U(iv, 1, 1, k));
            T const* const Bp = &(x(iv, 1, k + 1, 1));
            T* Cp = &(y(iv, 1, k, 1));

            gemm_nn_bf_device<T>(batchCount, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
        };
    };

    if(info != 0)
    {
        *pinfo = info;
    };
}
#undef x
#undef y

#undef A
#undef D
#undef U
#undef brhs

template <typename T>
void geblttrs_npvt_bf_template(hipStream_t stream,

                               rocblas_int const nb,
                               rocblas_int const nblocks,
                               rocblas_int const batchCount,
                               rocblas_int const nrhs,
                               T const* const A_,
                               rocblas_int const lda,
                               T const* const D_,
                               rocblas_int const ldd,
                               T const* const U_,
                               rocblas_int const ldu,
                               T* brhs_,
                               rocblas_int const ldbrhs,
                               rocblas_int* pinfo)
{
#ifdef USE_GPU
    int block_dim = 64;
    int grid_dim = (batchCount + (block_dim - 1)) / block_dim;
    hipLaunchKernelGGL((geblttrs_npvt_bf_kernel<T>), dim3(grid_dim), dim3(block_dim), 0, stream,

                       nb, nblocks, batchCount, nrhs, A_, lda, D_, ldd, U_, ldu, brhs_, ldbrhs,
                       pinfo);

#else
    geblttrs_npvt_bf_kernel<T>(nb, nblocks, batchCount, nrhs, A_, lda, D_, ldd, U_, ldu, brhs_,
                               ldbrhs, pinfo);

#endif
}

#endif
