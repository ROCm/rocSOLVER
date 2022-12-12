/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GBTRS_NPVT_HPP
#define GBTRS_NPVT_HPP

#include "geblt_common.h"

#include "gemm_nn.hpp"
#include "getrs_npvt.hpp"

template <typename T, typename I>
DEVICE_FUNCTION void geblttrs_npvt_device(I const nb,
                                          I const nblocks,
                                          I const nrhs,
                                          T * A_,
                                          I const lda,
                                          T * D_,
                                          I const ldd,
                                          T * U_,
                                          I const ldu,
                                          T* brhs_,
                                          I const ldbrhs,
                                          I* pinfo)

{
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

#define A(ia, ja, iblock) A_[indx3f(ia, ja, iblock, lda, nb)]
#define D(id, jd, iblock) D_[indx3f(id, jd, iblock, ldd, nb)]
#define U(iu, ju, iblock) U_[indx3f(iu, ju, iblock, ldu, nb)]

/*
 --------------------------------------
 dimension brhs(ldbrhs, nblocks, nrhs )
 --------------------------------------
*/
#define brhs(i, iblock, irhs) brhs_[indx3f(i, iblock, irhs, ldbrhs, nblocks)]

/*
 -------------------------
 reuse storage x,y in brhs
 -------------------------
*/
#define x(i, j, k) brhs(i, j, k)
#define y(i, j, k) brhs(i, j, k)

    I info = 0;

    I const ldx = ldbrhs;
    I const ldy = ldbrhs;

    T const one = 1;
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
!       y(1:nb,k,:) = y(1:nb,k,:) - A(1:nb,1:nb,k) * y(1:nb,k-1,:);
!     end;
!     if (use_getrf_npvt),
!      LU = D(1:nb,1:nb,k);
!      y(1:nb,k,:) = getrs_npvt( LU, y(1:nb,k,:) );
!     else
!      y(1:nb,k,:) = D(1:nb,1:nb,k) \ y(1:nb,k,:);
!     end;
! end;
*/

    for(I k = 1; k <= nblocks; k++)
    {
        if((k - 1) >= 1)
        {
            /*
!         ----------------------------------------------------
!         y(1:nb,k,1:nrhs) = y(1:nb,k,1:nrhs) - A(1:nb,k,1:nb) * y(1:nb,k-1,1:nrhs);
!         ----------------------------------------------------
*/
            {
                I const mm = nb;
                I const nn = nrhs;
                I const kk = nb;

                T const alpha = -one;
                T const beta = one;

                I const ld1 = lda;
                I const ld2 = ldy * nblocks;
                I const ld3 = ldy * nblocks;

                gemm_nn_device(mm, nn, kk, alpha, &(A(1, 1, k)), ld1, &(y(1, k - 1, 1)), ld2, beta,
                               &(y(1, k, 1)), ld3);
            };
        };

        /*
!      ----------------------------------------------------
!      y(1:nb,k,1:nrhs) = getrs_npvt( D(1:nb,1:nb,k), y(1:nb,k,1:nrhs) );
!      ----------------------------------------------------
*/
        {
            I const nn = nb;
            I const ld1 = ldd;
            I const ld2 = ldy * nblocks;
            I linfo = 0;

            getrs_npvt_device(nn, nrhs, &(D(1, 1, k)), ld1, &(y(1, k, 1)), ld2, &linfo);
            info = (linfo != 0) && (info == 0) ? (k - 1) * nb + linfo : info;
        };
    };

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
*/

    /*
! 
! x = zeros(nb,nblocks);
! for kr=1:nblocks,
!   k = nblocks - kr+1;
!   if (k+1 <= nblocks),
!     y(1:nb,k,:) = y(1:nb,k,:) - U(1:nb,1:nb,k) * x(1:nb,k+1,:);
!   end;
!   x(1:nb,k,:) = y(1:nb,k,:);
! end;
! 
*/

    for(I kr = 1; kr <= nblocks; kr++)
    {
        I const k = nblocks - kr + 1;
        if((k + 1) <= nblocks)
        {
            /*
!     ----------------------------------------------------------
!     y(1:nb,k,1:nrhs) = y(1:nb,k,1:nrhs) - U(1:nb,1:nb,k) * x(1:nb,k+1,1:nrhs);
!     ----------------------------------------------------------
*/
            I const mm = nb;
            I const nn = nrhs;
            I const kk = nb;

            T const alpha = -one;
            T const beta = one;

            I const ld1 = ldu;
            I const ld2 = ldx * nblocks;
            I const ld3 = ldy * nblocks;

            gemm_nn_device(mm, nn, kk, alpha, &(U(1, 1, k)), ld1, &(x(1, k + 1, 1)), ld2, beta,
                           &(y(1, k, 1)), ld3);
        };
    };

    {
        *pinfo = info;
    };
};
#undef brhs
#undef x
#undef y

#undef A
#undef D
#undef U

#endif
