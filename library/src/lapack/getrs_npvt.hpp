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
#ifndef GETRS_NPVT_HPP
#define GETRS_NPVT_HPP

#include "geblt_common.h"

template <typename T, typename I>
DEVICE_FUNCTION void
    getrs_npvt_device(
           const I  n, 
           const I  nrhs, 
           T *  A_, 
           const I  lda, 
           T* B_, 
           const I  ldb, 
           I* pinfo)
{
#define A(ia, ja) A_[indx2f(ia, ja, lda)]
#define B(ib, jb) B_[indx2f(ib, jb, ldb)]

    T const zero = 0;
    T const one = 1;
    I info = 0;
    /*
!     ---------------------------------------------------
!     Perform forward and backward solve without pivoting
!     ---------------------------------------------------
*/
    /*
! 
! % ------------------------
! % L * (U * X) = B
! % step 1: solve L * Y = B
! % step 2: solve U * X = Y
! % ------------------------
*/

    /*
! 
! 
! % ------------------------------
! % [I         ] [ Y1 ]   [ B1 ]
! % [L21 I     ] [ Y2 ] = [ B2 ]
! % [L31 L21 I ] [ Y3 ]   [ B3 ]
! % ------------------------------
*/

    /*
! 
! for i=1:n,
!   for j=1:(i-1),
!     B(i,1:nrhs) = B(i,1:nrhs) - LU(i,j) * B(j,1:nrhs);
!   end;
! end;
*/

    for(I i = 1; i <= n; i++)
    {
        for(I j = 1; j <= (i - 1); j++)
        {
            for(I k = 1; k <= nrhs; k++)
            {
                B(i, k) = B(i, k) - A(i, j) * B(j, k);
            };
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

    for(I ir = 1; ir <= n; ir++)
    {
        I const i = n - ir + 1;

        for(I j = (i + 1); j <= n; j++)
        {
            for(I k = 1; k <= nrhs; k++)
            {
                B(i, k) = B(i, k) - A(i, j) * B(j, k);
            };
        };

        bool const is_diag_zero = (A(i, i) == zero);
        T const inv_Uii = (is_diag_zero) ? one : one / A(i, i);
        info = is_diag_zero && (info == 0) ? i : info;

        for(I k = 1; k <= nrhs; k++)
        {
            B(i, k) *= inv_Uii;
        };
    };

    {
        *pinfo = info;
    };
};

#undef A
#undef B

#endif
