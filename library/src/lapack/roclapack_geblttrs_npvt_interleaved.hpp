/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocblas.hpp"
#include "roclapack_geblttrf_npvt_interleaved.hpp"
#include "rocsolver/rocsolver.h"

template <typename T>
GLOBAL_FUNCTION void geblttrs_npvt_bf_kernel(const rocblas_int nb,
                                             const rocblas_int nblocks,
                                             const rocblas_int nrhs,
                                             T* A_,
                                             const rocblas_int lda,
                                             T* D_,
                                             const rocblas_int ldd,
                                             T* U_,
                                             const rocblas_int ldu,
                                             T* brhs_,
                                             const rocblas_int ldbrhs,
                                             const rocblas_int batch_count)
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
    */

    for(rocblas_int k = 1; k <= nblocks; k++)
    {
        if((k - 1) >= 1)
        {
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
    */
    for(rocblas_int kr = 1; kr <= nblocks; kr++)
    {
        rocblas_int const k = nblocks - kr + 1;
        if(k + 1 <= nblocks)
        {
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

#undef x
#undef y

#undef A
#undef D
#undef U
#undef brhs
}

template <typename T>
void rocsolver_geblttrs_npvt_interleaved_getMemorySize(const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int nrhs,
                                                       const rocblas_int batch_count,
                                                       size_t* size_work)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
    {
        // TODO: set workspace sizes to zero
        *size_work = 0;
        return;
    }

    // TODO: calculate workspace sizes
    *size_work = 0;
}

template <typename T>
rocblas_status rocsolver_geblttrs_npvt_interleaved_argCheck(rocblas_handle handle,
                                                            const rocblas_int nb,
                                                            const rocblas_int nblocks,
                                                            const rocblas_int nrhs,
                                                            const rocblas_int lda,
                                                            const rocblas_int ldb,
                                                            const rocblas_int ldc,
                                                            const rocblas_int ldx,
                                                            T A,
                                                            T B,
                                                            T C,
                                                            T X,
                                                            const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || nrhs < 0 || lda < nb || ldb < nb || ldc < nb || ldx < nb
       || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (nb && nblocks && nrhs && !X))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_interleaved_template(rocblas_handle handle,
                                                            const rocblas_int nb,
                                                            const rocblas_int nblocks,
                                                            const rocblas_int nrhs,
                                                            U A,
                                                            const rocblas_int lda,
                                                            U B,
                                                            const rocblas_int ldb,
                                                            U C,
                                                            const rocblas_int ldc,
                                                            U X,
                                                            const rocblas_int ldx,
                                                            const rocblas_int batch_count,
                                                            void* work)
{
    ROCSOLVER_ENTER("geblttrs_npvt_interleaved", "nb:", nb, "nblocks:", nblocks, "nrhs:", nrhs,
                    "lda:", lda, "ldb:", ldb, "ldc:", ldc, "ldx:", ldx, "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int block_dim = GEBLT_BLOCK_DIM;
    int grid_dim = (batch_count + (block_dim - 1)) / block_dim;
    ROCSOLVER_LAUNCH_KERNEL(geblttrs_npvt_bf_kernel, dim3(grid_dim), dim3(block_dim), 0, stream, nb,
                            nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx, batch_count);

    return rocblas_status_success;
}
