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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "geblttrs_npvt_bf.hpp"

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
    if (handle == nullptr) {
         return(rocblas_status_invalid_handle);
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

    return( geblttrs_npvt_bf_template(
                       handle,  nb, nblocks, nrhs, 
                       A, lda, B, ldb, C, ldc,
                       X, ldx,
                       batch_count ) );
}
