/*****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void conj_in_place(const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int shifta,
                                    const rocblas_int lda,
                                    const rocblas_stride stridea)
{
    // do nothing
}

template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void conj_in_place(const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int shifta,
                                    const rocblas_int lda,
                                    const rocblas_stride stridea)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int b = hipBlockIdx_z;

    T* Ap = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < m && j < n)
        Ap[i + j * lda] = conj(Ap[i + j * lda]);
}

template <typename T>
rocblas_status
    rocsolver_lacgv_argCheck(rocblas_handle handle, const rocblas_int n, const rocblas_int incx, T x)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || !incx)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(n && !x)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_lacgv_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U x,
                                        const rocblas_int shiftx,
                                        const rocblas_int incx,
                                        const rocblas_stride stridex,
                                        const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("lacgv", "n:", n, "shiftX:", shiftx, "incx:", incx, "bc:", batch_count);

    // quick return
    if(n == 0 || !batch_count || !COMPLEX)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // handle negative increments
    rocblas_int offset = incx < 0 ? shiftx - (n - 1) * incx : shiftx;

    // conjugate x
    rocblas_int blocks = (n - 1) / 64 + 1;
    ROCSOLVER_LAUNCH_KERNEL(conj_in_place<T>, dim3(1, blocks, batch_count), dim3(1, 64, 1), 0,
                            stream, 1, n, x, offset, incx, stridex);

    return rocblas_status_success;
}
