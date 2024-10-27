/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2013
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

ROCSOLVER_BEGIN_NAMESPACE

template <typename SS, typename U>
rocblas_status rocsolver_lasr_argCheck(rocblas_handle handle,
                                       const rocblas_side side,
                                       const rocblas_pivot pivot,
                                       const rocblas_direct direct,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       SS* C,
                                       SS* S,
                                       U A,
                                       const rocblas_int lda)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(pivot != rocblas_pivot_variable && pivot != rocblas_pivot_top && pivot != rocblas_pivot_bottom)
        return rocblas_status_invalid_value;
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    bool is_side_left = (side == rocblas_side_left);
    bool is_side_right = (side == rocblas_side_right);
    if(m && n && !A)
        return rocblas_status_invalid_pointer;
    if(is_side_left && m > 1 && (!C || !S))
        return rocblas_status_invalid_pointer;
    if(is_side_right && n > 1 && (!C || !S))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename SS, typename U>
rocblas_status rocsolver_lasr_template(rocblas_handle handle,
                                       const rocblas_side side,
                                       const rocblas_pivot pivot,
                                       const rocblas_direct direct,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       SS* C,
                                       const rocblas_int shiftC,
                                       const rocblas_stride strideC,
                                       SS* S,
                                       const rocblas_int shiftS,
                                       const rocblas_stride strideS,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("lasr", "side:", side, "pivot:", pivot, "direct:", direct, "m:", m, "n:", n,
                    "shiftC:", shiftC, "shiftS:", shiftS, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    bool is_side_left = (side == rocblas_side_left);
    bool is_side_right = (side == rocblas_side_right);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;
    if((is_side_left && m < 2) || (is_side_right && n < 2))
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
