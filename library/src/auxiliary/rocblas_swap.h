
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas/rocblas.h"

// --------------------
// swap strided batched
// --------------------

rocblas_status rocblasCall_swap_strided_batched(rocblas_handle handle,
                                                rocblas_int n,
                                                float* x,
                                                rocblas_int incx,
                                                rocblas_stride stridex,
                                                float* y,
                                                rocblas_int incy,
                                                rocblas_stride stridey,
                                                rocblas_int batch_count)
{
    return (rocblas_sswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count));
}

rocblas_status rocblasCall_swap_strided_batched(rocblas_handle handle,
                                                rocblas_int n,
                                                double* x,
                                                rocblas_int incx,
                                                rocblas_stride stridex,
                                                double* y,
                                                rocblas_int incy,
                                                rocblas_stride stridey,
                                                rocblas_int batch_count)
{
    return (rocblas_dswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count));
}

rocblas_status rocblasCall_swap_strided_batched(rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_float_complex* x,
                                                rocblas_int incx,
                                                rocblas_stride stridex,
                                                rocblas_float_complex* y,
                                                rocblas_int incy,
                                                rocblas_stride stridey,
                                                rocblas_int batch_count)
{
    return (rocblas_cswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count));
}

rocblas_status rocblasCall_swap_strided_batched(rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_double_complex* x,
                                                rocblas_int incx,
                                                rocblas_stride stridex,
                                                rocblas_double_complex* y,
                                                rocblas_int incy,
                                                rocblas_stride stridey,
                                                rocblas_int batch_count)
{
    return (rocblas_zswap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count));
}

// ------------
// swap batched
// ------------
rocblas_status rocblasCall_swap_batched(rocblas_handle handle,
                                        rocblas_int n,
                                        float* const x[],
                                        rocblas_int incx,
                                        float* const y[],
                                        rocblas_int incy,
                                        rocblas_int batch_count)
{
    return (rocblas_sswap_batched(handle, n, x, incx, y, incy, batch_count));
}

rocblas_status rocblasCall_swap_batched(rocblas_handle handle,
                                        rocblas_int n,
                                        double* const x[],
                                        rocblas_int incx,
                                        double* const y[],
                                        rocblas_int incy,
                                        rocblas_int batch_count)
{
    return (rocblas_dswap_batched(handle, n, x, incx, y, incy, batch_count));
}

rocblas_status rocblasCall_swap_batched(rocblas_handle handle,
                                        rocblas_int n,
                                        rocblas_float_complex* const x[],
                                        rocblas_int incx,
                                        rocblas_float_complex* const y[],
                                        rocblas_int incy,
                                        rocblas_int batch_count)
{
    return (rocblas_cswap_batched(handle, n, x, incx, y, incy, batch_count));
}

rocblas_status rocblasCall_swap_batched(rocblas_handle handle,
                                        rocblas_int n,
                                        rocblas_double_complex* const x[],
                                        rocblas_int incx,
                                        rocblas_double_complex* const y[],
                                        rocblas_int incy,
                                        rocblas_int batch_count)
{
    return (rocblas_zswap_batched(handle, n, x, incx, y, incy, batch_count));
}

// ----------------
// swap non-batched
// ----------------

rocblas_status rocblasCall_swap(rocblas_handle handle,
                                rocblas_int n,
                                float* x,
                                rocblas_int incx,
                                float* y,
                                rocblas_int incy)
{
    return (rocblas_sswap(handle, n, x, incx, y, incy));
}

rocblas_status rocblasCall_swap(rocblas_handle handle,
                                rocblas_int n,
                                double* x,
                                rocblas_int incx,
                                double* y,
                                rocblas_int incy)
{
    return (rocblas_dswap(handle, n, x, incx, y, incy));
}

rocblas_status rocblasCall_swap(rocblas_handle handle,
                                rocblas_int n,
                                rocblas_float_complex* x,
                                rocblas_int incx,
                                rocblas_float_complex* y,
                                rocblas_int incy)
{
    return (rocblas_cswap(handle, n, x, incx, y, incy));
}

rocblas_status rocblasCall_swap(rocblas_handle handle,
                                rocblas_int n,
                                rocblas_double_complex* x,
                                rocblas_int incx,
                                rocblas_double_complex* y,
                                rocblas_int incy)
{
    return (rocblas_zswap(handle, n, x, incx, y, incy));
}
