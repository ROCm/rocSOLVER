/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution
 * ************************************************************************ */

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

/** Call this kernel with 'batch_count' groups in z, and enough
    groups in x and y to cover all the 'm' rows and 'n' columns of C. **/
template <typename T, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void ger_kernel(rocblas_int m,
                                 rocblas_int n,
                                 V alpha,
                                 rocblas_stride stridea,
                                 U1 xx,
                                 rocblas_stride shiftX,
                                 rocblas_int incx,
                                 rocblas_stride strideX,
                                 U2 yy,
                                 rocblas_stride shiftY,
                                 rocblas_int incy,
                                 rocblas_stride strideY,
                                 U3 AA,
                                 rocblas_stride shiftA,
                                 rocblas_int inca,
                                 rocblas_int lda,
                                 rocblas_stride strideA)
{
    // indices
    int bid = hipBlockIdx_z;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // batch instance
    T a = load_scalar(alpha, bid, stridea);
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* x = load_ptr_batch(xx, bid, shiftX, strideX);
    T* y = load_ptr_batch(yy, bid, shiftY, strideY);

    if(i < m && j < n)
    {
        A[i * inca + j * lda] += a * x[i * incx] * y[j * incy];
    }
}

/*************************************************************
    Launchers of specialized kernels
*************************************************************/

template <bool CONJ, typename T, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             rocblas_int m,
                             rocblas_int n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             rocblas_int incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             rocblas_int incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             rocblas_int inca,
                             rocblas_int lda,
                             rocblas_stride strideA,
                             rocblas_int batch_count,
                             T** work)
{
    ROCSOLVER_ENTER("ger", "m:", m, "n:", n, "shiftX:", shiftX, "incx:", incx, "shiftY:", shiftY,
                    "incy:", incy, "shiftA:", shiftA, "inca:", inca, "lda:", lda, "bc:", batch_count);

    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(inca == 1)
        return rocblasCall_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y,
                                        shiftY, incy, strideY, A, shiftA, lda, strideA, batch_count,
                                        work);

    // TODO: add interleaved support for conjugation
    if(CONJ)
        return rocblas_status_not_implemented;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode pmode;
    rocblas_get_pointer_mode(handle, &pmode);

    // launch specialized kernel
    rocblas_int blocksx = (m - 1) / BS2 + 1;
    rocblas_int blocksy = (n - 1) / BS2 + 1;
    dim3 grid(blocksx, blocksy, batch_count);
    dim3 threads(BS2, BS2, 1);
    if(pmode == rocblas_pointer_mode_device)
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, *alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA);
    }

    return rocblas_status_success;
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <bool CONJ, typename T, typename U>
inline rocblas_status rocsolver_ger(rocblas_handle handle,
                                    rocblas_int m,
                                    rocblas_int n,
                                    const T* alpha,
                                    rocblas_stride stridea,
                                    U x,
                                    rocblas_stride shiftX,
                                    rocblas_int incx,
                                    rocblas_stride strideX,
                                    U y,
                                    rocblas_stride shiftY,
                                    rocblas_int incy,
                                    rocblas_stride strideY,
                                    U A,
                                    rocblas_stride shiftA,
                                    rocblas_int lda,
                                    rocblas_stride strideA,
                                    rocblas_int batch_count,
                                    T** work)
{
    return rocsolver_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y, shiftY,
                                  incy, strideY, A, shiftA, 1, lda, strideA, batch_count, work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GER(CONJ, T, U)                                           \
    template rocblas_status rocsolver_ger<CONJ, T, U>(                        \
        rocblas_handle handle, rocblas_int m, rocblas_int n, const T* alpha,  \
        rocblas_stride stridea, U x, rocblas_stride shiftX, rocblas_int incx, \
        rocblas_stride strideX, U y, rocblas_stride shiftY, rocblas_int incy, \
        rocblas_stride strideY, U A, rocblas_stride shiftA, rocblas_int lda,  \
        rocblas_stride strideA, rocblas_int batch_count, T** work)
