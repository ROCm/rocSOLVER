/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

/*************************************************************
    Launchers of specilized kernels
*************************************************************/

template <bool CONJ, typename T, typename U>
void rocsolver_ger(rocblas_handle handle,
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

    rocblasCall_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y, shiftY,
                             incy, strideY, A, shiftA, lda, strideA, batch_count, work);
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <bool CONJ, typename T, typename U>
inline void rocsolver_ger(rocblas_handle handle,
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
    rocsolver_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y, shiftY, incy,
                           strideY, A, shiftA, 1, lda, strideA, batch_count, work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GER(CONJ, T, U)                                           \
    template void rocsolver_ger<CONJ, T, U>(                                  \
        rocblas_handle handle, rocblas_int m, rocblas_int n, const T* alpha,  \
        rocblas_stride stridea, U x, rocblas_stride shiftX, rocblas_int incx, \
        rocblas_stride strideX, U y, rocblas_stride shiftY, rocblas_int incy, \
        rocblas_stride strideY, U A, rocblas_stride shiftA, rocblas_int lda,  \
        rocblas_stride strideA, rocblas_int batch_count, T** work)
