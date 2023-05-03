/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

/*************************************************************
    Launchers of specilized kernels
*************************************************************/

template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_gemm(rocblas_handle handle,
                    rocblas_operation transA,
                    rocblas_operation transB,
                    rocblas_int m,
                    rocblas_int n,
                    rocblas_int k,
                    const T* alpha,
                    U A,
                    rocblas_stride shiftA,
                    rocblas_int inca,
                    rocblas_int lda,
                    rocblas_stride strideA,
                    U B,
                    rocblas_stride shiftB,
                    rocblas_int incb,
                    rocblas_int ldb,
                    rocblas_stride strideB,
                    const T* beta,
                    U C,
                    rocblas_stride shiftC,
                    rocblas_int incc,
                    rocblas_int ldc,
                    rocblas_stride strideC,
                    rocblas_int batch_count,
                    T** work)
{
    ROCSOLVER_ENTER("gemm", "transA:", transA, "transB:", transB, "m:", m, "n:", n, "k:", k,
                    "shiftA:", shiftA, "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb,
                    "ldb:", ldb, "shiftC:", shiftC, "incc:", incc, "ldc:", ldc, "bc:", batch_count);

    rocblasCall_gemm(handle, transA, transB, m, n, k, alpha, A, shiftA, lda, strideA, B, shiftB,
                     ldb, strideB, beta, C, shiftC, ldc, strideC, batch_count, work);

    /** This would be the call to the internal gemm, leaving it
            commented here until we are sure it won't be needed **/
    /*dimx = std::min({mm, (4096 / jb) / 2, 32});
        dimy = std::min({nn, (4096 / jb) / 2, 32});
        blocks = (mm - 1) / dimx + 1;
        blocksy = (nn - 1) / dimy + 1;
        grid = dim3(blocks, blocksy, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = jb * (dimx + dimy) * sizeof(T);
        hipLaunchKernelGGL(gemm_kernel<T>, grid, threads, lmemsize, stream, mm,
                           nn, jb, A, shiftA + idx2D(nextpiv, j, lda),
                           shiftA + idx2D(j, nextpiv, lda),
                           shiftA + idx2D(nextpiv, nextpiv, lda), lda, strideA);*/
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <bool BATCHED, bool STRIDED, typename T, typename U>
inline void rocsolver_gemm(rocblas_handle handle,
                           rocblas_operation transA,
                           rocblas_operation transB,
                           rocblas_int m,
                           rocblas_int n,
                           rocblas_int k,
                           const T* alpha,
                           U A,
                           rocblas_stride shiftA,
                           rocblas_int lda,
                           rocblas_stride strideA,
                           U B,
                           rocblas_stride shiftB,
                           rocblas_int ldb,
                           rocblas_stride strideB,
                           const T* beta,
                           U C,
                           rocblas_stride shiftC,
                           rocblas_int ldc,
                           rocblas_stride strideC,
                           rocblas_int batch_count,
                           T** work)
{
    rocsolver_gemm<BATCHED, STRIDED, T>(handle, transA, transB, m, n, k, alpha, A, shiftA, 1, lda,
                                        strideA, B, shiftB, 1, ldb, strideB, beta, C, shiftC, 1,
                                        ldc, strideC, batch_count, work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GEMM(BATCHED, STRIDED, T, U)                                                   \
    template void rocsolver_gemm<BATCHED, STRIDED, T, U>(                                          \
        rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m,  \
        rocblas_int n, rocblas_int k, const T* alpha, U A, rocblas_stride shiftA, rocblas_int lda, \
        rocblas_stride strideA, U B, rocblas_stride shiftB, rocblas_int ldb,                       \
        rocblas_stride strideB, const T* beta, U C, rocblas_stride shiftC, rocblas_int ldc,        \
        rocblas_stride strideC, rocblas_int batch_count, T** work)
