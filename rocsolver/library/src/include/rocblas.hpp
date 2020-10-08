/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef _ROCBLAS_HPP_
#define _ROCBLAS_HPP_

template <typename T>
struct rocblas_index_value_t;

#include "common_device.hpp"
#include "helpers.hpp"
#include "internal/rocblas-exported-proto.hpp"
#include "internal/rocblas_device_malloc.hpp"
#include <rocblas.h>

// iamax
template <bool ISBATCHED, typename T, typename S, typename U>
rocblas_status rocblasCall_iamax(rocblas_handle handle,
                                 rocblas_int n,
                                 U x,
                                 rocblas_int shiftx,
                                 rocblas_int incx,
                                 rocblas_stride stridex,
                                 rocblas_int batch_count,
                                 rocblas_int* result,
                                 rocblas_index_value_t<S>* workspace)
{
    return rocblas_iamax_template<ROCBLAS_IAMAX_NB, ISBATCHED>(
        handle, n, cast2constType<T>(x), shiftx, incx, stridex, batch_count, result, workspace);
}

// scal
template <typename T, typename U, typename V>
rocblas_status rocblasCall_scal(rocblas_handle handle,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stridea,
                                V x,
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                rocblas_int batch_count)
{
    return rocblas_scal_template<ROCBLAS_SCAL_NB, T>(handle, n, alpha, stridea, x, offsetx, incx,
                                                     stridex, batch_count);
}

// dot
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               U x,
                               rocblas_int offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               U y,
                               rocblas_int offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               T* workspace)
{
    return rocblas_dot_template<ROCBLAS_DOT_NB, CONJ, T>(
        handle, n, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(y), offsety,
        incy, stridey, batch_count, results, workspace);
}

// ger
template <bool CONJ, typename T, typename U, typename V>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               rocblas_int m,
                               rocblas_int n,
                               U alpha,
                               rocblas_stride stridea,
                               V x,
                               rocblas_int offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               V y,
                               rocblas_int offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               V A,
                               rocblas_int offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    return rocblas_ger_template<CONJ, T>(handle, m, n, alpha, stridea, cast2constType<T>(x),
                                         offsetx, incx, stridex, cast2constType<T>(y), offsety,
                                         incy, stridey, A, offsetA, lda, strideA, batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               rocblas_int m,
                               rocblas_int n,
                               U alpha,
                               rocblas_stride stridea,
                               T* const x[],
                               rocblas_int offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* y,
                               rocblas_int offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               T* const A[],
                               rocblas_int offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey, batch_count);

    return rocblas_ger_template<CONJ, T>(handle, m, n, alpha, stridea, cast2constType<T>(x),
                                         offsetx, incx, stridex, cast2constType<T>(work), offsety,
                                         incy, stridey, A, offsetA, lda, strideA, batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               rocblas_int m,
                               rocblas_int n,
                               U alpha,
                               rocblas_stride stridea,
                               T* x,
                               rocblas_int offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* const y[],
                               rocblas_int offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               T* const A[],
                               rocblas_int offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex, batch_count);

    return rocblas_ger_template<CONJ, T>(handle, m, n, alpha, stridea, cast2constType<T>(work),
                                         offsetx, incx, stridex, cast2constType<T>(y), offsety,
                                         incy, stridey, A, offsetA, lda, strideA, batch_count);
}

// gemv
template <typename T, typename U, typename V>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                V A,
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                V x,
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                V y,
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    return rocblas_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(A),
                                    offseta, lda, strideA, cast2constType<T>(x), offsetx, incx,
                                    stridex, beta, stride_beta, y, offsety, incy, stridey,
                                    batch_count);
}

// gemv overload
template <typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                T* A,
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* const y[],
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA, batch_count);

    return rocblas_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha,
                                    cast2constType<T>(work), offseta, lda, strideA,
                                    cast2constType<T>(x), offsetx, incx, stridex, beta, stride_beta,
                                    y, offsety, incy, stridey, batch_count);
}

// gemv overload
template <typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                T* const A[],
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* const y[],
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex, batch_count);

    return rocblas_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(A),
                                    offseta, lda, strideA, cast2constType<T>(work), offsetx, incx,
                                    stridex, beta, stride_beta, y, offsety, incy, stridey,
                                    batch_count);
}

// gemv overload
template <typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                T* const A[],
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey, batch_count);

    return rocblas_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(A),
                                    offseta, lda, strideA, cast2constType<T>(x), offsetx, incx,
                                    stridex, beta, stride_beta, cast2constPointer<T>(work), offsety,
                                    incy, stridey, batch_count);
}

// gemv overload
template <typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                T* const A[],
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex, batch_count);
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                       stridey, batch_count);

    return rocblas_gemv_template<T>(
        handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(A), offseta, lda, strideA,
        cast2constType<T>(work), offsetx, incx, stridex, beta, stride_beta,
        cast2constPointer<T>(work + batch_count), offsety, incy, stridey, batch_count);
}

// gemv overload
template <typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                T* A,
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_int offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA, batch_count);
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                       stridey, batch_count);

    return rocblas_gemv_template<T>(
        handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(work), offseta, lda, strideA,
        cast2constType<T>(x), offsetx, incx, stridex, beta, stride_beta,
        cast2constPointer<T>(work + batch_count), offsety, incy, stridey, batch_count);
}

// trmv
template <typename T, typename U>
rocblas_status rocblasCall_trmv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transa,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                U a,
                                rocblas_int offseta,
                                rocblas_int lda,
                                rocblas_stride stridea,
                                U x,
                                rocblas_int offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* w,
                                rocblas_stride stridew,
                                rocblas_int batch_count)
{
    return rocblas_trmv_template<ROCBLAS_TRMV_NB>(handle, uplo, transa, diag, m,
                                                  cast2constType<T>(a), offseta, lda, stridea, x,
                                                  offsetx, incx, stridex, w, stridew, batch_count);
}

// gemm
template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int m,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                V A,
                                rocblas_int offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                V B,
                                rocblas_int offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                V C,
                                rocblas_int offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    return rocblas_gemm_template<BATCHED, T>(handle, trans_a, trans_b, m, n, k, alpha,
                                             cast2constType<T>(A), offset_a, ld_a, stride_a,
                                             cast2constType<T>(B), offset_b, ld_b, stride_b, beta,
                                             C, offset_c, ld_c, stride_c, batch_count);
}

// gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int m,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                T* A,
                                rocblas_int offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* const B[],
                                rocblas_int offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* const C[],
                                rocblas_int offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a, batch_count);

    return rocblas_gemm_template<BATCHED, T>(handle, trans_a, trans_b, m, n, k, alpha,
                                             cast2constType<T>(work), offset_a, ld_a, stride_a,
                                             cast2constType<T>(B), offset_b, ld_b, stride_b, beta,
                                             C, offset_c, ld_c, stride_c, batch_count);
}

// gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int m,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                T* const A[],
                                rocblas_int offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_int offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* const C[],
                                rocblas_int offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, stride_b, batch_count);

    return rocblas_gemm_template<BATCHED, T>(handle, trans_a, trans_b, m, n, k, alpha,
                                             cast2constType<T>(A), offset_a, ld_a, stride_a,
                                             cast2constType<T>(work), offset_b, ld_b, stride_b,
                                             beta, C, offset_c, ld_c, stride_c, batch_count);
}

// gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int m,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                T* const A[],
                                rocblas_int offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* const B[],
                                rocblas_int offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* C,
                                rocblas_int offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, work, C, stride_c, batch_count);

    return rocblas_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(A), offset_a, ld_a, stride_a,
        cast2constType<T>(B), offset_b, ld_b, stride_b, beta, cast2constPointer(work), offset_c,
        ld_c, stride_c, batch_count);
}

// trmm
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                T* A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_int offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T* work,
                                T** workArr)
{
    constexpr rocblas_int nb = ROCBLAS_TRMM_NB;
    constexpr rocblas_stride strideW = 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB;

    // adding offsets directly to the arrays A and B until rocblas_trmm
    // supports offset arguments
    return rocblas_trmm_template<BATCHED, nb, nb, T>(
        handle, side, uplo, transA, diag, m, n, cast2constType<T>(alpha),
        cast2constType<T>(A + offsetA), lda, strideA, B + offsetB, ldb, strideB, batch_count, work,
        strideW);
}

// trmm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                T* const* A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const* B,
                                rocblas_int offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T* work,
                                T** workArr)
{
    constexpr rocblas_int nb = ROCBLAS_TRMM_NB;
    constexpr rocblas_stride strideW = 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, work, strideW,
                       batch_count);

    // until rocblas_trmm support offset arguments,
    // we need to manually offset A and B and store in temporary arrays AA and BB
    T **AA, **BB;
    hipMalloc(&AA, sizeof(T*) * batch_count);
    hipMalloc(&BB, sizeof(T*) * batch_count);
    hipLaunchKernelGGL(shift_array, dim3(blocks), dim3(256), 0, stream, AA, A, offsetA, batch_count);
    hipLaunchKernelGGL(shift_array, dim3(blocks), dim3(256), 0, stream, BB, B, offsetB, batch_count);

    rocblas_status status = rocblas_trmm_template<BATCHED, nb, nb, T>(
        handle, side, uplo, transA, diag, m, n, cast2constType<T>(alpha),
        cast2constType<T>(cast2constPointer<T>(AA)), lda, strideA, cast2constPointer<T>(BB), ldb,
        strideB, batch_count, cast2constPointer<T>(workArr), strideW);

    hipFree(AA);
    hipFree(BB);

    return status;
}

// trmm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                T* const* A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_int offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T* work,
                                T** workArr)
{
    constexpr rocblas_int nb = ROCBLAS_TRMM_NB;
    constexpr rocblas_stride strideW = 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (batch_count - 1) / 256 + 1;

    // adding offsets directly to the array B until rocblas_trmm
    // supports offset arguments
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, B + offsetB, strideB,
                       batch_count);
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr + batch_count, work,
                       strideW, batch_count);

    // until rocblas_trmm support offset arguments,
    // we need to manually offset A and store in temporary array AA
    T** AA;
    hipMalloc(&AA, sizeof(T*) * batch_count);
    hipLaunchKernelGGL(shift_array, dim3(blocks), dim3(256), 0, stream, AA, A, offsetA, batch_count);

    rocblas_status status = rocblas_trmm_template<BATCHED, nb, nb, T>(
        handle, side, uplo, transA, diag, m, n, cast2constType<T>(alpha),
        cast2constType<T>(cast2constPointer<T>(AA)), lda, strideA, cast2constPointer<T>(workArr),
        ldb, strideB, batch_count, cast2constPointer<T>(workArr + batch_count), strideW);

    hipFree(AA);

    return status;
}

// syrk
template <typename T, typename U, typename V>
rocblas_status rocblasCall_syrk(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                V A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                U beta,
                                V C,
                                rocblas_int offsetC,
                                rocblas_int ldc,
                                rocblas_stride strideC,
                                rocblas_int batch_count)
{
    return rocblas_syrk_template(handle, uplo, transA, n, k, cast2constType<T>(alpha),
                                 cast2constType<T>(A), offsetA, lda, strideA,
                                 cast2constType<T>(beta), C, offsetC, ldc, strideC, batch_count);
}

// herk
template <typename S, typename T, typename U, typename V, std::enable_if_t<!is_complex<T>, int> = 0>
rocblas_status rocblasCall_herk(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                V A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                U beta,
                                V C,
                                rocblas_int offsetC,
                                rocblas_int ldc,
                                rocblas_stride strideC,
                                rocblas_int batch_count)
{
    return rocblas_syrk_template(handle, uplo, transA, n, k, cast2constType<S>(alpha),
                                 cast2constType<T>(A), offsetA, lda, strideA,
                                 cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

template <typename S, typename T, typename U, typename V, std::enable_if_t<is_complex<T>, int> = 0>
rocblas_status rocblasCall_herk(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_int n,
                                rocblas_int k,
                                U alpha,
                                V A,
                                rocblas_int offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                U beta,
                                V C,
                                rocblas_int offsetC,
                                rocblas_int ldc,
                                rocblas_stride strideC,
                                rocblas_int batch_count)
{
    return rocblas_herk_template(handle, uplo, transA, n, k, cast2constType<S>(alpha),
                                 cast2constType<T>(A), offsetA, lda, strideA,
                                 cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

// trsm memory sizes
template <bool BATCHED, typename T>
void rocblasCall_trsm_mem(rocblas_side side,
                          rocblas_int m,
                          rocblas_int n,
                          rocblas_int batch_count,
                          size_t* x_temp,
                          size_t* x_temp_arr,
                          size_t* invA,
                          size_t* invA_arr)
{
    const rocblas_int BLOCK = ROCBLAS_TRSM_BLOCK;
    rocblas_int k = (side == rocblas_side_left) ? m : n;
    const bool exact_blocks = (k % BLOCK) == 0;
    size_t invA_els = k * BLOCK;
    size_t invA_bytes = invA_els * sizeof(T) * batch_count;
    size_t c_temp_els = (k / BLOCK) * ((BLOCK / 2) * (BLOCK / 2));
    size_t c_temp_bytes = c_temp_els * sizeof(T);

    size_t arrBytes = BATCHED ? sizeof(T*) * batch_count : 0;
    size_t xarrBytes = BATCHED ? sizeof(T*) * batch_count : 0;

    if(!exact_blocks)
    {
        // TODO: Make this more accurate -- right now it's much larger than
        // necessary
        size_t remainder_els = ROCBLAS_TRTRI_NB * BLOCK * 2;

        // C is the maximum of the temporary space needed for TRTRI
        c_temp_els = std::max(c_temp_els, remainder_els);
        c_temp_bytes = c_temp_els * sizeof(T);
    }

    // Chunk size for special algorithm
    size_t B_chunk_size = 0;

    // Temporary solution matrix
    size_t x_temp_els;
    size_t x_temp_bytes;

    if(exact_blocks)
    {
        // Optimal B_chunk_size is the orthogonal dimension to k
        B_chunk_size = size_t(m) + size_t(n) - size_t(k);

        // When k % BLOCK == 0, we only need BLOCK * B_chunk_size space
        x_temp_els = BLOCK * B_chunk_size;
        x_temp_bytes = x_temp_els * sizeof(T) * batch_count;
    }
    else
    {
        // When k % BLOCK != 0, we need m * n space
        x_temp_els = size_t(m) * n;
        x_temp_bytes = x_temp_els * sizeof(T) * batch_count;
    }

    // X and C temporaries can share space, so the maximum size is allocated
    size_t x_c_temp_bytes = std::max(x_temp_bytes, c_temp_bytes);

    // return required memory sizes
    *x_temp = x_c_temp_bytes;
    *x_temp_arr = xarrBytes;
    *invA = invA_bytes;
    *invA_arr = arrBytes;
}

// trsm
template <bool BATCHED, typename T, typename U>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                U A,
                                rocblas_int offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                U B,
                                rocblas_int offset_B,
                                rocblas_int ldb,
                                rocblas_stride stride_B,
                                rocblas_int batch_count,
                                bool optimal_mem,
                                void* x_temp,
                                void* x_temp_arr,
                                void* invA,
                                void* invA_arr,
                                T** workArr = nullptr)
{
    U supplied_invA = nullptr;
    return rocblas_trsm_template<ROCBLAS_TRSM_BLOCK, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType(A), offset_A, lda, stride_A,
        B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA, invA_arr,
        cast2constType(supplied_invA), 0);
}

// trsm overload
template <bool BATCHED, typename T>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                T* A,
                                rocblas_int offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                T* const B[],
                                rocblas_int offset_B,
                                rocblas_int ldb,
                                rocblas_stride stride_B,
                                rocblas_int batch_count,
                                bool optimal_mem,
                                void* x_temp,
                                void* x_temp_arr,
                                void* invA,
                                void* invA_arr,
                                T** workArr)
{
    using U = T* const*;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, A, stride_A,
                       batch_count);

    U supplied_invA = nullptr;
    return rocblas_trsm_template<ROCBLAS_TRSM_BLOCK, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType((U)workArr), offset_A, lda,
        stride_A, B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA,
        invA_arr, cast2constType(supplied_invA), 0);
}

// trtri memory sizes
template <bool BATCHED, typename T>
void rocblasCall_trtri_mem(rocblas_int n, rocblas_int batch_count, size_t* c_temp, size_t* c_temp_arr)
{
    size_t c_temp_els = rocblas_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, batch_count);
    *c_temp = c_temp_els * sizeof(T);

    *c_temp_arr = BATCHED ? sizeof(T*) * batch_count : 0;
}

// trtri
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 U A,
                                 rocblas_int offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 U invA,
                                 rocblas_int offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 U c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    return rocblas_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
        handle, uplo, diag, n, cast2constType(A), offset_A, lda, stride_A, 0, invA, offset_invA,
        ldinvA, stride_invA, 0, batch_count, 1, c_temp);
}

// trtri overload
template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 T* const A[],
                                 rocblas_int offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* const invA[],
                                 rocblas_int offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    size_t c_temp_els = rocblas_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                       c_temp_els, batch_count);

    return rocblas_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
        handle, uplo, diag, n, cast2constType(A), offset_A, lda, stride_A, 0, invA, offset_invA,
        ldinvA, stride_invA, 0, batch_count, 1, cast2constPointer(c_temp_arr));
}

// trtri overload
template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocblasCall_trtri(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int n,
                                 T* const A[],
                                 rocblas_int offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* invA,
                                 rocblas_int offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    size_t c_temp_els = rocblas_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, invA, stride_invA,
                       batch_count);
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                       c_temp_els, batch_count);

    return rocblas_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
        handle, uplo, diag, n, cast2constType(A), offset_A, lda, stride_A, 0,
        cast2constPointer(workArr), offset_invA, ldinvA, stride_invA, 0, batch_count, 1,
        cast2constPointer(c_temp_arr));
}

#endif // _ROCBLAS_HPP_
