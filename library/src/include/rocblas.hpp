/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <rocblas/rocblas.h>

#include "common_host_helpers.hpp"
#include "init_scalars.hpp"
#include "lib_device_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocblas/internal/rocblas-exported-proto.hpp"
#include "rocblas/internal/rocblas_block_sizes.h"
#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocsolver_logger.hpp"

template <typename T>
struct rocblas_index_value_t;

// axpy
template <typename T, typename U>
rocblas_status rocblasCall_axpy(rocblas_handle handle,
                                rocblas_int n,
                                T* alpha,
                                rocblas_stride stride_alpha,
                                U x,
                                rocblas_stride shiftx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U y,
                                rocblas_stride shifty,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count)
{
    // TODO: How to get alpha for trace logging
    //ROCBLAS_ENTER("axpy", "n:", n, "shiftX:", shiftx, "incx:", incx, "shiftY:", shifty, "incy:", incy, "bc:", batch_count);

    return rocblas_internal_axpy_template<ROCBLAS_AXPY_NB, T>(
        handle, n, cast2constType<T>(alpha), stride_alpha, cast2constType<T>(x), shiftx, incx,
        stridex, y, shifty, incy, stridey, batch_count);
}

// iamax
template <typename T, typename S, typename U>
rocblas_status rocblasCall_iamax(rocblas_handle handle,
                                 rocblas_int n,
                                 U x,
                                 rocblas_stride shiftx,
                                 rocblas_int incx,
                                 rocblas_stride stridex,
                                 rocblas_int batch_count,
                                 rocblas_int* result,
                                 rocblas_index_value_t<S>* workspace)
{
    ROCBLAS_ENTER("iamax", "n:", n, "shiftX:", shiftx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_iamax_template<ROCBLAS_IAMAX_NB>(
        handle, n, cast2constType<T>(x), shiftx, incx, stridex, batch_count, result, workspace);
}

// scal
template <typename T, typename U, typename V>
rocblas_status rocblasCall_scal(rocblas_handle handle,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stridea,
                                V x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                rocblas_int batch_count)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("scal", "n:", n, "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_scal_template<ROCBLAS_SCAL_NB, T>(handle, n, alpha, stridea, x, offsetx,
                                                              incx, stridex, batch_count);
}

// dot
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               U x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               U y,
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               T* workspace,
                               T** work = nullptr)
{
    ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "bc:", batch_count);

    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, CONJ, T>(
        handle, n, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(y), offsety,
        incy, stridey, batch_count, results, workspace);
}

// dot overload
template <bool CONJ, typename T>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                               rocblas_int n,
                               T* x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* const y[],
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               rocblas_int batch_count,
                               T* results,
                               T* workspace,
                               T** work)
{
    ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, CONJ, T>(
        handle, n, cast2constType<T>(work), offsetx, incx, stridex, cast2constType<T>(y), offsety,
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
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               V y,
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               V A,
                               rocblas_stride offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    return rocblas_internal_ger_template<CONJ, T>(
        handle, m, n, alpha, stridea, cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(y), offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               rocblas_int m,
                               rocblas_int n,
                               U alpha,
                               rocblas_stride stridea,
                               T* const x[],
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* y,
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               T* const A[],
                               rocblas_stride offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    return rocblas_internal_ger_template<CONJ, T>(
        handle, m, n, alpha, stridea, cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(work), offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle,
                               rocblas_int m,
                               rocblas_int n,
                               U alpha,
                               rocblas_stride stridea,
                               T* x,
                               rocblas_stride offsetx,
                               rocblas_int incx,
                               rocblas_stride stridex,
                               T* const y[],
                               rocblas_stride offsety,
                               rocblas_int incy,
                               rocblas_stride stridey,
                               T* const A[],
                               rocblas_stride offsetA,
                               rocblas_int lda,
                               rocblas_stride strideA,
                               rocblas_int batch_count,
                               T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("ger", "m:", m, "n:", n, "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety,
                  "incy:", incy, "shiftA:", offsetA, "lda:", lda, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    return rocblas_internal_ger_template<CONJ, T>(
        handle, m, n, alpha, stridea, cast2constType<T>(work), offsetx, incx, stridex,
        cast2constType<T>(y), offsety, incy, stridey, A, offsetA, lda, strideA, batch_count);
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                V x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                V y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    return rocblas_internal_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha,
                                             cast2constType<T>(A), offseta, lda, strideA,
                                             cast2constType<T>(x), offsetx, incx, stridex, beta,
                                             stride_beta, y, offsety, incy, stridey, batch_count);
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* const y[],
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA,
                            batch_count);

    return rocblas_internal_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha,
                                             cast2constType<T>(work), offseta, lda, strideA,
                                             cast2constType<T>(x), offsetx, incx, stridex, beta,
                                             stride_beta, y, offsety, incy, stridey, batch_count);
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* const y[],
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);

    return rocblas_internal_gemv_template<T>(handle, transA, m, n, alpha, stride_alpha,
                                             cast2constType<T>(A), offseta, lda, strideA,
                                             cast2constType<T>(work), offsetx, incx, stridex, beta,
                                             stride_beta, y, offsety, incy, stridey, batch_count);
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    return rocblas_internal_gemv_template<T>(
        handle, transA, m, n, alpha, stride_alpha, cast2constType<T>(A), offseta, lda, strideA,
        cast2constType<T>(x), offsetx, incx, stridex, beta, stride_beta, cast2constPointer<T>(work),
        offsety, incy, stridey, batch_count);
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, x, stridex,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                            stridey, batch_count);

    return rocblas_internal_gemv_template<T>(
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* const x[],
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                U beta,
                                rocblas_stride stride_beta,
                                T* y,
                                rocblas_stride offsety,
                                rocblas_int incy,
                                rocblas_stride stridey,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemv", "trans:", transA, "m:", m, "n:", n, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "shiftY:", offsety, "incy:", incy,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, strideA,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, (work + batch_count), y,
                            stridey, batch_count);

    return rocblas_internal_gemv_template<T>(
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
                                rocblas_stride offseta,
                                rocblas_int lda,
                                rocblas_stride stridea,
                                U x,
                                rocblas_stride offsetx,
                                rocblas_int incx,
                                rocblas_stride stridex,
                                T* w,
                                rocblas_stride stridew,
                                rocblas_int batch_count)
{
    ROCBLAS_ENTER("trmv", "trans:", transa, "diag:", diag, "m:", m, "shiftA:", offseta, "lda:", lda,
                  "shiftX:", offsetx, "incx:", incx, "bc:", batch_count);

    return rocblas_internal_trmv_template(handle, uplo, transa, diag, m, cast2constType<T>(a),
                                          offseta, lda, stridea, x, offsetx, incx, stridex, w,
                                          stridew, batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                V B,
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                V C,
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(A), offset_a, ld_a, stride_a,
        cast2constType<T>(B), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c,
        batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* const B[],
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* const C[],
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a, stride_a,
        cast2constType<T>(B), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c,
        batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* const C[],
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, stride_b,
                            batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(A), offset_a, ld_a, stride_a,
        cast2constType<T>(work), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c, stride_c,
        batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* const B[],
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* C,
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, C, stride_c,
                            batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(A), offset_a, ld_a, stride_a,
        cast2constType<T>(B), offset_b, ld_b, stride_b, beta, cast2constPointer(work), offset_c,
        ld_c, stride_c, batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* C,
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, stride_b,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, C,
                            stride_c, batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(A), offset_a, ld_a, stride_a,
        cast2constType<T>(work), offset_b, ld_b, stride_b, beta,
        cast2constPointer(work + batch_count), offset_c, ld_c, stride_c, batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* const B[],
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* C,
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, C,
                            stride_c, batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a, stride_a,
        cast2constType<T>(B), offset_b, ld_b, stride_b, beta, cast2constPointer(work + batch_count),
        offset_c, ld_c, stride_c, batch_count);
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
                                rocblas_stride offset_a,
                                rocblas_int ld_a,
                                rocblas_stride stride_a,
                                T* B,
                                rocblas_stride offset_b,
                                rocblas_int ld_b,
                                rocblas_stride stride_b,
                                U beta,
                                T* const C[],
                                rocblas_stride offset_c,
                                rocblas_int ld_c,
                                rocblas_stride stride_c,
                                rocblas_int batch_count,
                                T** work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("gemm", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n, "k:", k,
                  "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, A, stride_a,
                            batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work + batch_count, B,
                            stride_b, batch_count);

    return rocblas_internal_gemm_template<BATCHED, T>(
        handle, trans_a, trans_b, m, n, k, alpha, cast2constType<T>(work), offset_a, ld_a, stride_a,
        cast2constType<T>(work + batch_count), offset_b, ld_b, stride_b, beta, C, offset_c, ld_c,
        stride_c, batch_count);
}

// trmm
template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblasCall_trmm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                U alpha,
                                rocblas_stride stride_alpha,
                                V A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                V B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T** workArr = nullptr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trmm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftB:", offsetB, "ldb:", ldb,
                  "bc:", batch_count);

    constexpr rocblas_int nb = (!rocblas_is_complex<T> ? ROCBLAS_SDTRMM_NB : ROCBLAS_CZTRMM_NB);

    return rocblas_internal_trmm_template<nb, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, cast2constType<T>(alpha), stride_alpha,
        cast2constType<T>(A), offsetA, lda, strideA, cast2constType<T>(B), offsetB, ldb, strideB, B,
        offsetB, ldb, strideB, batch_count);
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
                                rocblas_stride stride_alpha,
                                T* const* A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count,
                                T** workArr)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trmm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftB:", offsetB, "ldb:", ldb,
                  "bc:", batch_count);

    constexpr rocblas_int nb = (!rocblas_is_complex<T> ? ROCBLAS_SDTRMM_NB : ROCBLAS_CZTRMM_NB);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (batch_count - 1) / 256 + 1;

    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, B, strideB,
                            batch_count);

    return rocblas_internal_trmm_template<nb, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, cast2constType<T>(alpha), stride_alpha,
        cast2constType<T>(A), offsetA, lda, strideA, cast2constType<T>(workArr), offsetB, ldb,
        strideB, cast2constPointer<T>(workArr), offsetB, ldb, strideB, batch_count);
}

// syr2
template <typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     V x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     V y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("syr2", "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx,
                  "shiftY:", offsety, "incy:", incy, "shiftA:", offsetA, "lda:", lda,
                  "bc:", batch_count);

    return rocblas_internal_syr2_template(
        handle, uplo, n, cast2constType<T>(alpha), cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(y), offsety, incy, stridey, A, lda, offsetA, strideA, batch_count);
}

// syr2 overload
template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     T* const x[],
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     T* const A[],
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("syr2", "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx,
                  "shiftY:", offsety, "incy:", incy, "shiftA:", offsetA, "lda:", lda,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    return rocblas_internal_syr2_template(
        handle, uplo, n, cast2constType<T>(alpha), cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(work), offsety, incy, stridey, A, lda, offsetA, strideA, batch_count);
}

// her2
template <typename T, typename U, typename V, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     V x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     V y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("her2", "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx,
                  "shiftY:", offsety, "incy:", incy, "shiftA:", offsetA, "lda:", lda,
                  "bc:", batch_count);

    return rocblas_internal_her2_template(
        handle, uplo, n, cast2constType<T>(alpha), cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(y), offsety, incy, stridey, A, lda, offsetA, strideA, batch_count);
}

// her2 overload
template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2_her2(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     T* const x[],
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     T* const A[],
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     rocblas_int batch_count,
                                     T** work)
{
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("her2", "uplo:", uplo, "n:", n, "shiftX:", offsetx, "incx:", incx,
                  "shiftY:", offsety, "incy:", incy, "shiftA:", offsetA, "lda:", lda,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, y, stridey,
                            batch_count);

    return rocblas_internal_her2_template(
        handle, uplo, n, cast2constType<T>(alpha), cast2constType<T>(x), offsetx, incx, stridex,
        cast2constType<T>(work), offsety, incy, stridey, A, lda, offsetA, strideA, batch_count);
}

// syrk
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syrk_herk(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_int n,
                                     rocblas_int k,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syrk", "uplo:", uplo, "trans:", transA, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftC:", offsetC, "ldc:", ldc, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    constexpr rocblas_int NB = BATCHED ? ROCBLAS_SDSYRK_BATCHED_NB : ROCBLAS_SDZSYRK_NB;

    return rocblas_internal_syrk_template<NB, BATCHED, T>(
        handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

// herk
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syrk_herk(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_int n,
                                     rocblas_int k,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("herk", "uplo:", uplo, "trans:", transA, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftC:", offsetC, "ldc:", ldc, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    constexpr rocblas_int NB = BATCHED                  ? ROCBLAS_HERK_BATCHED_NB
        : std::is_same<T, rocblas_float_complex>::value ? ROCBLAS_CHERK_NB
                                                        : ROCBLAS_ZHERK_NB;

    return rocblas_internal_herk_template<NB, BATCHED, T>(
        handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}

// syr2k
template <bool BATCHED,
          typename T,
          typename Ua,
          typename Ub,
          typename V,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       V A,
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       V B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       V C,
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syr2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    constexpr bool TWOK = true;
    constexpr bool HERK = false;
    constexpr rocblas_int NB = BATCHED                  ? ROCBLAS_SDSYR2K_BATCHED_NB
        : std::is_same<T, float>::value                 ? ROCBLAS_SSYR2K_NB
        : std::is_same<T, double>::value                ? ROCBLAS_DSYR2K_NB
        : std::is_same<T, rocblas_float_complex>::value ? ROCBLAS_CSYR2K_NB
                                                        : ROCBLAS_ZSYR2K_NB;

    return rocblas_internal_syr2k_her2k_template<NB, BATCHED, TWOK, HERK>(
        handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<T>(beta), C, offsetC,
        ldc, strideC, batch_count);
}

// syr2k overload
template <bool BATCHED, typename T, typename Ua, typename Ub, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       T* const A[],
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       T* B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       T* const C[],
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syr2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, strideB,
                            batch_count);

    constexpr bool TWOK = true;
    constexpr bool HERK = false;
    constexpr rocblas_int NB = BATCHED                  ? ROCBLAS_SDSYR2K_BATCHED_NB
        : std::is_same<T, float>::value                 ? ROCBLAS_SSYR2K_NB
        : std::is_same<T, double>::value                ? ROCBLAS_DSYR2K_NB
        : std::is_same<T, rocblas_float_complex>::value ? ROCBLAS_CSYR2K_NB
                                                        : ROCBLAS_ZSYR2K_NB;

    return rocblas_internal_syr2k_her2k_template<NB, BATCHED, TWOK, HERK>(
        handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<T>(beta), C,
        offsetC, ldc, strideC, batch_count);
}

// her2k
template <bool BATCHED,
          typename T,
          typename Ua,
          typename Ub,
          typename V,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       V A,
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       V B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       V C,
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("her2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    using S = decltype(std::real(T{}));

    constexpr bool TWOK = true;
    constexpr bool HERK = true;
    constexpr rocblas_int NB = BATCHED                  ? ROCBLAS_HER2K_BATCHED_NB
        : std::is_same<T, rocblas_float_complex>::value ? ROCBLAS_CHER2K_NB
                                                        : ROCBLAS_ZHER2K_NB;

    return rocblas_internal_syr2k_her2k_template<NB, BATCHED, TWOK, HERK>(
        handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<S>(beta), C, offsetC,
        ldc, strideC, batch_count);
}

// her2k overload
template <bool BATCHED, typename T, typename Ua, typename Ub, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syr2k_her2k(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int k,
                                       Ua alpha,
                                       T* const A[],
                                       rocblas_stride offsetA,
                                       rocblas_int lda,
                                       rocblas_stride strideA,
                                       T* B,
                                       rocblas_stride offsetB,
                                       rocblas_int ldb,
                                       rocblas_stride strideB,
                                       Ub beta,
                                       T* const C[],
                                       rocblas_stride offsetC,
                                       rocblas_int ldc,
                                       rocblas_stride strideC,
                                       rocblas_int batch_count,
                                       T** work = nullptr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("her2k", "uplo:", uplo, "trans:", trans, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    using S = decltype(std::real(T{}));

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, work, B, strideB,
                            batch_count);

    constexpr bool TWOK = true;
    constexpr bool HERK = true;
    constexpr rocblas_int NB = BATCHED                  ? ROCBLAS_HER2K_BATCHED_NB
        : std::is_same<T, rocblas_float_complex>::value ? ROCBLAS_CHER2K_NB
                                                        : ROCBLAS_ZHER2K_NB;

    return rocblas_internal_syr2k_her2k_template<NB, BATCHED, TWOK, HERK>(
        handle, uplo, trans, n, k, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(work), offsetB, ldb, strideB, cast2constType<S>(beta), C,
        offsetC, ldc, strideC, batch_count);
}

// symv/hemv memory sizes
template <bool BATCHED, typename T>
void rocblasCall_symv_hemv_mem(rocblas_int n, rocblas_int batch_count, size_t* w_temp)
{
    *w_temp = rocblas_internal_hemv_symv_kernel_workspace_size<T>(n, batch_count);
}

// symv
template <typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     rocblas_stride stridea,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     V x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     U beta,
                                     rocblas_stride strideb,
                                     V y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("symv", "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    return rocblas_internal_hemv_symv_template<false, T>(
        handle, uplo, n, cast2constType<T>(alpha), stridea, cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(beta), strideb, y,
        offsety, incy, stridey, batch_count, work);
}

// symv overload
template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     rocblas_stride stridea,
                                     T* const A[],
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     T* const x[],
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     U beta,
                                     rocblas_stride strideb,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("symv", "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, y, stridey,
                            batch_count);

    return rocblas_internal_hemv_symv_template<false, T>(
        handle, uplo, n, cast2constType<T>(alpha), stridea, cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(beta), strideb,
        cast2constPointer<T>(workArr), offsety, incy, stridey, batch_count, work);
}

// hemv
template <typename T, typename U, typename V, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     rocblas_stride stridea,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     V x,
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     U beta,
                                     rocblas_stride strideb,
                                     V y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("hemv", "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    return rocblas_internal_hemv_symv_template<true, T>(
        handle, uplo, n, cast2constType<T>(alpha), stridea, cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(beta), strideb, y,
        offsety, incy, stridey, batch_count, work);
}

// hemv overload
template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symv_hemv(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     U alpha,
                                     rocblas_stride stridea,
                                     T* const A[],
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     T* const x[],
                                     rocblas_stride offsetx,
                                     rocblas_int incx,
                                     rocblas_stride stridex,
                                     U beta,
                                     rocblas_stride strideb,
                                     T* y,
                                     rocblas_stride offsety,
                                     rocblas_int incy,
                                     rocblas_stride stridey,
                                     rocblas_int batch_count,
                                     T* work,
                                     T** workArr)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("hemv", "uplo:", uplo, "n:", n, "shiftA:", offsetA, "lda:", lda, "shiftX:", offsetx,
                  "incx:", incx, "shiftY:", offsety, "incy:", incy, "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, y, stridey,
                            batch_count);

    return rocblas_internal_hemv_symv_template<true, T>(
        handle, uplo, n, cast2constType<T>(alpha), stridea, cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(x), offsetx, incx, stridex, cast2constType<T>(beta), strideb,
        cast2constPointer<T>(workArr), offsety, incy, stridey, batch_count, work);
}

// symm
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symm_hemm(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_int m,
                                     rocblas_int n,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     V B,
                                     rocblas_stride offsetB,
                                     rocblas_int ldb,
                                     rocblas_stride strideB,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("symm", "side:", side, "uplo:", uplo, "m:", m, "n:", n, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    return rocblas_internal_symm_template<BATCHED, false, T>(
        handle, side, uplo, m, n, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<T>(beta), C, offsetC,
        ldc, strideC, batch_count);
}

// hemm
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_symm_hemm(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_int m,
                                     rocblas_int n,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     V B,
                                     rocblas_stride offsetB,
                                     rocblas_int ldb,
                                     rocblas_stride strideB,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("hemm", "side:", side, "uplo:", uplo, "m:", m, "n:", n, "shiftA:", offsetA,
                  "lda:", lda, "shiftB:", offsetB, "ldb:", ldb, "shiftC:", offsetC, "ldc:", ldc,
                  "bc:", batch_count);

    return rocblas_internal_symm_template<BATCHED, true, T>(
        handle, side, uplo, m, n, cast2constType<T>(alpha), cast2constType<T>(A), offsetA, lda,
        strideA, cast2constType<T>(B), offsetB, ldb, strideB, cast2constType<T>(beta), C, offsetC,
        ldc, strideC, batch_count);
}

// trsv
template <bool BATCHED,
          typename T,
          typename U,
          std::enable_if_t<!std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                U A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                U x,
                                rocblas_stride offset_x,
                                rocblas_int incx,
                                rocblas_stride stride_x,
                                rocblas_int batch_count,
                                rocblas_int* w_completed_sec,
                                T** workArr = nullptr)
{
    ROCBLAS_ENTER("trsv", "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "shiftA:", offset_A, "lda:", lda, "shiftx:", offset_x, "incx:", incx,
                  "bc:", batch_count);

    // nullptr for optional alpha
    return rocblas_internal_trsv_substitution_template<ROCBLAS_SDCTRSV_NB, T>(
        handle, uplo, transA, diag, m, cast2constType(A), offset_A, lda, stride_A, nullptr, x,
        offset_x, incx, stride_x, batch_count, w_completed_sec);
}

template <bool BATCHED,
          typename T,
          typename U,
          std::enable_if_t<std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsv(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                U A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                U x,
                                rocblas_stride offset_x,
                                rocblas_int incx,
                                rocblas_stride stride_x,
                                rocblas_int batch_count,
                                rocblas_int* w_completed_sec,
                                T** workArr = nullptr)
{
    ROCBLAS_ENTER("trsv", "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "shiftA:", offset_A, "lda:", lda, "shiftx:", offset_x, "incx:", incx,
                  "bc:", batch_count);

    // nullptr for optional alpha
    return rocblas_internal_trsv_substitution_template<ROCBLAS_ZTRSV_NB, T>(
        handle, uplo, transA, diag, m, cast2constType(A), offset_A, lda, stride_A, nullptr, x,
        offset_x, incx, stride_x, batch_count, w_completed_sec);
}

// trsm memory sizes
template <bool BATCHED, typename T>
void rocblasCall_trsm_mem(rocblas_side side,
                          rocblas_operation transA,
                          rocblas_int m,
                          rocblas_int n,
                          rocblas_int batch_count,
                          size_t* x_temp,
                          size_t* x_temp_arr,
                          size_t* invA,
                          size_t* invA_arr)
{
    size_t no_opt_size;
    /** TODO: For now, we always request the size for optimal performance.
        no_opt_size could be used in the future if we generalize the use of
        rocblas_workmode parameter **/

    rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, BATCHED, T>(
        side, transA, m, n, batch_count, 0, x_temp, x_temp_arr, invA, invA_arr, &no_opt_size);
}

// trsm
template <bool BATCHED,
          typename T,
          typename U,
          std::enable_if_t<!std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                U A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                U B,
                                rocblas_stride offset_B,
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
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    U supplied_invA = nullptr;
    return rocblas_internal_trsm_template<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType(A), offset_A, lda, stride_A,
        B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA, invA_arr,
        cast2constType(supplied_invA), 0);
}

template <bool BATCHED,
          typename T,
          typename U,
          std::enable_if_t<std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                U A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                U B,
                                rocblas_stride offset_B,
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
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    U supplied_invA = nullptr;
    return rocblas_internal_trsm_template<ROCBLAS_TRSM_NB, ROCBLAS_ZTRSV_NB, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType(A), offset_A, lda, stride_A,
        B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA, invA_arr,
        cast2constType(supplied_invA), 0);
}

// trsm overload
template <bool BATCHED, typename T, std::enable_if_t<!std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                T* const B[],
                                rocblas_stride offset_B,
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
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    using U = T* const*;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, A, stride_A,
                            batch_count);

    U supplied_invA = nullptr;
    return rocblas_internal_trsm_template<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType((U)workArr), offset_A, lda,
        stride_A, B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA,
        invA_arr, cast2constType(supplied_invA), 0);
}

template <bool BATCHED, typename T, std::enable_if_t<std::is_same<T, rocblas_double_complex>::value, int> = 0>
rocblas_status rocblasCall_trsm(rocblas_handle handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                T* A,
                                rocblas_stride offset_A,
                                rocblas_int lda,
                                rocblas_stride stride_A,
                                T* const B[],
                                rocblas_stride offset_B,
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
    // TODO: How to get alpha for trace logging
    ROCBLAS_ENTER("trsm", "side:", side, "uplo:", uplo, "trans:", transA, "diag:", diag, "m:", m,
                  "n:", n, "shiftA:", offset_A, "lda:", lda, "shiftB:", offset_B, "ldb:", ldb,
                  "bc:", batch_count);

    using U = T* const*;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, A, stride_A,
                            batch_count);

    U supplied_invA = nullptr;
    return rocblas_internal_trsm_template<ROCBLAS_TRSM_NB, ROCBLAS_ZTRSV_NB, BATCHED, T>(
        handle, side, uplo, transA, diag, m, n, alpha, cast2constType((U)workArr), offset_A, lda,
        stride_A, B, offset_B, ldb, stride_B, batch_count, optimal_mem, x_temp, x_temp_arr, invA,
        invA_arr, cast2constType(supplied_invA), 0);
}

// trtri memory sizes
template <bool BATCHED, typename T>
void rocblasCall_trtri_mem(rocblas_int n, rocblas_int batch_count, size_t* c_temp, size_t* c_temp_arr)
{
    size_t c_temp_els = rocblas_internal_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, batch_count);
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
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 U invA,
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 U c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    return rocblas_internal_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
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
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* const invA[],
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    size_t c_temp_els = rocblas_internal_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                            c_temp_els, batch_count);

    return rocblas_internal_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
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
                                 rocblas_stride offset_A,
                                 rocblas_int lda,
                                 rocblas_stride stride_A,
                                 T* invA,
                                 rocblas_stride offset_invA,
                                 rocblas_int ldinvA,
                                 rocblas_stride stride_invA,
                                 rocblas_int batch_count,
                                 T* c_temp,
                                 T** c_temp_arr,
                                 T** workArr)
{
    ROCBLAS_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", offset_A, "lda:", lda,
                  "shiftC:", offset_invA, "ldc:", ldinvA, "bc:", batch_count);

    size_t c_temp_els = rocblas_internal_trtri_temp_size<ROCBLAS_TRTRI_NB>(n, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, invA,
                            stride_invA, batch_count);
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, c_temp_arr, c_temp,
                            c_temp_els, batch_count);

    return rocblas_internal_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, STRIDED, T>(
        handle, uplo, diag, n, cast2constType(A), offset_A, lda, stride_A, 0,
        cast2constPointer(workArr), offset_invA, ldinvA, stride_invA, 0, batch_count, 1,
        cast2constPointer(c_temp_arr));
}
