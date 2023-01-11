/************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <typename T>
rocblas_status rocsolver_lauum_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        T A,
                                        const rocblas_int lda)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(n && !A)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T>
void rocsolver_lauum_getMemorySize(const rocblas_int n, const rocblas_int batch_count, size_t* size_work)
{
    *size_work = 0;

    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
        return;

    // size of workspace
    *size_work = sizeof(T) * n * n * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lauum_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const rocblas_int batch_count,
                                        U work)
{
    ROCSOLVER_ENTER("lauum", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "strideA:", strideA, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int strideW = n * n;
    rocblas_int blocks = (n - 1) / BS2 + 1;
    dim3 grid(blocks, blocks, batch_count);
    dim3 threads(BS2, BS2);
    S one = 1;
    S zero = 0;

    rocblas_operation transW
        = (uplo == rocblas_fill_upper) ? rocblas_operation_transpose : rocblas_operation_none;

    // put the triangular factor of interest in work
    ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, grid, threads, 0, stream, n, n, work, 0, n, strideW, uplo);
    ROCSOLVER_LAUNCH_KERNEL(copy_trans_mat<T>, grid, threads, 0, stream, transW, n, n, A, shiftA,
                            lda, strideA, work, 0, n, strideW, no_mask{}, uplo);

    // work = work * work' or work = work' * work
    rocblasCall_syrk_herk<false, T>(handle, uplo, rocblas_operation_conjugate_transpose, n, n, &one,
                                    work, 0, n, strideW, &zero, A, shiftA, lda, strideA, batch_count);

    return rocblas_status_success;
}
