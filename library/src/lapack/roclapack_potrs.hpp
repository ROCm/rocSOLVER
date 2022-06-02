/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

template <typename T>
rocblas_status rocsolver_potrs_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        const rocblas_int lda,
                                        const rocblas_int ldb,
                                        T A,
                                        T B,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (nrhs * n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_potrs_getMemorySize(const rocblas_int n,
                                   const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   bool* optim_mem)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *optim_mem = true;
        return;
    }

    // workspace required for calling TRSM
    // call with both rocblas_operation_none and rocblas_operation_conjugate_transpose and take maximum memory
    size_t size_work1_temp1, size_work1_temp2, size_work2_temp1, size_work2_temp2, size_work3_temp1,
        size_work3_temp2, size_work4_temp1, size_work4_temp2;
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none, n, nrhs,
                                            batch_count, &size_work1_temp1, &size_work2_temp1,
                                            &size_work3_temp1, &size_work4_temp1, optim_mem);
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(
        rocblas_side_left, rocblas_operation_conjugate_transpose, n, nrhs, batch_count,
        &size_work1_temp2, &size_work2_temp2, &size_work3_temp2, &size_work4_temp2, optim_mem);

    *size_work1 = std::max(size_work1_temp1, size_work1_temp2);
    *size_work2 = std::max(size_work2_temp1, size_work2_temp2);
    *size_work3 = std::max(size_work3_temp1, size_work3_temp2);
    *size_work4 = std::max(size_work4_temp1, size_work4_temp2);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_potrs_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        U B,
                                        const rocblas_int shiftB,
                                        const rocblas_int ldb,
                                        const rocblas_stride strideB,
                                        const rocblas_int batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("potrs", "uplo:", uplo, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    if(uplo == rocblas_fill_upper)
    {
        // solve U'*X = B, overwriting B with X
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
            rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
            batch_count, optim_mem, work1, work2, work3, work4);

        // solve U*X = B, overwriting B with X
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, rocblas_operation_none,
                                                  rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                                  lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                                  optim_mem, work1, work2, work3, work4);
    }
    else
    {
        // solve L*X = B, overwriting B with X
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, rocblas_operation_none,
                                                  rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                                  lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                                  optim_mem, work1, work2, work3, work4);

        // solve L'*X = B, overwriting B with X
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
            rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
            batch_count, optim_mem, work1, work2, work3, work4);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
