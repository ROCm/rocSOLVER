/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_potrf.hpp"
#include "roclapack_potrs.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_posv_argCheck(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       T B,
                                       rocblas_int* info,
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
    if((n && !A) || (nrhs * n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_posv_getMemorySize(const rocblas_int n,
                                  const rocblas_int nrhs,
                                  const rocblas_fill uplo,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work1,
                                  size_t* size_work2,
                                  size_t* size_work3,
                                  size_t* size_work4,
                                  size_t* size_pivots_savedB,
                                  size_t* size_iinfo)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots_savedB = 0;
        *size_iinfo = 0;
        return;
    }

    size_t w1, w2, w3, w4;

    // workspace required for potrf
    rocsolver_potrf_getMemorySize<BATCHED, T>(n, uplo, batch_count, size_scalars, size_work1,
                                              size_work2, size_work3, size_work4,
                                              size_pivots_savedB, size_iinfo);

    // workspace required for potrs
    rocsolver_potrs_getMemorySize<BATCHED, T>(n, nrhs, batch_count, &w1, &w2, &w3, &w4);

    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);

    // extra space to copy B
    *size_pivots_savedB = std::max(*size_pivots_savedB, sizeof(T) * n * nrhs * batch_count);
}

template <bool BATCHED, typename T, typename S, typename U>
rocblas_status rocsolver_posv_template(rocblas_handle handle,
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
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       void* work1,
                                       void* work2,
                                       void* work3,
                                       void* work4,
                                       T* pivots_savedB,
                                       rocblas_int* iinfo,
                                       bool optim_mem)
{
    ROCSOLVER_ENTER("posv", "uplo:", uplo, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if A or B are empty
    if(n == 0 || nrhs == 0)
        return rocblas_status_success;

    // constants in host memory
    const rocblas_int copyblocksx = (n - 1) / 32 + 1;
    const rocblas_int copyblocksy = (nrhs - 1) / 32 + 1;

    // compute Cholesky factorization of A
    rocsolver_potrf_template<BATCHED, T, S>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                            batch_count, scalars, work1, work2, work3, work4,
                                            pivots_savedB, iinfo, optim_mem);

    // save elements of B that will be overwritten by POTRS for cases where info is nonzero
    hipLaunchKernelGGL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                       dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb, strideB,
                       pivots_savedB, info_mask(info));

    // solve AX = B, overwriting B with X
    rocsolver_potrs_template<BATCHED, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, B, shiftB,
                                         ldb, strideB, batch_count, work1, work2, work3, work4,
                                         optim_mem);

    // restore elements of B that were overwritten by POTRS in cases where info is nonzero
    hipLaunchKernelGGL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                       dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB, ldb,
                       strideB, pivots_savedB, info_mask(info));

    return rocblas_status_success;
}
