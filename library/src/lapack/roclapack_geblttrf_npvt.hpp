/************************************************************************
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

template <typename T>
ROCSOLVER_KERNEL void
    geblttrf_update_info(T* info, T* iinfo, const rocblas_int k_shift, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] == 0 && iinfo[b] != 0)
            info[b] = iinfo[b] + k_shift;
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrf_npvt_getMemorySize(const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_pivotval,
                                           size_t* size_pivotidx,
                                           size_t* size_iipiv,
                                           size_t* size_iinfo1,
                                           size_t* size_iinfo2,
                                           bool* optim_mem)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iipiv = 0;
        *size_iinfo1 = 0;
        *size_iinfo2 = 0;
        return;
    }

    bool unused;
    size_t a1 = 0, a2 = 0;
    size_t b1 = 0, b2 = 0;
    size_t c1 = 0, c2 = 0;
    size_t d1 = 0, d2 = 0;

    // size requirements for getrf
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T>(nb, nb, false, batch_count, size_scalars, &a1,
                                                       &b1, &c1, &d1, size_pivotval, size_pivotidx,
                                                       size_iipiv, size_iinfo1, optim_mem);

    // size requirements for getrs
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(rocblas_operation_none, nb, nb, batch_count,
                                                       &a2, &b2, &c2, &d2, &unused);

    *size_work1 = max(a1, a2);
    *size_work2 = max(b1, b2);
    *size_work3 = max(c1, c2);
    *size_work4 = max(d1, d2);

    // size for temporary info storage
    *size_iinfo2 = sizeof(rocblas_int) * batch_count;
}

template <typename T>
rocblas_status rocsolver_geblttrf_npvt_argCheck(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int lda,
                                                const rocblas_int ldb,
                                                const rocblas_int ldc,
                                                T A,
                                                T B,
                                                T C,
                                                rocblas_int* info,
                                                const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrf_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                U A,
                                                const rocblas_int shiftA,
                                                const rocblas_int inca,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                U B,
                                                const rocblas_int shiftB,
                                                const rocblas_int incb,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                U C,
                                                const rocblas_int shiftC,
                                                const rocblas_int incc,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                T* scalars,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                T* pivotval,
                                                rocblas_int* pivotidx,
                                                rocblas_int* iipiv,
                                                rocblas_int* iinfo1,
                                                rocblas_int* iinfo2,
                                                bool optim_mem)
{
    ROCSOLVER_ENTER("geblttrf_npvt", "nb:", nb, "nblocks:", nblocks, "shiftA:", shiftA,
                    "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb, "ldb:", ldb,
                    "shiftC:", shiftC, "incc:", incc, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    T one = T(1);
    T minone = T(-1);

    rocsolver_getrf_template<BATCHED, STRIDED, T>(
        handle, nb, nb, B, shiftB, incb, ldb, strideB, nullptr, 0, 0, info, batch_count, scalars,
        work1, work2, work3, work4, pivotval, pivotidx, iipiv, iinfo1, optim_mem, false);

    for(rocblas_int k = 0; k < nblocks - 1; k++)
    {
        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nb, B, shiftB + k * ldb * nb, incb, ldb, strideB,
            nullptr, 0, C, shiftC + k * ldc * nb, incc, ldc, strideC, batch_count, work1, work2,
            work3, work4, optim_mem, false);

        rocsolver_gemm<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, rocblas_operation_none, nb, nb, nb, &minone, A,
            shiftA + k * lda * nb, inca, lda, strideA, C, shiftC + k * ldc * nb, incc, ldc, strideC,
            &one, B, shiftB + (k + 1) * ldb * nb, incb, ldb, strideB, batch_count, nullptr);

        rocsolver_getrf_template<BATCHED, STRIDED, T>(
            handle, nb, nb, B, shiftB + (k + 1) * ldb * nb, incb, ldb, strideB, nullptr, 0, 0,
            iinfo2, batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, iipiv,
            iinfo1, optim_mem, false);

        ROCSOLVER_LAUNCH_KERNEL(geblttrf_update_info, gridReset, threads, 0, stream, info, iinfo2,
                                (k + 1) * nb, batch_count);
    }

    return rocblas_status_success;
}
