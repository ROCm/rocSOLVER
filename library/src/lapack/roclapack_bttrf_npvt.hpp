/************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_bttrf_npvt_getMemorySize(const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int batch_count,
                                        size_t* size_work)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || batch_count == 0)
    {
        // TODO: set workspace sizes to zero
        *size_work = 0;
        return;
    }

    // TODO: calculate workspace sizes
    *size_work = 0;
}

template <typename T>
rocblas_status rocsolver_bttrf_npvt_argCheck(rocblas_handle handle,
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
    // N/A

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
rocblas_status rocsolver_bttrf_npvt_template(rocblas_handle handle,
                                             const rocblas_int nb,
                                             const rocblas_int nblocks,
                                             U A,
                                             const rocblas_int shiftA,
                                             const rocblas_int lda,
                                             const rocblas_stride strideA,
                                             U B,
                                             const rocblas_int shiftB,
                                             const rocblas_int ldb,
                                             const rocblas_stride strideB,
                                             U C,
                                             const rocblas_int shiftC,
                                             const rocblas_int ldc,
                                             const rocblas_stride strideC,
                                             rocblas_int* info,
                                             const rocblas_int batch_count,
                                             void* work)
{
    ROCSOLVER_ENTER("bttrf_npvt", "nb:", nb, "nblocks:", nblocks, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    return rocblas_status_not_implemented;
}
