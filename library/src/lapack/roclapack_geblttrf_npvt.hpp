/************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "roclapack_geblttrf_npvt_strided_batched.hpp"
#include "roclapack_geblttrf_npvt_batched.hpp"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrf_npvt_getMemorySize(const rocblas_int nb,
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
    if (handle == nullptr) {
       return( rocblas_status_invalid_handle );
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
    ROCSOLVER_ENTER("geblttrf_npvt", "nb:", nb, "nblocks:", nblocks, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || batch_count == 0)
        return rocblas_status_success;


    bool constexpr is_strided_batched = (!BATCHED) && STRIDED;
    bool constexpr is_batched_only = BATCHED && (!STRIDED);
    bool constexpr is_no_batched = (!BATCHED) && (!STRIDED);
    
    rocblas_status istat = rocblas_status_not_implemented;
    if constexpr (is_no_batched ) {
       // -------------------------------------------------
       // treat non batched case  as trivial strided batched 
       // of batch_count == 1
       // -------------------------------------------------
       const rocblas_int dummy_batch_count = 1;

       const rocblas_stride lnblocks = nblocks;
       const rocblas_stride dummy_strideA = lda * nb * lnblocks;
       const rocblas_stride dummy_strideB = ldb * nb * lnblocks;
       const rocblas_stride dummy_strideC = ldc * nb * lnblocks;

       T * Ap = (T *) A;
       T * Bp = (T *) B;
       T * Cp = (T *) C;

       istat = rocsolver_geblttrf_npvt_strided_batched_template( 
                        handle,
                        nb, nblocks, 
                        Ap,  lda, dummy_strideA,
                        Bp,  ldb, dummy_strideB,
                        Cp,  ldc, dummy_strideC,
                        info,
                        dummy_batch_count);
      }
   else if constexpr (is_strided_batched) {
       T * Ap = (T *) A;
       T * Bp = (T *) B;
       T * Cp = (T *) C;

       istat = rocsolver_geblttrf_npvt_strided_batched_template(
                        handle,
                        nb, nblocks,
                        Ap, lda, strideA,
                        Bp, ldb, strideB,
                        Cp, ldc, strideC,
                        info,
                        batch_count );
      }
   else if constexpr (is_batched_only) {
       T ** A_array =  (T **) A;
       T ** B_array =  (T **) B;
       T ** C_array =  (T **) C;
       istat =  rocsolver_geblttrf_npvt_batched_template(
                        handle,
                        nb, nblocks,
                        A_array, lda, 
                        B_array, ldb, 
                        C_array, ldc, 
                        info,
                        batch_count);
      };


    return( istat ); 
}
