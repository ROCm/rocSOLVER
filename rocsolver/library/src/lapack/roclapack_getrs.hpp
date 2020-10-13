/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRS_HPP
#define ROCLAPACK_GETRS_HPP

#include "../auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_getrs_argCheck(const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        const rocblas_int lda,
                                        const rocblas_int ldb,
                                        T A,
                                        T B,
                                        const rocblas_int* ipiv,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (nrhs * n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_getrs_getMemorySize(const rocblas_int n,
                                   const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        return;
    }

    // workspace required for calling TRSM
    rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, n, nrhs, batch_count, size_work1,
                                     size_work2, size_work3, size_work4);
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocsolver_getrs_template(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
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
    // quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        return rocblas_status_success;
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host

    if(trans == rocblas_operation_none)
    {
        // first apply row interchanges to the right hand sides
        rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n, ipiv, 0, strideP,
                                    1, batch_count);

        // solve L*X = B, overwriting B with X
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower, trans,
                                     rocblas_diagonal_unit, n, nrhs, &one, A, shiftA, lda, strideA,
                                     B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
                                     work3, work4);

        // solve U*X = B, overwriting B with X
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper, trans,
                                     rocblas_diagonal_non_unit, n, nrhs, &one, A, shiftA, lda,
                                     strideA, B, shiftB, ldb, strideB, batch_count, optim_mem,
                                     work1, work2, work3, work4);
    }
    else
    {
        // solve U**T *X = B or U**H *X = B, overwriting B with X
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper, trans,
                                     rocblas_diagonal_non_unit, n, nrhs, &one, A, shiftA, lda,
                                     strideA, B, shiftB, ldb, strideB, batch_count, optim_mem,
                                     work1, work2, work3, work4);

        // solve L**T *X = B, or L**H *X = B overwriting B with X
        rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower, trans,
                                     rocblas_diagonal_unit, n, nrhs, &one, A, shiftA, lda, strideA,
                                     B, shiftB, ldb, strideB, batch_count, optim_mem, work1, work2,
                                     work3, work4);

        // then apply row interchanges to the solution vectors
        rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n, ipiv, 0, strideP,
                                    -1, batch_count);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRS_HPP */
