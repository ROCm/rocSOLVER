/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMTR_UNMTR_HPP
#define ROCLAPACK_ORMTR_UNMTR_HPP

#include "rocauxiliary_ormql_unmql.hpp"
#include "rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_ormtr_unmtr_getMemorySize(const rocblas_side side,
                                         const rocblas_fill uplo,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_AbyxORwork,
                                         size_t* size_diagORtmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_AbyxORwork = 0;
        *size_diagORtmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    rocblas_int nq = side == rocblas_side_left ? m : n;

    // requirements for calling ORMQL/UNMQL or ORMQR/UNMQR
    if(uplo == rocblas_fill_upper)
        rocsolver_ormql_unmql_getMemorySize<T, BATCHED>(side, m, n, nq, batch_count, size_scalars,
                                                        size_AbyxORwork, size_diagORtmptr,
                                                        size_trfact, size_workArr);

    else
        rocsolver_ormqr_unmqr_getMemorySize<T, BATCHED>(side, m, n, nq, batch_count, size_scalars,
                                                        size_AbyxORwork, size_diagORtmptr,
                                                        size_trfact, size_workArr);
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_ormtr_argCheck(const rocblas_side side,
                                        const rocblas_fill uplo,
                                        const rocblas_operation trans,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        const rocblas_int ldc,
                                        T A,
                                        T C,
                                        U ipiv)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if((COMPLEX && trans == rocblas_operation_transpose)
       || (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    rocblas_int nq = left ? m : n;
    if(m < 0 || n < 0 || ldc < m || lda < nq)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((nq > 0 && !A) || (nq > 0 && !ipiv) || (m * n && !C))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_template(rocblas_handle handle,
                                              const rocblas_side side,
                                              const rocblas_fill uplo,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              U C,
                                              const rocblas_int shiftC,
                                              const rocblas_int ldc,
                                              const rocblas_stride strideC,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* AbyxORwork,
                                              T* diagORtmptr,
                                              T* trfact,
                                              T** workArr)
{
    // quick return
    if(!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nq = side == rocblas_side_left ? m : n;
    rocblas_int cols, rows, colC, rowC;
    if(side == rocblas_side_left)
    {
        rows = m - 1;
        cols = n;
        rowC = 1;
        colC = 0;
    }
    else
    {
        rows = m;
        cols = n - 1;
        rowC = 0;
        colC = 1;
    }

    if(uplo == rocblas_fill_upper)
    {
        rocsolver_ormql_unmql_template<BATCHED, STRIDED, T>(
            handle, side, trans, rows, cols, nq - 1, A, shiftA + idx2D(0, 1, lda), lda, strideA,
            ipiv, strideP, C, shiftC, ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr,
            trfact, workArr);
    }
    else
    {
        rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
            handle, side, trans, rows, cols, nq - 1, A, shiftA + idx2D(1, 0, lda), lda, strideA,
            ipiv, strideP, C, shiftC + idx2D(rowC, colC, ldc), ldc, strideC, batch_count, scalars,
            AbyxORwork, diagORtmptr, trfact, workArr);
    }

    return rocblas_status_success;
}

#endif
