/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMQL_UNMQL_HPP
#define ROCLAPACK_ORMQL_UNMQL_HPP

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_orm2l_unm2l.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_ormql_unmql_getMemorySize(const rocblas_side side,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_AbyxORwork,
                                         size_t* size_diagORtmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_AbyxORwork = 0;
        *size_diagORtmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    size_t s1, s2, unused;
    rocsolver_orm2l_unm2l_getMemorySize<T, BATCHED>(side, m, n, k, batch_count, size_scalars,
                                                    size_AbyxORwork, size_diagORtmptr, size_workArr);

    if(k > ORMxx_ORMxx_BLOCKSIZE)
    {
        rocblas_int jb = ORMxx_ORMxx_BLOCKSIZE;

        // requirements for calling larft
        rocsolver_larft_getMemorySize<T, BATCHED>(max(m, n), min(jb, k), batch_count, &unused, &s1,
                                                  &unused);

        // requirements for calling larfb
        rocsolver_larfb_getMemorySize<T, BATCHED>(side, m, n, min(jb, k), batch_count, &s2,
                                                  size_diagORtmptr, &unused);

        // size of workspace is maximum of what is needed by larft and larfb
        *size_AbyxORwork = max(s1, s2);

        // size of temporary array for triangular factor
        *size_trfact = sizeof(T) * jb * jb * batch_count;
    }
    else
        *size_trfact = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_ormql_unmql_template(rocblas_handle handle,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
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
    if(!n || !m || !k || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked variant of the algorithm
    if(k <= ORMxx_ORMxx_BLOCKSIZE)
        return rocsolver_orm2l_unm2l_template<T>(
            handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
            strideC, batch_count, scalars, AbyxORwork, diagORtmptr, workArr);

    rocblas_int ldw = ORMxx_ORMxx_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    // determine limits and indices
    bool left = (side == rocblas_side_left);
    bool transpose = (trans != rocblas_operation_none);
    rocblas_int start, step, nq, ncol, nrow;
    if(left)
    {
        nq = m;
        ncol = n;
        if(!transpose)
        {
            start = 0;
            step = 1;
        }
        else
        {
            start = (k - 1) / ldw * ldw;
            step = -1;
        }
    }
    else
    {
        nq = n;
        nrow = m;
        if(!transpose)
        {
            start = (k - 1) / ldw * ldw;
            step = -1;
        }
        else
        {
            start = 0;
            step = 1;
        }
    }

    rocblas_int i, ib;
    for(rocblas_int j = 0; j < k; j += ldw)
    {
        i = start + step * j; // current householder block
        ib = min(ldw, k - i);
        if(left)
        {
            nrow = m - k + i + ib;
        }
        else
        {
            ncol = n - k + i + ib;
        }

        // generate triangular factor of current block reflector
        rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                                    nq - k + i + ib, ib, A, shiftA + idx2D(0, i, lda), lda, strideA,
                                    ipiv + i, strideP, trfact, ldw, strideW, batch_count, scalars,
                                    AbyxORwork, workArr);

        // apply current block reflector
        rocsolver_larfb_template<BATCHED, STRIDED, T>(
            handle, side, trans, rocblas_backward_direction, rocblas_column_wise, nrow, ncol, ib, A,
            shiftA + idx2D(0, i, lda), lda, strideA, trfact, 0, ldw, strideW, C, shiftC, ldc,
            strideC, batch_count, AbyxORwork, diagORtmptr, workArr);
    }

    return rocblas_status_success;
}

#endif
