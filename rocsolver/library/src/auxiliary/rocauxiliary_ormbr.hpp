/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMBR_HPP
#define ROCLAPACK_ORMBR_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_ormlq.hpp"
#include "../auxiliary/rocauxiliary_ormqr.hpp"

template <typename T, bool BATCHED>
void rocsolver_ormbr_getMemorySize(const rocblas_storev storev, const rocblas_side side, const rocblas_int m, const rocblas_int n, const rocblas_int k, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    rocblas_int nq = side == rocblas_side_left ? m : n;
    if (storev == rocblas_column_wise)
        rocsolver_ormqr_getMemorySize<T,BATCHED>(side,m,n,min(nq,k),batch_count,size_1,size_2,size_3,size_4);
    else
        rocsolver_ormlq_getMemorySize<T,BATCHED>(side,m,n,min(nq,k),batch_count,size_1,size_2,size_3,size_4);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_ormbr_template(rocblas_handle handle, const rocblas_storev storev, const rocblas_side side, const rocblas_operation trans, 
                                   const rocblas_int m, const rocblas_int n, 
                                   const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                   const rocblas_stride strideA, T* ipiv, 
                                   const rocblas_stride strideP, U C, const rocblas_int shiftC, const rocblas_int ldc,
                                   const rocblas_stride strideC, const rocblas_int batch_count,
                                   T* scalars, T* work, T** workArr, T* trfact)
{
    // quick return
    if (!n || !m || !k || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nq = side == rocblas_side_left ? m : n;
    rocblas_int cols, rows, colC, rowC;
    if (side == rocblas_side_left) {
        rows = m - 1;
        cols = n;
        rowC = 1;
        colC = 0;
    } else {
        rows = m;
        cols = n - 1;
        rowC = 0;
        colC = 1;
    }
    
    // if column-wise, apply the orthogonal matrix Q generated in the bi-diagonalization
    // gebrd to a general matrix C
    if (storev == rocblas_column_wise) {
        if (nq >= k) {
            rocsolver_ormqr_template<BATCHED,STRIDED,T>(handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, 
                                        C, shiftC, ldc, strideC, batch_count, scalars, work, workArr, trfact);
        } else {
            // shift the householder vectors provided by gebrd as they come below the first subdiagonal
            rocsolver_ormqr_template<BATCHED,STRIDED,T>(handle, side, trans, rows, cols, nq-1, 
                                        A, shiftA + idx2D(1,0,lda), lda, strideA, ipiv, strideP, 
                                        C, shiftC + idx2D(rowC,colC,ldc), ldc, strideC, batch_count, scalars, work, workArr, trfact);
        }
    }

    // if row-wise, apply the orthogonal matrix P generated in the bi-diagonalization
    // gebrd to a general matrix C
    else {
        rocblas_operation transP;
        if (trans == rocblas_operation_none)
            transP = rocblas_operation_transpose;
        else
            transP = rocblas_operation_none;
        if (nq > k) {
            rocsolver_ormlq_template<BATCHED,STRIDED,T>(handle, side, transP, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, 
                                        C, shiftC, ldc, strideC, batch_count, scalars, work, workArr, trfact);
        } else {
            // shift the householder vectors provided by gebrd as they come above the first superdiagonal
            rocsolver_ormlq_template<BATCHED,STRIDED,T>(handle, side, transP, rows, cols, nq-1, 
                                        A, shiftA + idx2D(0,1,lda), lda, strideA, ipiv, strideP, 
                                        C, shiftC + idx2D(rowC,colC,ldc), ldc, strideC, batch_count, scalars, work, workArr, trfact);
        }
    }

 
    return rocblas_status_success;
}

#endif
