/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMLQ_HPP
#define ROCLAPACK_ORMLQ_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include "rocauxiliary_ormlq.hpp"
#include "rocauxiliary_ormqr.hpp"

template <typename T, typename U>
rocblas_status rocsolver_ormbr_template(rocsolver_handle handle, const rocsolver_storev storev, const rocsolver_side side, const rocsolver_operation trans, 
                                   const rocsolver_int m, const rocsolver_int n, 
                                   const rocsolver_int k, U A, const rocsolver_int shiftA, const rocsolver_int lda, 
                                   const rocsolver_int strideA, T* ipiv, 
                                   const rocsolver_int strideP, U C, const rocsolver_int shiftC, const rocsolver_int ldc,
                                   const rocsolver_int strideC, const rocsolver_int batch_count)
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
    if (storev == rocsolver_column_wise) {
        if (nq >= k) {
            rocsolver_ormqr_template<T>(handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, 
                                        C, shiftC, ldc, strideC, batch_count);
        } else {
            // shift the householder vectors provided by gebrd as they come below the first subdiagonal
            rocsolver_ormqr_template<T>(handle, side, trans, rows, cols, nq-1, 
                                        A, shiftA + idx2D(1,0,lda), lda, strideA, ipiv, strideP, 
                                        C, shiftC + idx2D(rowC,colC,ldc), ldc, strideC, batch_count);
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
            rocsolver_ormlq_template<T>(handle, side, transP, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, 
                                        C, shiftC, ldc, strideC, batch_count);
        } else {
            // shift the householder vectors provided by gebrd as they come above the first superdiagonal
            rocsolver_ormlq_template<T>(handle, side, transP, rows, cols, nq-1, 
                                        A, shiftA + idx2D(0,1,lda), lda, strideA, ipiv, strideP, 
                                        C, shiftC + idx2D(rowC,colC,ldc), ldc, strideC, batch_count);
        }
    }

 
    return rocblas_status_success;
}

#endif
