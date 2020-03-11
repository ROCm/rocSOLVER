/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMQR_HPP
#define ROCLAPACK_ORMQR_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_orm2r.hpp"
#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"

template <typename T, typename U>
rocblas_status rocsolver_ormqr_template(rocsolver_handle handle, const rocsolver_side side, const rocsolver_operation trans, 
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

    // if the matrix is small, use the unblocked variant of the algorithm
    if (k <= ORMQR_ORM2R_BLOCKSIZE) 
        return rocsolver_orm2r_template<T>(handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc, strideC, batch_count);

    //memory in GPU (workspace)
    T* work;
    rocblas_int ldw = ORMQR_ORM2R_BLOCKSIZE;
    rocblas_int strideW = ldw *ldw;
    hipMalloc(&work, sizeof(T)*strideW*batch_count);    

    // determine limits and indices
    bool left = (side == rocblas_side_left);
    bool transpose = (trans == rocblas_operation_transpose);
    int start, step, ncol, nrow, ic, jc, order;
    if (left) {
        ncol = n;
        order = m;
        jc = 0;
        if (transpose) {
            start = 0;
            step = 1;
        } else {
            start = (k-1)/ldw * ldw;
            step = -1;
        }
    } else {
        nrow = m;
        order = n;
        ic = 0;
        if (transpose) {
            start = (k-1)/ldw * ldw;
            step = -1;
        } else {
            start = 0;
            step = 1;
        }
    }

    int i;
    for (int j = 0; j < k; j += ldw) {
        i = start + step*j;    // current householder block
        if (left) {
            nrow = m - i;
            ic = i;
        } else {
            ncol = n - i;
            jc = i;
        }

        // generate triangular factor of current block reflector
        rocsolver_larft_template(handle,rocsolver_forward_direction,rocsolver_column_wise,
                                 order-i,min(ldw,k-i),
                                 A, shiftA + idx2D(i,i,lda),lda, strideA,
                                 ipiv + i, strideP,
                                 work,ldw,strideW,
                                 batch_count);

        // apply current block reflector
        rocsolver_larfb_template(handle,side,trans,
                                 rocsolver_forward_direction,rocsolver_column_wise,
                                 nrow,ncol,min(ldw,k-i),
                                 A, shiftA + idx2D(i,i,lda),lda, strideA,
                                 work,0,ldw,strideW,
                                 C, shiftC + idx2D(ic,jc,ldc),ldc,strideC,
                                 batch_count);
    }
 
    return rocblas_status_success;
}

#endif
