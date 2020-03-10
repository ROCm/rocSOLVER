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
#include "../auxiliary/rocauxiliary_larf.hpp"

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
 
    return rocblas_status_success;
}

#endif
