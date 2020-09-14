/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGQL_UNGQL_HPP
#define ROCLAPACK_ORGQL_UNGQL_HPP

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_org2l_ung2l.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orgql_ungql_getMemorySize(
    const rocblas_int m, const rocblas_int n, const rocblas_int k,
    const rocblas_int batch_count, size_t *size_1, size_t *size_2,
    size_t *size_3, size_t *size_4, size_t *size_5) {
  *size_1 = *size_2 = *size_3 = *size_4 = *size_5 = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgql_ungql_template(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP,
    const rocblas_int batch_count, T *scalars, T *work, T **workArr, T *trfact,
    T *workTrmm) {
  return rocblas_status_not_implemented;
}

#endif
