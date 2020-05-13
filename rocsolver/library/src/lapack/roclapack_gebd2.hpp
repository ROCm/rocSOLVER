/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEBD2_H
#define ROCLAPACK_GEBD2_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"

template <typename T, bool BATCHED>
void rocsolver_gebd2_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    size_t s1, s2;
    rocsolver_larf_getMemorySize<T,BATCHED>(rocblas_side_left,m,n,batch_count,size_1,&s1,size_3);
    rocsolver_larfg_getMemorySize<T>(n,batch_count,size_4,&s2);
    *size_2 = max(s1, s2);
}

template <typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_gebd2_template(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        S* D, const rocblas_stride strideD, S* E, const rocblas_stride strideE,
                                        T* tauq, const rocblas_stride strideQ, T* taup, const rocblas_stride strideP,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr, T* diag)
{
    return rocblas_status_not_implemented;
}

#endif /* ROCLAPACK_GEBD2_H */
