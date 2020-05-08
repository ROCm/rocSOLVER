/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQL2_H
#define ROCLAPACK_GEQL2_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"

template <typename T, bool BATCHED>
void rocsolver_geql2_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    size_t s1, s2;
    rocsolver_larf_getMemorySize<T,BATCHED>(rocblas_side_left,m,n,batch_count,size_1,&s1,size_3);
    rocsolver_larfg_getMemorySize<T>(n,batch_count,size_4,&s2);
    *size_2 = max(s1, s2);
}

template <typename T, typename U>
rocblas_status rocsolver_geql2_geqlf_argCheck(const rocblas_int m, const rocblas_int n, const rocblas_int lda, 
                                              T A, U ipiv, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((m*n && !A) || (m*n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_geql2_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        const rocblas_stride strideA, T* ipiv,  
                                        const rocblas_stride strideP, const rocblas_int batch_count,
                                        T* scalars, T* work, T** workArr, T* diag)
{
    return rocblas_status_not_implemented;
}

#endif /* ROCLAPACK_GEQL2_H */
