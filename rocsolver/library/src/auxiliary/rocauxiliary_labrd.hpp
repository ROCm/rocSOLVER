/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LABRD_H
#define ROCLAPACK_LABRD_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"


template <typename T, bool BATCHED>
void rocsolver_labrd_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    // size of scalars (constants)
    *size_1 = sizeof(T) * 3;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = sizeof(T*) * batch_count;
    else
        *size_3 = 0;

    rocsolver_larfg_getMemorySize<T>(m,n,batch_count,size_4,size_2);
}

template <typename S, typename T, typename U>
rocblas_status rocsolver_labrd_argCheck(const rocblas_int m, const rocblas_int n, const rocblas_int nb, const rocblas_int lda,
                                        const rocblas_int ldx, const rocblas_int ldy, T A, S D, S E, U tauq, U taup, T X, T Y,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if (m < 0 || n < 0 || nb < 0 || lda < m || ldx < m || ldy < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if (m*n*nb && (!A || !D || !E || !tauq || !taup || !X || !Y))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_labrd_template(rocblas_handle handle, const rocblas_int m, const rocblas_int n, const rocblas_int k,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        S* D, const rocblas_stride strideD, S* E, const rocblas_stride strideE,
                                        T* tauq, const rocblas_stride strideQ, T* taup, const rocblas_stride strideP,
                                        U X, const rocblas_int shiftX, const rocblas_int ldx, const rocblas_stride strideX,
                                        U Y, const rocblas_int shiftY, const rocblas_int ldy, const rocblas_stride strideY,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr, T* norms)
{
    return rocblas_status_not_implemented;
}

#endif /* ROCLAPACK_LABRD_H */
