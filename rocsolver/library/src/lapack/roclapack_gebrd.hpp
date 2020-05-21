/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEBRD_H
#define ROCLAPACK_GEBRD_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "roclapack_gebd2.hpp"
#include "../auxiliary/rocauxiliary_labrd.hpp"


template <typename T, bool BATCHED>
void rocsolver_gebrd_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4, size_t *size_5, size_t *size_6)
{
    if (m <= GEBRD_GEBD2_SWITCHSIZE || n <= GEBRD_GEBD2_SWITCHSIZE)
    {
        rocsolver_gebd2_getMemorySize<T,BATCHED>(m,n,batch_count,size_1,size_2,size_3,size_4);
        *size_5 = 0;
        *size_6 = 0;
    }
    else
    {
        size_t s1, s2, s3, s4;
        rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
        rocblas_int d = min(m / k, n / k);
        rocsolver_gebd2_getMemorySize<T,BATCHED>(m-d*k,n-d*k,batch_count,size_1,size_2,size_3,size_4);
        rocsolver_labrd_getMemorySize<T,BATCHED>(m,n,batch_count,&s1,&s2,&s3,&s4);
        *size_1 = max(*size_1, s1);
        *size_2 = max(*size_2, s2);
        *size_3 = max(*size_3, s3);
        *size_4 = max(*size_4, s4);

        // size of matrix X
        *size_5 = m * k;
        *size_5 *= sizeof(T) * batch_count;

        // size of matrix Y
        *size_6 = n * k;
        *size_6 *= sizeof(T) * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        S* D, const rocblas_stride strideD, S* E, const rocblas_stride strideE,
                                        T* tauq, const rocblas_stride strideQ, T* taup, const rocblas_stride strideP,
                                        U X, const rocblas_int shiftX, const rocblas_int ldx, const rocblas_stride strideX,
                                        U Y, const rocblas_int shiftY, const rocblas_int ldy, const rocblas_stride strideY,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr, T* diag)
{
    return rocblas_status_not_implemented;
}

#endif /* ROCLAPACK_GEBRD_H */
