/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQLF_H
#define ROCLAPACK_GEQLF_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "roclapack_geql2.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"
#include "../auxiliary/rocauxiliary_larfb.hpp"

template <typename T, bool BATCHED>
void rocsolver_geqlf_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4, size_t *size_5)
{
    size_t s1,s2,s3;
    rocsolver_geql2_getMemorySize<T,BATCHED>(m,n,batch_count,size_1,&s1,size_3,size_4);
    if (m <= GEQLF_GEQL2_SWITCHSIZE || n <= GEQLF_GEQL2_SWITCHSIZE) {
        *size_2 = s1;
        *size_5 = 0;
    } else {
        rocblas_int jb = GEQLF_GEQL2_BLOCKSIZE;
        rocsolver_larft_getMemorySize<T>(jb,batch_count,&s2);
        rocsolver_larfb_getMemorySize<T>(rocblas_side_left,m,n-jb,jb,batch_count,&s3);
        *size_2 = max(s1,max(s2,s3));
        *size_5 = sizeof(T)*jb*jb*batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geqlf_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        const rocblas_stride strideA, T* ipiv,  
                                        const rocblas_stride strideP, const rocblas_int batch_count,
                                        T* scalars, T* work, T** workArr, T* diag, T* trfact)
{
    return rocblas_status_not_implemented;
}

#endif /* ROCLAPACK_GEQLF_H */
