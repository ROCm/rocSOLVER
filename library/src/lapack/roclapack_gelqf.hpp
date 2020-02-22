/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELQF_H
#define ROCLAPACK_GELQF_H

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "definitions.h"
#include "helpers.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"
#include "roclapack_gelq2.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"
#include "../auxiliary/rocauxiliary_larfb.hpp"
#include <vector>

template <typename T, typename U>
rocblas_status rocsolver_gelqf_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        rocblas_int const strideA, T* ipiv,  
                                        const rocblas_int strideP, const rocblas_int batch_count)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the algorithm
    if (m <= GEQRF_GEQR2_SWITCHSIZE || n <= GEQRF_GEQR2_SWITCHSIZE) 
        return rocsolver_gelq2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count);
    
    rocblas_int dim = min(m, n);    //total number of pivots
    rocblas_int jb, j = 0;

    //memory in GPU (workspace)
    T* work;
    rocblas_int ldw = GEQRF_GEQR2_BLOCKSIZE;
    rocblas_int strideW = ldw *ldw;
    hipMalloc(&work, sizeof(T)*strideW*batch_count);

    while (j < dim - GEQRF_GEQR2_SWITCHSIZE) {
        // Factor diagonal and subdiagonal blocks 
        jb = min(dim - j, GEQRF_GEQR2_BLOCKSIZE);  //number of rows in the block
        rocsolver_gelq2_template<T>(handle, jb, n-j, A, shiftA + idx2D(j,j,lda), lda, strideA, (ipiv + j), strideP, batch_count);

        //apply transformation to the rest of the matrix
        if (j + jb < m) {
            
            //compute block reflector
            rocsolver_larft_template<T>(handle, rocsolver_forward_direction, 
                                        rocsolver_row_wise, n-j, jb, 
                                        A, shiftA + idx2D(j,j,lda), lda, strideA, 
                                        (ipiv + j), strideP,
                                        work, ldw, strideW, batch_count);

            //apply the block reflector
            rocsolver_larfb_template<T>(handle,rocblas_side_right,rocblas_operation_none,
                                        rocsolver_forward_direction,rocsolver_row_wise,
                                        m-j-jb, n-j, jb,
                                        A, shiftA + idx2D(j,j,lda), lda, strideA,
                                        work, 0, ldw, strideW,
                                        A, shiftA + idx2D(j+jb,j,lda), lda, strideA, batch_count);

        }
        j += GEQRF_GEQR2_BLOCKSIZE;
    }

    //factor last block
    if (j < dim) 
        rocsolver_gelq2_template<T>(handle, m-j, n-j, A, shiftA + idx2D(j,j,lda), lda, strideA, (ipiv + j), strideP, batch_count);
        
    hipFree(work);

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GELQF_H */
