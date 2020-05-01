/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQR2_H
#define ROCLAPACK_GEQR2_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"

template <typename T, bool BATCHED>
void rocsolver_geqr2_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3, size_t *size_4)
{
    size_t s1, s2;
    rocsolver_larf_getMemorySize<T,BATCHED>(rocblas_side_left,m,n,batch_count,size_1,&s1,size_3);
    rocsolver_larfg_getMemorySize<T>(n,batch_count,size_4,&s2);
    *size_2 = max(s1, s2);
}

template <typename T, typename U, bool COMPLEX = !std::is_floating_point<T>::value>
rocblas_status rocsolver_geqr2_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        const rocblas_stride strideA, T* ipiv,  
                                        const rocblas_stride strideP, const rocblas_int batch_count,
                                        T* scalars, T* work, T** workArr, T* diag)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = min(m, n);    //total number of pivots    

    for (rocblas_int j = 0; j < dim; ++j) {
        // generate Householder reflector to work on column j
        rocsolver_larfg_template(handle,
                                 m - j,                                 //order of reflector
                                 A, shiftA + idx2D(j,j,lda),            //value of alpha
                                 A, shiftA + idx2D(min(j+1,m-1),j,lda), //vector x to work on
                                 1, strideA,                            //inc of x    
                                 (ipiv + j), strideP,                   //tau
                                 batch_count, diag, work);

        // insert one in A(j,j) tobuild/apply the householder matrix 
        hipLaunchKernelGGL(set_one_diag,dim3(batch_count,1,1),dim3(1,1,1),0,stream,diag,A,shiftA+idx2D(j,j,lda),strideA);
        
        // conjugate tau
        if (COMPLEX)
            hipLaunchKernelGGL(conj_in_place<T>,dim3(1,1,batch_count),dim3(1,1,1),0,stream,1,1,ipiv,j,1,strideP);

        // Apply Householder reflector to the rest of matrix from the left 
        if (j < n - 1) {
            rocsolver_larf_template(handle,rocblas_side_left,           //side
                                    m - j,                              //number of rows of matrix to modify
                                    n - j - 1,                          //number of columns of matrix to modify    
                                    A, shiftA + idx2D(j,j,lda),         //householder vector x
                                    1, strideA,                         //inc of x
                                    (ipiv + j), strideP,                //householder scalar (alpha)
                                    A, shiftA + idx2D(j,j+1,lda),       //matrix to work on
                                    lda, strideA,                       //leading dimension
                                    batch_count, scalars, work, workArr);
        }

        // restore original value of A(j,j)
        hipLaunchKernelGGL(restore_diag,dim3(batch_count,1,1),dim3(1,1,1),0,stream,diag,A,shiftA+idx2D(j,j,lda),strideA);
        
        // restore tau
        if (COMPLEX)
            hipLaunchKernelGGL(conj_in_place<T>,dim3(1,1,batch_count),dim3(1,1,1),0,stream,1,1,ipiv,j,1,strideP);
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQR2_H */
