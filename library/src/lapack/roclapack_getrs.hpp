/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_GETRS_HPP
#define ROCLAPACK_GETRS_HPP

//#include <iostream>

//#include "rocsolver-export.h"
//#include <hip/hip_runtime.h>
#include <rocblas.hpp>

//#include <vector>
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_laswp.hpp"


template <typename T, typename U>
rocblas_status rocsolver_getrs_template(rocblas_handle handle, const rocblas_operation trans,
                         const rocblas_int n, const rocblas_int nrhs, U A, const rocblas_int shiftA,
                         const rocblas_int lda, const rocblas_int strideA, const rocblas_int *ipiv, const rocblas_int strideP, U B,
                         const rocblas_int shiftB, const rocblas_int ldb, const rocblas_int strideB, const rocblas_int batch_count) 
{
    // quick return
    if (n == 0 || nrhs == 0 || batch_count == 0) {
      return rocblas_status_success;
    }

    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* AA[batch_count];
        T* BB[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
        hipMemcpy(BB, B, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
        T* BB = B;
    #endif

    //constants to use when calling rocablas functions
    T one = 1;            //constant 1 in host
    T* oneInt;            //constant 1 in device
    hipMalloc(&oneInt, sizeof(T));
    hipMemcpy(oneInt, &one, sizeof(T), hipMemcpyHostToDevice);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *Ap, *Bp;

    if (trans == rocblas_operation_none) {

        // first apply row interchanges to the right hand sides
        rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n, ipiv, 0, strideP, 1, batch_count);

/*rocblas_int size_B = ldb * nrhs;
std::vector<T> hBRes(size_B);
hipMemcpy(hBRes.data(), B, sizeof(T) * size_B, hipMemcpyDeviceToHost);
printf("\n");
for(int i=0;i<n;++i) {
    for(int j=0;j<nrhs;++j) {
        printf("%2.15f ",hBRes[i+j*lda]);
    }
    printf("\n");
}*/


        for (int b = 0; b < batch_count; ++b) {
            Ap = load_ptr_batch<T>(AA,shiftA,b,strideA);
            Bp = load_ptr_batch<T>(BB,shiftB,b,strideB);
            
            // solve L*X = B, overwriting B with X
            rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_lower,
                    trans, rocblas_diagonal_unit, n, nrhs,
                    oneInt, Ap, lda, Bp, ldb);

            // solve U*X = B, overwriting B with X
            rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_upper,
                    trans, rocblas_diagonal_non_unit, n, nrhs,
                    oneInt, Ap, lda, Bp, ldb);
        }
    
    } else {

        for (int b = 0; b < batch_count; ++b) {
            Ap = load_ptr_batch<T>(AA,shiftA,b,strideA);
            Bp = load_ptr_batch<T>(BB,shiftB,b,strideB);
            
            // solve U**T *X = B or U**H *X = B, overwriting B with X
            rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_upper, trans,
                    rocblas_diagonal_non_unit, n, nrhs,
                    oneInt, Ap, lda, Bp, ldb);

            // solve L**T *X = B, or L**H *X = B overwriting B with X
            rocblas_trsm<T>(handle, rocblas_side_left, rocblas_fill_lower, trans,
                    rocblas_diagonal_unit, n, nrhs, oneInt,
                    Ap, lda, Bp, ldb);
        }

        // then apply row interchanges to the solution vectors
        rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n, ipiv, 0, strideP, -1, batch_count);
    }

    hipFree(oneInt);

    return rocblas_status_success;
}


#endif /* ROCLAPACK_GETRS_HPP */
