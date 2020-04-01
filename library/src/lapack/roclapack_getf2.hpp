/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETF2_H
#define ROCLAPACK_GETF2_H

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_laswp.hpp"

template <typename T, typename U>
inline __global__ void getf2_check_singularity(U AA, const rocblas_int shiftA, const rocblas_stride strideA,
                                        rocblas_int* ipivA, const rocblas_int shiftP,
                                        const rocblas_stride strideP, const rocblas_int j,
                                        const rocblas_int lda,
                                        T* invpivot, rocblas_int* info)
{
    int id = hipBlockIdx_x;

    T* A = load_ptr_batch<T>(AA,shiftA,id,strideA);
    rocblas_int *ipiv = ipivA + id*strideP + shiftP;

    ipiv[j] += j;           //update the pivot index
    if (A[j * lda + ipiv[j] - 1] == 0) {
        invpivot[id] = 1.0;
        if (info[id] == 0)
           info[id] = j + 1;   //use Fortran 1-based indexing
    }
    else
        invpivot[id] = 1.0 / A[j * lda + ipiv[j] - 1];
}


template <typename T, typename U>
rocblas_status rocsolver_getf2_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        const rocblas_stride strideA, rocblas_int *ipiv, const rocblas_int shiftP, 
                                        const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;
        
    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      IAMAX_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    //constants to use when calling rocablas functions
    rocblas_int oneInt = 1;       //constant 1 in host
    T minone = -1;                //constant -1 in host
    T* minoneInt;                 //constant -1 in device
    hipMalloc(&minoneInt, sizeof(T));
    hipMemcpy(minoneInt, &minone, sizeof(T), hipMemcpyHostToDevice);

    // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
    //pivoting info in device (to avoid continuous synchronization with CPU)
    T *pivotGPU; 
    hipMalloc(&pivotGPU, sizeof(T)*batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocksPivot = (n - 1) / GETF2_BLOCKSIZE + 1;
    rocblas_int blocksReset = (batch_count - 1) / GETF2_BLOCKSIZE + 1;
    dim3 gridPivot(blocksPivot, batch_count, 1);
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(GETF2_BLOCKSIZE, 1, 1);
    rocblas_int dim = min(m, n);    //total number of pivots
    T* M;

    //info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);
    

    // **** IAMAX_BATCH IS EXECUTED IN A FOR-LOOP UNTIL 
    //      FUNCITONALITY IS ENABLED. ****

    for (rocblas_int j = 0; j < dim; ++j) {
        // find pivot. Use Fortran 1-based indexing for the ipiv array as iamax does that as well!
        for (int b=0;b<batch_count;++b) {
            M = load_ptr_batch<T>(AA,shiftA,b,strideA);
            rocblas_iamax(handle, m - j, (M + idx2D(j, j, lda)), 1, 
                        (ipiv + shiftP + b*strideP + j));
        }

        // adjust pivot indices and check singularity
        hipLaunchKernelGGL(getf2_check_singularity<T>, dim3(batch_count), dim3(1), 0, stream,
                  A, shiftA, strideA, ipiv, shiftP, strideP, j, lda, pivotGPU, info);

        // Swap pivot row and j-th row 
        rocsolver_laswp_template<T>(handle, n, A, shiftA, lda, strideA, j+1, j+1, ipiv, shiftP, strideP, 1, batch_count);

        // Compute elements J+1:M of J'th column
        rocblas_scal<T>(handle, m-j-1, pivotGPU, 1, A, shiftA+idx2D(j+1, j, lda), 1, strideA, batch_count);

        // update trailing submatrix
        if (j < min(m, n) - 1) {
            rocblas_ger<false,T>(handle, m-j-1, n-j-1, minoneInt, 0,
                                 A, shiftA+idx2D(j+1, j, lda), 1, strideA, 
                                 A, shiftA+idx2D(j, j+1, lda), lda, strideA, 
                                 A, shiftA+idx2D(j+1, j+1, lda), lda, strideA,
                                 batch_count); 
        }
    }

    hipFree(pivotGPU);
    hipFree(minoneInt);

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETF2_H */
