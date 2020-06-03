/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Small sizes algorithm derived from MAGMA project
 * http://icl.cs.utk.edu/magma/.
 * https://doi.org/10.1016/j.procs.2017.05.250
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETF2_H
#define ROCLAPACK_GETF2_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "../auxiliary/rocauxiliary_laswp.hpp"

#define runLUfact(DIM)                                                          \
    hipLaunchKernelGGL(LUfact_kernel<DIM,T>,grid,block,0,stream,                \
                       m,A,shiftA,lda,strideA,ipiv,shiftP,strideP,info,pivot);     

#define runLUfactSmall(DIM)                                                     \
    m == n ?                                                                    \
        hipLaunchKernelGGL(LUfact_kernel_sq<DIM,T>,grid,block,0,stream,         \
                           A,shiftA,lda,strideA,ipiv,shiftP,strideP,info,pivot):\
        hipLaunchKernelGGL(LUfact_kernel<DIM,T>,grid,block,0,stream,            \
                           m,A,shiftA,lda,strideA,ipiv,shiftP,strideP,info,pivot);
     

template <typename T, typename U>
__global__ void getf2_check_singularity(U AA, const rocblas_int shiftA, const rocblas_stride strideA,
                                        rocblas_int* ipivA, const rocblas_int shiftP,
                                        const rocblas_stride strideP, const rocblas_int j,
                                        const rocblas_int lda,
                                        T* invpivot, rocblas_int* info)
{
    int id = hipBlockIdx_x;

    T* A = load_ptr_batch<T>(AA,id,shiftA,strideA);
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

template <rocblas_int DIM, typename T, typename U>
__attribute__((amdgpu_flat_work_group_size(WaveSize,WaveSize)))
__global__ void LUfact_kernel(const rocblas_int m, U AA, const rocblas_int shiftA, const rocblas_int lda, 
                                          const rocblas_stride strideA, rocblas_int* ipivA, const rocblas_int shiftP,
                                          const rocblas_stride strideP, rocblas_int* info, const rocblas_int pivot)
{
    int id = hipBlockIdx_x;
    int myrow = hipThreadIdx_x;

    if (myrow >= m)
        return;
    
    // batch instance
    T* A = load_ptr_batch<T>(AA,id,shiftA,strideA);
    rocblas_int *ipiv = load_ptr_batch<rocblas_int>(ipivA,id,shiftP,strideP);
       
    // read corresponding row from global memory in local array
    T rA[WaveSize];
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j*lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[WaveSize];
    T pivot_value;
    T test_value;
    int pivot_index;

    // for each pivot
    for (int k = 0; k < DIM; ++k) { 
        // share current column
        common[myrow] = rA[k];
        __syncthreads();

        // search pivot index 
        pivot_index = k;
        if (pivot) {
            pivot_value = common[k];    
            for (int i = k+1; i < m; ++i) {
                test_value = common[i];
                if (std::abs(pivot_value) < std::abs(test_value)) {
                    pivot_value = test_value;
                    pivot_index = i;
                }
            } 
        }
        if (myrow == k)
            ipiv[k] = pivot_index + 1;

        // swap rows (lazy swaping)
        if (myrow == k || myrow == pivot_index)
            myrow = (myrow == k) ? pivot_index : k;

        // check singularity and scale current column 
        if (pivot_value != T(0.0)) {
            if (myrow > k)
                rA[k] /= pivot_value;
        } else {
            if (myrow == k && info[id] == 0)
                info[id] = k+1;
        }

        //share pivot row
        if (myrow == k) {
            #pragma unroll
            for (int j = k+1; j < DIM; ++j)
                common[j] = rA[j];
        }
        __syncthreads();
            
        // update trailing matrix
        if (myrow > k) {
            #pragma unroll
            for (int j = k+1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];   
        }   
    }

    // write results to global memory from local array
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        A[myrow + j*lda] = rA[j];
}


template <rocblas_int DIM, typename T, typename U>
__attribute__((amdgpu_flat_work_group_size(WaveSize,WaveSize)))
__global__ void LUfact_kernel_sq(U AA, const rocblas_int shiftA, const rocblas_int lda, 
                                          const rocblas_stride strideA, rocblas_int* ipivA, const rocblas_int shiftP,
                                          const rocblas_stride strideP, rocblas_int* info, const rocblas_int pivot)
{
    int id = hipBlockIdx_x;
    int myrow = hipThreadIdx_x;

    if (myrow >= DIM)
        return;
    
    // batch instance
    T* A = load_ptr_batch<T>(AA,id,shiftA,strideA);
    rocblas_int *ipiv = load_ptr_batch<rocblas_int>(ipivA,id,shiftP,strideP);
       
    // read corresponding row from global memory in local array
    T rA[DIM];
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j*lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    T pivot_value;
    T test_value;
    int pivot_index;

    // for each pivot
    #pragma unroll
    for (int k = 0; k < DIM; ++k) { 
        // share current column
        common[myrow] = rA[k];
        __syncthreads();

        // search pivot index 
        pivot_index = k;
        if (pivot) {
            pivot_value = common[k];    
            for (int i = k+1; i < DIM; ++i) {
                test_value = common[i];
                if (std::abs(pivot_value) < std::abs(test_value)) {
                    pivot_value = test_value;
                    pivot_index = i;
                }
            }
        } 
        if (myrow == k)
            ipiv[k] = pivot_index + 1;

        // swap rows (lazy swaping)
        if (myrow == k || myrow == pivot_index)
            myrow = (myrow == k) ? pivot_index : k;

        // check singularity and scale current column 
        if (pivot_value != T(0.0)) {
            if (myrow > k)
                rA[k] /= pivot_value;
        } else {
            if (myrow == k && info[id] == 0)
                info[id] = k+1;
        }

        //share pivot row
        if (myrow == k) {
            for (int j = k+1; j < DIM; ++j)
                common[j] = rA[j];
        }
        __syncthreads();
            
        // update trailing matrix
        if (myrow > k) {
            for (int j = k+1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];   
        }   
    }

    // write results to global memory from local array
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        A[myrow + j*lda] = rA[j];
}



template <typename T, typename U>
rocblas_status LUfact_small_sizes(rocblas_handle handle, const rocblas_int m,
                                  const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                                  const rocblas_stride strideA, rocblas_int *ipiv, const rocblas_int shiftP,
                                  const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count,
                                  const rocblas_int pivot)
{
    dim3 grid(batch_count,1,1);
    dim3 block(WaveSize,1,1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch (n) {
        case  1: runLUfactSmall( 1); break;
        case  2: runLUfactSmall( 2); break;
        case  3: runLUfactSmall( 3); break;
        case  4: runLUfactSmall( 4); break;
        case  5: runLUfactSmall( 5); break;
        case  6: runLUfactSmall( 6); break;
        case  7: runLUfactSmall( 7); break;
        case  8: runLUfactSmall( 8); break;
        case  9: runLUfactSmall( 9); break;
        case 10: runLUfactSmall(10); break;
        case 11: runLUfactSmall(11); break;
        case 12: runLUfactSmall(12); break;
        case 13: runLUfactSmall(13); break;
        case 14: runLUfactSmall(14); break;
        case 15: runLUfactSmall(15); break;
        case 16: runLUfactSmall(16); break;
        case 17: runLUfactSmall(17); break;
        case 18: runLUfactSmall(18); break;
        case 19: runLUfactSmall(19); break;
        case 20: runLUfactSmall(20); break;
        case 21: runLUfactSmall(21); break;
        case 22: runLUfactSmall(22); break;
        case 23: runLUfactSmall(23); break;
        case 24: runLUfactSmall(24); break;
        case 25: runLUfactSmall(25); break;
        case 26: runLUfactSmall(26); break;
        case 27: runLUfactSmall(27); break;
        case 28: runLUfactSmall(28); break;
        case 29: runLUfactSmall(29); break;
        case 30: runLUfactSmall(30); break;
        case 31: runLUfactSmall(31); break;
        case 32: runLUfactSmall(32); break;
        case 33: runLUfact(33); break;
        case 34: runLUfact(34); break;
        case 35: runLUfact(35); break;
        case 36: runLUfact(36); break;
        case 37: runLUfact(37); break;
        case 38: runLUfact(38); break;
        case 39: runLUfact(39); break;
        case 40: runLUfact(40); break;
        case 41: runLUfact(41); break;
        case 42: runLUfact(42); break;
        case 43: runLUfact(43); break;
        case 44: runLUfact(44); break;
        case 45: runLUfact(45); break;
        case 46: runLUfact(46); break;
        case 47: runLUfact(47); break;
        case 48: runLUfact(48); break;
        case 49: runLUfact(49); break;
        case 50: runLUfact(50); break;
        case 51: runLUfact(51); break;
        case 52: runLUfact(52); break;
        case 53: runLUfact(53); break;
        case 54: runLUfact(54); break;
        case 55: runLUfact(55); break;
        case 56: runLUfact(56); break;
        case 57: runLUfact(57); break;
        case 58: runLUfact(58); break;
        case 59: runLUfact(59); break;
        case 60: runLUfact(60); break;
        case 61: runLUfact(61); break;
        case 62: runLUfact(62); break;
        case 63: runLUfact(63); break;
        case 64: runLUfact(64); break;
    }
    
    return rocblas_status_success;
}

template <typename T>
void rocsolver_getf2_getMemorySize(const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2)
{
    // for scalars
    *size_1 = sizeof(T)*3;

    // for pivots
    *size_2 = sizeof(T)*batch_count;
}

template <typename T>
rocblas_status rocsolver_getf2_getrf_argCheck(const rocblas_int m, const rocblas_int n, const rocblas_int lda, 
                                              T A, rocblas_int *ipiv, rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((m*n && !A) || (m*n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_getf2_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        const rocblas_stride strideA, rocblas_int *ipiv, const rocblas_int shiftP, 
                                        const rocblas_stride strideP, rocblas_int* info, const rocblas_int batch_count, const rocblas_int pivot, 
                                        T* scalars, T* pivotGPU)
{
    // quick return if zero instances in batch
    if (batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    rocblas_int blocksReset = (batch_count - 1) / GETF2_BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(GETF2_BLOCKSIZE, 1, 1);
    rocblas_int dim = min(m, n);    //total number of pivots
    T* M;
    
    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info,gridReset,threads,0,stream,info,batch_count,0);

    // quick return if no dimensions
    if (m == 0 || n == 0) 
        return rocblas_status_success;

    // if very small size, use optimized LU factorization
    if (m <= WaveSize && n <= WaveSize)
        return LUfact_small_sizes<T>(handle,m,n,A,shiftA,lda,strideA,ipiv,shiftP,strideP,info,batch_count,pivot);
        
    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device);    

    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      IAMAX_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* AA = A;
    #endif

    // **** IAMAX_BATCH IS EXECUTED IN A FOR-LOOP UNTIL 
    //      FUNCITONALITY IS ENABLED. ****

    for (rocblas_int j = 0; j < dim; ++j) {
        // find pivot. Use Fortran 1-based indexing for the ipiv array as iamax does that as well!
        for (int b=0;b<batch_count;++b) {
            M = load_ptr_batch<T>(AA,b,shiftA,strideA);
            rocblas_iamax(handle, m - j, (M + idx2D(j, j, lda)), 1, 
                        (ipiv + shiftP + b*strideP + j));
        }

        // adjust pivot indices and check singularity
        hipLaunchKernelGGL(getf2_check_singularity<T>, dim3(batch_count), dim3(1), 0, stream,
                  A, shiftA, strideA, ipiv, shiftP, strideP, j, lda, pivotGPU, info);

        // Swap pivot row and j-th row 
        rocsolver_laswp_template<T>(handle, n, A, shiftA, lda, strideA, j+1, j+1, ipiv, shiftP, strideP, 1, batch_count);

        // Compute elements J+1:M of J'th column
        rocblasCall_scal<T>(handle, m-j-1, pivotGPU, 1, A, shiftA+idx2D(j+1, j, lda), 1, strideA, batch_count);

        // update trailing submatrix
        if (j < min(m, n) - 1) {
            rocblasCall_ger<false,T>(handle, m-j-1, n-j-1, scalars, 0,
                                 A, shiftA+idx2D(j+1, j, lda), 1, strideA, 
                                 A, shiftA+idx2D(j, j+1, lda), lda, strideA, 
                                 A, shiftA+idx2D(j+1, j+1, lda), lda, strideA,
                                 batch_count,nullptr); 
        }
    }

    rocblas_set_pointer_mode(handle,old_mode);    
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETF2_H */
