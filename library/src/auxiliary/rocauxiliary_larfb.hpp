/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2013
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARFB_HPP
#define ROCLAPACK_LARFB_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"
#include <vector>


template <typename T, typename U>
__global__ void copymatA1(const rocsolver_int ldw, const rocsolver_int order, U A, const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int strideA, T* work) 
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocsolver_int strideW = ldw*order;

    if (i < ldw && j < order) {
        T *Ap, *Wp;
        Wp = work + b*strideW;
        Ap = load_ptr_batch<T>(A,shiftA,b,strideA);

        Wp[i + j*ldw] = Ap[i + j*lda];
    }
}

template <typename T, typename U>
__global__ void addmatA1(const rocsolver_int ldw, const rocsolver_int order, U A, const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int strideA, T* work) 
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocsolver_int strideW = ldw*order;

    if (i < ldw && j < order) {
        T *Ap, *Wp;
        Wp = work + b*strideW;
        Ap = load_ptr_batch<T>(A,shiftA,b,strideA);

        Ap[i + j*lda] -= Wp[i + j*ldw];    
    }
}

template <typename T, typename U>
rocblas_status rocsolver_larfb_template(rocsolver_handle handle, const rocsolver_side side, 
                                        const rocsolver_operation trans, const rocsolver_direct direct, 
                                        const rocsolver_storev storev,
                                        const rocsolver_int m, const rocsolver_int n,
                                        const rocsolver_int k, U V, const rocblas_int shiftV, const rocsolver_int ldv, 
                                        const rocsolver_int strideV, T *F, const rocsolver_int shiftF,
                                        const rocsolver_int ldf, const rocsolver_int strideF, 
                                        U A, const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int strideA,
                                        const rocsolver_int batch_count)
{
    // quick return
    if (!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *Vp, *Ap, *Fp;

    //constants to use when calling rocablas functions
    T minone = -1;                //constant -1 in host
    T* minoneInt;                 //constant -1 in device
    hipMalloc(&minoneInt, sizeof(T));
    hipMemcpy(minoneInt, &minone, sizeof(T), hipMemcpyHostToDevice);
    T one = 1;                 //constant 1 in host
    T* oneInt;                 //constant 1 in device
    hipMalloc(&oneInt, sizeof(T));
    hipMemcpy(oneInt, &one, sizeof(T), hipMemcpyHostToDevice);

    T* FF = F;
    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* VV[batch_count];
        hipMemcpy(VV, V, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* VV = V;
        T* AA = A;
    #endif

    //determine the side, size of workspace
    //and whether V is trapezoidal
    rocsolver_operation transp; 
    rocsolver_fill uploV;
    bool trap;
    rocblas_int order, ldw;
    bool colwise = (storev == rocsolver_column_wise); 
    bool leftside = (side == rocblas_side_left);
    size_t offsetV;
    
    if (leftside) {
        order = n;
        ldw = k;
        trap = (m > k);
    } else {
        order = k;
        ldw = m;
        trap = (n > k);
    }
    if (colwise) {
        uploV = rocblas_fill_lower;
        offsetV = idx2D(k,0,ldv);
        if (leftside) 
            transp = rocblas_operation_transpose;
        else 
            transp = rocblas_operation_none;
    } else {
        uploV = rocblas_fill_upper;
        offsetV = idx2D(0,k,ldv);
        if (leftside) 
            transp = rocblas_operation_none;
        else 
            transp = rocblas_operation_transpose;
    }

    //memory in GPU (workspace)
    rocblas_int strideW = ldw*order;
    T *work;
    hipMalloc(&work, sizeof(T)*strideW*batch_count);

    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****

    //copy A1 to work
    rocblas_int blocksx = (order - 1)/32 + 1;
    rocblas_int blocksy = (ldw - 1)/32 + 1;
    hipLaunchKernelGGL(copymatA1,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,ldw,order,A,shiftA,lda,strideA,work);
    
    // BACKWARD DIRECTION TO BE IMPLEMENTED...
    rocsolver_fill uploT = rocblas_fill_upper;
    if (direct == rocsolver_backward_direction)
        return rocblas_status_not_implemented;
    
    //compute:
    // V1' * A1, or
    //   or 
    // A1 * V1
    for (int b=0;b<batch_count;++b) {
        Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
        rocblas_trmm(handle,side,uploV,transp,rocblas_diagonal_unit,ldw,order,oneInt,Vp,ldv,(work + b*strideW),ldw);
    }

    // compute:
    // V1' * A1 + V2' * A2 
    //        or 
    // A1 * V1 + A2 * V2
    if (trap) { 
        for (int b=0;b<batch_count;++b) {
            Ap = load_ptr_batch<T>(AA,shiftA,b,strideA);
            Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
            if (leftside) { 
                rocblas_gemm(handle,transp,rocblas_operation_none,ldw,order,m-k,oneInt,
                             (Vp + offsetV),ldv,
                             (Ap + idx2D(k,0,lda)),lda,
                             oneInt,(work + b*strideW),ldw);
            } else {
                rocblas_gemm(handle,rocblas_operation_none,transp,ldw,order,n-k,oneInt,
                             (Ap + idx2D(0,k,lda)),lda,
                             (Vp + offsetV),ldv,
                             oneInt,(work + b*strideW),ldw);
            }
        }
    }

    // compute: 
    // trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // (A1 * V1 + A2 * V2) * trans(T)    
    for (int b=0;b<batch_count;++b) {
        Fp = load_ptr_batch<T>(FF,shiftF,b,strideF);
        rocblas_trmm(handle,side,uploT,trans,rocblas_diagonal_non_unit,ldw,order,oneInt,Fp,ldf,(work + b*strideW),ldw);
    }

    // compute:
    // A2 - V2 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // A2 - (A1 * V1 + A2 * V2) * trans(T) * V2'    
    if (transp == rocblas_operation_transpose)
        transp = rocblas_operation_none;
    else
        transp = rocblas_operation_transpose;

    if (trap) {
        for (int b=0;b<batch_count;++b) {
            Ap = load_ptr_batch<T>(AA,shiftA,b,strideA);
            Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
            if (leftside) { 
                rocblas_gemm(handle,transp,rocblas_operation_none,m-k,order,ldw,minoneInt,
                             (Vp + offsetV),ldv,
                             (work + b*strideW),ldw,
                             oneInt,(Ap + idx2D(k,0,lda)),lda);
            } else {
                rocblas_gemm(handle,rocblas_operation_none,transp,ldw,n-k,order,minoneInt,
                             (work + b*strideW),ldw,
                             (Vp + offsetV),ldv,
                             oneInt,(Ap + idx2D(0,k,lda)),lda);
            }
        }
    }
        
    // compute:
    // V1 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // (A1 * V1 + A2 * V2) * trans(T) * V1'    
    for (int b=0;b<batch_count;++b) {
        Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
        rocblas_trmm(handle,side,uploV,transp,rocblas_diagonal_unit,ldw,order,oneInt,Vp,ldv,(work + b*strideW),ldw);
    }
    
    // compute:
    // A1 - V1 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // A1 - (A1 * V1 + A2 * V2) * trans(T) * V1'
    hipLaunchKernelGGL(addmatA1,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,ldw,order,A,shiftA,lda,strideA,work);
    
    hipFree(minoneInt);
    hipFree(oneInt);
    hipFree(work);

    return rocblas_status_success;
}

#endif
