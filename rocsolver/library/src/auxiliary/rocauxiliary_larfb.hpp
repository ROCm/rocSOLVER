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

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U>
__global__ void copymatA1(const rocblas_int ldw, const rocblas_int order, U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, T* work) 
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw)*order;

    if (i < ldw && j < order) {
        T *Ap, *Wp;
        Wp = work + b*strideW;
        Ap = load_ptr_batch<T>(A,b,shiftA,strideA);

        Wp[i + j*ldw] = Ap[i + j*lda];
    }
}

template <typename T, typename U>
__global__ void addmatA1(const rocblas_int ldw, const rocblas_int order, U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, T* work) 
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw)*order;

    if (i < ldw && j < order) {
        T *Ap, *Wp;
        Wp = work + b*strideW;
        Ap = load_ptr_batch<T>(A,b,shiftA,strideA);

        Ap[i + j*lda] -= Wp[i + j*ldw];    
    }
}

template <typename T, bool BATCHED>
void rocsolver_larfb_getMemorySize(const rocblas_side side, const rocblas_int m, const rocblas_int n, const rocblas_int k, const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2)
{
    // size of workspace
    if (side == rocblas_side_left)
        *size_1 = n;
    else
        *size_1 = m;
    *size_1 *= sizeof(T)*k*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_2 = sizeof(T*)*batch_count;
    else
        *size_2 = 0;
}

template <typename T>
void rocsolver_larfb_getMemorySize(const rocblas_side side, const rocblas_int m, const rocblas_int n, const rocblas_int k, const rocblas_int batch_count,
                                   size_t *size)
{
    // size of workspace
    if (side == rocblas_side_left)
        *size = n;
    else
        *size = m;
    *size *= sizeof(T)*k*batch_count;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_larfb_template(rocblas_handle handle, const rocblas_side side, 
                                        const rocblas_operation trans, const rocblas_direct direct, 
                                        const rocblas_storev storev,
                                        const rocblas_int m, const rocblas_int n,
                                        const rocblas_int k, U V, const rocblas_int shiftV, const rocblas_int ldv, 
                                        const rocblas_stride strideV, T *F, const rocblas_int shiftF,
                                        const rocblas_int ldf, const rocblas_stride strideF, 
                                        U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                        const rocblas_int batch_count, T* work, T** workArr)
{
    // quick return
    if (!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *Vp, *Fp;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

    //constants to use when calling rocablas functions
    T minone = -1;               
    T one = 1;               

    // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
    //      TRMM_BATCH FUNCTIONALITY IS ENABLED. ****
    #ifdef batched
        T* VV[batch_count];
        hipMemcpy(VV, V, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* VV = V;
    #endif

    //determine the side, size of workspace
    //and whether V is trapezoidal
    bool trap;
    bool colwise = (storev == rocblas_column_wise); 
    bool leftside = (side == rocblas_side_left);
    rocblas_operation transt = (leftside && trans == rocblas_operation_transpose ?
                                rocblas_operation_conjugate_transpose : trans);
    rocblas_operation transp; 
    rocblas_fill uploV;
    rocblas_int order, ldw;
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
            transp = rocblas_operation_conjugate_transpose;
        else 
            transp = rocblas_operation_none;
    } else {
        uploV = rocblas_fill_upper;
        offsetV = idx2D(0,k,ldv);
        if (leftside) 
            transp = rocblas_operation_none;
        else 
            transp = rocblas_operation_conjugate_transpose;
    }
    rocblas_stride strideW = rocblas_stride(ldw)*order;

    // **** TRMM_BATCH IS EXECUTED IN A FOR-LOOP UNTIL 
    //      FUNCITONALITY IS ENABLED ****

    //copy A1 to work
    rocblas_int blocksx = (order - 1)/32 + 1;
    rocblas_int blocksy = (ldw - 1)/32 + 1;
    hipLaunchKernelGGL(copymatA1,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,ldw,order,A,shiftA,lda,strideA,work);
    
    // BACKWARD DIRECTION TO BE IMPLEMENTED...
    rocblas_fill uploT = rocblas_fill_upper;
    if (direct == rocblas_backward_direction)
        return rocblas_status_not_implemented;
    
    //compute:
    // V1' * A1, or
    //   or 
    // A1 * V1
    for (int b=0;b<batch_count;++b) {
        Vp = load_ptr_batch<T>(VV,b,shiftV,strideV);
        rocblas_trmm(handle,side,uploV,transp,rocblas_diagonal_unit,ldw,order,&one,Vp,ldv,(work + b*strideW),ldw);
    }

    // compute:
    // V1' * A1 + V2' * A2 
    //        or 
    // A1 * V1 + A2 * V2
    if (trap) { 
        if (leftside) { 
            rocblasCall_gemm<BATCHED,STRIDED,T>(handle, transp, rocblas_operation_none,
                                            ldw, order, m-k, &one,
                                            V, shiftV+offsetV, ldv, strideV,
                                            A, shiftA+idx2D(k,0,lda), lda, strideA, &one,
                                            work, 0, ldw, strideW, batch_count, workArr);   
        } else {
            rocblasCall_gemm<BATCHED,STRIDED,T>(handle, rocblas_operation_none, transp,
                                            ldw, order, n-k, &one,
                                            A, shiftA+idx2D(0,k,lda), lda, strideA, 
                                            V, shiftV+offsetV, ldv, strideV, &one,
                                            work, 0, ldw, strideW, batch_count, workArr);   
        }
    }

    // compute: 
    // trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // (A1 * V1 + A2 * V2) * trans(T)    
    for (int b=0;b<batch_count;++b) {
        Fp = load_ptr_batch<T>(F,b,shiftF,strideF);
        rocblas_trmm(handle,side,uploT,transt,rocblas_diagonal_non_unit,ldw,order,&one,Fp,ldf,(work + b*strideW),ldw);
    }

    // compute:
    // A2 - V2 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // A2 - (A1 * V1 + A2 * V2) * trans(T) * V2'    
    if (transp == rocblas_operation_none)
        transp = rocblas_operation_conjugate_transpose;
    else
        transp = rocblas_operation_none;

    if (trap) {
        if (leftside) { 
            rocblasCall_gemm<BATCHED,STRIDED,T>(handle, transp, rocblas_operation_none, 
                                            m-k, order, ldw, &minone,
                                            V, shiftV+offsetV, ldv, strideV, 
                                            work, 0, ldw, strideW, &one,   
                                            A, shiftA+idx2D(k,0,lda), lda, strideA, batch_count, workArr); 
        } else {
            rocblasCall_gemm<BATCHED,STRIDED,T>(handle, rocblas_operation_none, transp,
                                            ldw, n-k, order, &minone,
                                            work, 0, ldw, strideW,    
                                            V, shiftV+offsetV, ldv, strideV, &one,
                                            A, shiftA+idx2D(0,k,lda), lda, strideA, batch_count, workArr); 
        }
    }
        
    // compute:
    // V1 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // (A1 * V1 + A2 * V2) * trans(T) * V1'    
    for (int b=0;b<batch_count;++b) {
        Vp = load_ptr_batch<T>(VV,b,shiftV,strideV);
        rocblas_trmm(handle,side,uploV,transp,rocblas_diagonal_unit,ldw,order,&one,Vp,ldv,(work + b*strideW),ldw);
    }
    
    // compute:
    // A1 - V1 * trans(T) * (V1' * A1 + V2' * A2)
    //              or
    // A1 - (A1 * V1 + A2 * V2) * trans(T) * V1'
    hipLaunchKernelGGL(addmatA1,dim3(blocksx,blocksy,batch_count),dim3(32,32),0,stream,ldw,order,A,shiftA,lda,strideA,work);
   
    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif
