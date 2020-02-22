/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARFT_HPP
#define ROCLAPACK_LARFT_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "helpers.h"
#include "common_device.hpp"

template <typename T, typename U>
__global__ void set_triangular(const rocsolver_int k, U V, const rocsolver_int shiftV, const rocsolver_int ldv, const rocsolver_int strideV, 
                         T* tau, const rocsolver_int strideT, 
                         T* F, const rocsolver_int ldf, const rocsolver_int strideF, const rocsolver_storev storev)
{
    const auto blocksize = hipBlockDim_x;
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * blocksize + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * blocksize + hipThreadIdx_y;

    if (i < k && j < k) {
        T *Vp, *tp, *Fp;
        tp = tau + b*strideT;
        Vp = load_ptr_batch<T>(V,shiftV,b,strideV);
        Fp = F + b*strideF;

        if (j < i) {
            if (storev == rocsolver_column_wise)
                Fp[j + i*ldf] = -tp[i] * Vp[i + j*ldv];
            else
                Fp[j + i*ldf] = -tp[i] * Vp[j + i*ldv];
        } else if (j == i) {
            Fp[j + i*ldf] = tp[i];
        } else {
            Fp[j + i*ldf] = 0;
        }
    }
}

template <typename T>
__global__ void set_tau(const rocsolver_int k, T* tau, const rocsolver_int strideT)
{
    const auto blocksize = hipBlockDim_x;
    const auto b = hipBlockIdx_x;
    const auto i = hipBlockIdx_y * blocksize + hipThreadIdx_x;
   
    if (i < k) {
        T *tp;
        tp = tau + b*strideT;
        tp[i] = -tp[i];
    }
}
         

template <typename T, typename U>
rocblas_status rocsolver_larft_template(rocsolver_handle handle, const rocsolver_direct direct, 
                                   const rocsolver_storev storev, const rocsolver_int n,
                                   const rocsolver_int k, U V, const rocblas_int shiftV, const rocsolver_int ldv, 
                                   const rocsolver_int strideV, T* tau, const rocsolver_int strideT, T* F, 
                                   const rocsolver_int ldf, const rocsolver_int strideF, const rocsolver_int batch_count)
{
    // quick return
    if (!n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *Vp, *tp, *Fp;

    //constants to use when calling rocablas functions
    T one = 1;                //constant 1 in host
    T zero = 0;               //constant 0 in host
    T* oneInt;                //constant 1 in device
    T* zeroInt;               //constant 0 in device
    hipMalloc(&oneInt, sizeof(T));
    hipMemcpy(oneInt, &one, sizeof(T), hipMemcpyHostToDevice);
    hipMalloc(&zeroInt, sizeof(T));
    hipMemcpy(zeroInt, &zero, sizeof(T), hipMemcpyHostToDevice);
    
    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* VV[batch_count];
        hipMemcpy(VV, V, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* VV = V;
    #endif

    // BACKWARD DIRECTION TO BE IMPLEMENTED...
    if (direct == rocsolver_backward_direction)
        return rocblas_status_not_implemented;

    //Fix diagonal of T, make zero the non used triangular part, 
    //setup tau (changing signs) and account for the non-stored 1's on the householder vectors
    rocblas_int blocks = (k - 1)/32 + 1;
    hipLaunchKernelGGL(set_triangular,dim3(blocks,blocks,batch_count),dim3(32,32),0,stream,
                        k,V,shiftV,ldv,strideV,tau,strideT,F,ldf,strideF,storev);
    hipLaunchKernelGGL(set_tau,dim3(batch_count,blocks),dim3(32,1),0,stream,k,tau,strideT);

    // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS 
    //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****
 
    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****
    
    rocblas_operation trans;  

    
    for (int i = 1; i < k; ++i) { 
        //compute the matrix vector product, using the householder vectors
        for (int b=0;b<batch_count;++b) {
            tp = tau + b*strideT;
            Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
            Fp = F + b*strideF;
            if (storev == rocsolver_column_wise) {
                trans = rocblas_operation_transpose;
                rocblas_gemv(handle, trans, n-1-i, i, (tp + i), (Vp + idx2D(i+1,0,ldv)),
                              ldv, (Vp + idx2D(i+1,i,ldv)), 1, oneInt, (Fp + idx2D(0,i,ldf)), 1);
            } else {
                trans = rocblas_operation_none;
                rocblas_gemv(handle, trans, i, n-1-i, (tp + i), (Vp + idx2D(0,i+1,ldv)),
                              ldv, (Vp + idx2D(i,i+1,ldv)), ldv, oneInt, (Fp + idx2D(0,i,ldf)), 1);
            }
        }

        //multiply by the previous triangular factor
        //THIS SHOULD BE DONE USING TRMV ONCE THIS
        //FUNCTIONALITY IS AVAILABLE IN ROCBLAS
        trans = rocblas_operation_none; 
        for (int b=0;b<batch_count;++b) {
            Vp = load_ptr_batch<T>(VV,shiftV,b,strideV);
            Fp = F + b*strideF;
            rocblas_gemv(handle, trans, i, i, oneInt, Fp, ldf, 
                        (Fp + idx2D(0,i,ldf)), 1, zeroInt, (Fp + idx2D(0,i,ldf)), 1);
        } 
    }

    //restore tau
    hipLaunchKernelGGL(set_tau,dim3(batch_count,blocks),dim3(32,1),0,stream,k,tau,strideT);

    hipFree(oneInt);
    hipFree(zeroInt);
    
    return rocblas_status_success;
}

#endif
