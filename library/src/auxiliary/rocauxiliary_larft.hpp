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
#include "common_device.hpp"

template <typename T, typename U>
__global__ void set_triangular(const rocblas_int k, U V, const rocblas_int shiftV, const rocblas_int ldv, const rocblas_stride strideV, 
                         T* tau, const rocblas_stride strideT, 
                         T* F, const rocblas_int ldf, const rocblas_stride strideF, const rocsolver_storev storev)
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
__global__ void set_tau(const rocblas_int k, T* tau, const rocblas_stride strideT)
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
rocblas_status rocsolver_larft_template(rocblas_handle handle, const rocsolver_direct direct, 
                                   const rocsolver_storev storev, const rocblas_int n,
                                   const rocblas_int k, U V, const rocblas_int shiftV, const rocblas_int ldv, 
                                   const rocblas_stride strideV, T* tau, const rocblas_stride strideT, T* F, 
                                   const rocblas_int ldf, const rocblas_stride strideF, const rocblas_int batch_count)
{
    // quick return
    if (!n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    //constants to use when calling rocablas functions
    T one = 1;                //constant 1 in host
    T zero = 0;               //constant 0 in host
    T* oneInt;                //constant 1 in device
    T* zeroInt;               //constant 0 in device
    hipMalloc(&oneInt, sizeof(T));
    hipMemcpy(oneInt, &one, sizeof(T), hipMemcpyHostToDevice);
    hipMalloc(&zeroInt, sizeof(T));
    hipMemcpy(zeroInt, &zero, sizeof(T), hipMemcpyHostToDevice);

    // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
    //memory in GPU (workspace)
    T *work;
    rocblas_stride stridew = k;
    hipMalloc(&work, sizeof(T)*stridew*batch_count);

    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_fill uplo;
    rocblas_operation trans;

    // BACKWARD DIRECTION TO BE IMPLEMENTED...
    if (direct == rocsolver_backward_direction)
        return rocblas_status_not_implemented;
    // else

    uplo = rocblas_fill_upper;

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
 
    for (rocblas_int i = 1; i < k; ++i) { 
        //compute the matrix vector product, using the householder vectors
        if (storev == rocsolver_column_wise) {
            trans = rocblas_operation_transpose;
            rocblasCall_gemv<T>(handle, trans, n-1-i, i, tau + i, strideT, 
                            V, shiftV + idx2D(i+1,0,ldv), ldv, strideV,
                            V, shiftV + idx2D(i+1,i,ldv), 1, strideV, oneInt, 0,
                            F, idx2D(0,i,ldf), 1, strideF, batch_count);
        } else {
            trans = rocblas_operation_none;
            rocblasCall_gemv<T>(handle, trans, i, n-1-i, tau + i, strideT, 
                            V, shiftV + idx2D(0,i+1,ldv), ldv, strideV,
                            V, shiftV + idx2D(i,i+1,ldv), ldv, strideV, oneInt, 0,
                            F, idx2D(0,i,ldf), 1, strideF, batch_count);
        }

        //multiply by the previous triangular factor
        trans = rocblas_operation_none; 
        rocblasCall_trmv<T>(handle, uplo, trans, diag, i, F, 0, ldf, strideF, 
                        F, idx2D(0,i,ldf), 1, strideF,
                        work, stridew, batch_count);
    }

    //restore tau
    hipLaunchKernelGGL(set_tau,dim3(batch_count,blocks),dim3(32,1),0,stream,k,tau,strideT);

    hipFree(oneInt);
    hipFree(zeroInt);
    hipFree(work);   
 
    return rocblas_status_success;
}

#endif
