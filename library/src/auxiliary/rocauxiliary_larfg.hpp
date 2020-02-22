/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.8.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2017
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARFG_HPP
#define ROCLAPACK_LARFG_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"

template <typename T, typename U>
__global__ void set_taubeta(T *tau, const rocblas_int strideP, T *norms, U alpha, const rocblas_int shifta, const rocblas_int stride)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(alpha,shifta,b,stride);
    T* t = tau + b*strideP;

    if(norms[b] > 0) {
        T n = T(sqrt(norms[b]*norms[b] + a[0]*a[0]));
        n = a[0] > 0 ? -n : n;

        //scalling factor:
        norms[b] = 1.0 / (a[0] - n);
        //tau:
        t[0] = (n - a[0]) / n;
        //beta:
        a[0] = n;
    } else {
        norms[b] = 1;
        t[0] = 0;
    }
}


template <typename T, typename U>
rocblas_status rocsolver_larfg_template(rocblas_handle handle, const rocblas_int n, U alpha, const rocblas_int shifta, 
                                        U x, const rocblas_int shiftx, const rocblas_int incx, const rocblas_int stridex,
                                        T *tau, const rocblas_int strideP, const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || !batch_count)
        return rocblas_status_success;

    //if n==1 return tau=0
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    dim3 gridReset(1, batch_count, 1);
    dim3 threads(1, 1, 1); 
    if (n == 1) {
        hipLaunchKernelGGL(reset_batch_info,gridReset,threads,0,stream,tau,strideP,1,0);
        return rocblas_status_success;    
    } 

    T *xp;

    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* xx[batch_count];
        hipMemcpy(xx, x, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* xx = x;
    #endif

    //memory in GPU (workspace)
    T *norms;
    hipMalloc(&norms, sizeof(T)*batch_count);    

    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****
    
    //compute norm of x
    for (int b=0;b<batch_count;++b) {
        xp = load_ptr_batch<T>(xx,shiftx,b,stridex);
        rocblas_nrm2(handle, n - 1, xp, incx, (norms + b));
    }

    //set value of tau and beta and scalling factor for vector x
    //alpha <- beta
    //norms <- scalling   
    hipLaunchKernelGGL(set_taubeta<T>,dim3(batch_count),dim3(1),0,stream,tau,strideP,norms,alpha,shifta,stridex);
     
    //compute vector v=x*norms
    for (int b=0;b<batch_count;++b) {
        xp = load_ptr_batch<T>(xx,shiftx,b,stridex);
        rocblas_scal(handle, n - 1, (norms + b), xp, incx);
    }

    hipFree(norms);

    return rocblas_status_success;
}

#endif
