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

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
__global__ void set_taubeta(T *tau, const rocblas_stride strideP, T *norms, U alpha, const rocblas_int shifta, const rocblas_stride stride)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(alpha,b,shifta,stride);
    T* t = tau + b*strideP;

    if(norms[b] > 0) {
        T n = sqrt(norms[b] + a[0]*a[0]);
        n = a[0] > 0 ? -n : n;

        //scaling factor:
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

template <typename T, typename U, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
__global__ void set_taubeta(T *tau, const rocblas_stride strideP, T *norms, U alpha, const rocblas_int shifta, const rocblas_stride stride)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(alpha,b,shifta,stride);
    T* t = tau + b*strideP;

    auto m = a[0].y*a[0].y;

    if(norms[b].x > 0 || m > 0) {
        T n;
        m += a[0].x*a[0].x;
        n.x = sqrt(norms[b].x + m);

        //n = -sgn(alpha) * norm(x)
        n = m > 0 ?
            -copysign(n.x, a[0].x != 0 ? a[0].x : a[0].y) :
            0;

        //scaling factor:
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

template <typename T>
void rocsolver_larfg_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2)
{
    // size of norms
    *size_1 = sizeof(T)*batch_count;

    // size of workspace
    *size_2 = (n-2)/ROCBLAS_DOT_NB + 2;
    *size_2 *= sizeof(T)*batch_count;
}

template <typename T, typename U, bool COMPLEX = !std::is_floating_point<T>::value>
rocblas_status rocsolver_larfg_template(rocblas_handle handle, const rocblas_int n, U alpha, const rocblas_int shifta, 
                                        U x, const rocblas_int shiftx, const rocblas_int incx, const rocblas_stride stridex,
                                        T *tau, const rocblas_stride strideP, const rocblas_int batch_count, T* norms, T* work)
{
    // quick return
    if (n == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    
    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device);  
  
    //if n==1 return tau=0
    dim3 gridReset(1, batch_count, 1);
    dim3 threads(1, 1, 1); 
    if (n == 1 && !COMPLEX) {
        hipLaunchKernelGGL(reset_batch_info,gridReset,threads,0,stream,tau,strideP,1,0);
        return rocblas_status_success;    
    }
    
    //compute squared norm of x
    rocblasCall_dot<COMPLEX,T>(handle, n-1, x, shiftx, incx, stridex,
                               x, shiftx, incx, stridex, batch_count, norms, work);

    //set value of tau and beta and scalling factor for vector x
    //alpha <- beta, norms <- scaling   
    hipLaunchKernelGGL(set_taubeta<T>,dim3(batch_count),dim3(1),0,stream,tau,strideP,norms,alpha,shifta,stridex);
     
    //compute vector v=x*norms
    rocblasCall_scal<T>(handle, n-1, norms, 1, x, shiftx, incx, stridex, batch_count);

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif
