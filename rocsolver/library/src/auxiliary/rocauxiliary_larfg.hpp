/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.8.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2017
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARFG_HPP
#define ROCLAPACK_LARFG_HPP

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
__global__ void set_taubeta(T* tau,
                            const rocblas_stride strideP,
                            T* norms,
                            U alpha,
                            const rocblas_int shifta,
                            const rocblas_stride stride)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(alpha, b, shifta, stride);
    T* t = tau + b * strideP;

    if(norms[b] > 0)
    {
        T n = sqrt(norms[b] + a[0] * a[0]);
        n = a[0] >= 0 ? -n : n;

        // scaling factor:
        norms[b] = 1.0 / (a[0] - n);

        // tau:
        t[0] = (n - a[0]) / n;

        // beta:
        a[0] = n;
    }
    else
    {
        norms[b] = 1;
        t[0] = 0;
    }
}

template <typename T, typename U, std::enable_if_t<is_complex<T>, int> = 0>
__global__ void set_taubeta(T* tau,
                            const rocblas_stride strideP,
                            T* norms,
                            U alpha,
                            const rocblas_int shifta,
                            const rocblas_stride stride)
{
    using S = decltype(std::real(T{}));
    int b = hipBlockIdx_x;
    S r, rr, ri, ar, ai;

    T* a = load_ptr_batch<T>(alpha, b, shifta, stride);
    T* t = tau + b * strideP;

    ar = a[0].real();
    ai = a[0].imag();
    S m = ai * ai;

    if(norms[b].real() > 0 || m > 0)
    {
        m += ar * ar;
        S n = sqrt(norms[b].real() + m);
        n = ar >= 0 ? -n : n;

        // scaling factor:
        //    norms[b] = 1.0 / (a[0] - n);
        r = (ar - n) * (ar - n) + ai * ai;
        rr = (ar - n) / r;
        ri = -ai / r;
        norms[b] = rocblas_complex_num<S>(rr, ri);

        // tau:
        //t[0] = (n - a[0]) / n;
        rr = (n - ar) / n;
        ri = -ai / n;
        t[0] = rocblas_complex_num<S>(rr, ri);

        // beta:
        a[0] = n;
    }
    else
    {
        norms[b] = 1;
        t[0] = 0;
    }
}

template <typename T>
void rocsolver_larfg_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work,
                                   size_t* size_norms)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_norms = 0;
        *size_work = 0;
        return;
    }

    // size of space to store norms
    *size_norms = sizeof(T) * batch_count;

    // size of re-usable workspace
    *size_work = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    *size_work *= sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status
    rocsolver_larfg_argCheck(const rocblas_int n, const rocblas_int incx, T alpha, T x, U tau)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || incx < 1)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n > 1 && !x) || (n && !alpha) || (n && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_larfg_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U alpha,
                                        const rocblas_int shifta,
                                        U x,
                                        const rocblas_int shiftx,
                                        const rocblas_int incx,
                                        const rocblas_stride stridex,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* work,
                                        T* norms)
{
    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if n==1 return tau=0
    dim3 gridReset(1, batch_count, 1);
    dim3 threads(1, 1, 1);
    if(n == 1 && !COMPLEX)
    {
        hipLaunchKernelGGL(reset_batch_info<T>, gridReset, threads, 0, stream, tau, strideP, 1, 0);
        return rocblas_status_success;
    }

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // compute squared norm of x
    rocblasCall_dot<COMPLEX, T>(handle, n - 1, x, shiftx, incx, stridex, x, shiftx, incx, stridex,
                                batch_count, norms, work);

    // set value of tau and beta and scalling factor for vector x
    // alpha <- beta, norms <- scaling
    hipLaunchKernelGGL(set_taubeta<T>, dim3(batch_count), dim3(1), 0, stream, tau, strideP, norms,
                       alpha, shifta, stridex);

    // compute vector v=x*norms
    rocblasCall_scal<T>(handle, n - 1, norms, 1, x, shiftx, incx, stridex, batch_count);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif
