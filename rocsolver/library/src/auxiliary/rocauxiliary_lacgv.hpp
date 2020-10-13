/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LACGV_HPP
#define ROCLAPACK_LACGV_HPP

#include "common_device.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
__global__ void conj_in_place(const rocblas_int m,
                              const rocblas_int n,
                              U A,
                              const rocblas_int shifta,
                              const rocblas_int lda,
                              const rocblas_stride stridea)
{
    // do nothing
}

template <typename T, typename U, std::enable_if_t<is_complex<T>, int> = 0>
__global__ void conj_in_place(const rocblas_int m,
                              const rocblas_int n,
                              U A,
                              const rocblas_int shifta,
                              const rocblas_int lda,
                              const rocblas_stride stridea)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int b = hipBlockIdx_z;

    T* Ap = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < m && j < n)
        Ap[i + j * lda] = conj(Ap[i + j * lda]);
}

template <typename T>
rocblas_status rocsolver_lacgv_argCheck(const rocblas_int n, const rocblas_int incx, T x)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || !incx)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if(n && !x)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_lacgv_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U x,
                                        const rocblas_int shiftx,
                                        const rocblas_int incx,
                                        const rocblas_stride stridex,
                                        const rocblas_int batch_count)
{
    // quick return
    if(n == 0 || !batch_count || !COMPLEX)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // handle negative increments
    rocblas_int offset = incx < 0 ? shiftx - (n - 1) * incx : shiftx;

    // conjugate x
    rocblas_int blocks = (n - 1) / 64 + 1;
    hipLaunchKernelGGL(conj_in_place<T>, dim3(1, blocks, batch_count), dim3(1, 64, 1), 0, stream, 1,
                       n, x, offset, incx, stridex);

    return rocblas_status_success;
}

#endif
