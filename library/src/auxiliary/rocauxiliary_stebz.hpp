/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.10.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

/*template <typename T>
void rocsolver_stebz_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_stack = 0;
        return;
    }

    // size of stack (for lasrt)
    *size_stack = sizeof(rocblas_int) * (2 * 32) * batch_count;
}*/

template <typename T>
rocblas_status rocsolver_stebz_argCheck(rocblas_handle handle,
                                        const rocblas_eval_range range,
                                        const rocblas_eval_order order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        T* D,
                                        T* E,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        rocblas_int* IB,
                                        rocblas_int* IS,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(range != rocblas_range_all && range != rocblas_range_value && range != rocblas_range_index)
        return rocblas_status_invalid_value;
    if(order != rocblas_order_blocks && order != rocblas_order_entire)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(range == rocblas_range_value && vlow >= vup)
        return rocblas_status_invalid_size;
    if(range == rocblas_range_index && (ilow < 1 || iup < 0))
        return rocblas_status_invalid_size;
    if(range == rocblas_range_index && (iup > n || (n > 0 && ilow > iup)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !W || !IB || !IS)) || (n > 1 && !E) || !info || !nev || !nsplit)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_stebz_template(rocblas_handle handle,
                                        const rocblas_eval_range range,
                                        const rocblas_eval_order order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        const T abstol,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* IB,
                                        const rocblas_stride strideIB,
                                        rocblas_int* IS,
                                        const rocblas_stride strideIS,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    //    ROCSOLVER_ENTER("stebz", "range:", range, "order:", order, "n:", n, "vlow:", vlow, "vup:", vup,
    //                    "ilow:", ilow, "iup:", iup, "abstol:", abstol, "shiftD:", shiftD,
    //                    "shiftE:", shiftE, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = nev = nsplit = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nsplit, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    T ulp = get_epsilon<T>();
    T ssfmin = get_safemin<T>();
    T rtol = 2 * ulp;

    //    ROCSOLVER_LAUNCH_KERNEL(stebz_kernel<T>, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
    //                            strideD, E + shiftE, strideE, info, stack, 30 * n, eps, ssfmin, ssfmax);

    return rocblas_status_success;
}
