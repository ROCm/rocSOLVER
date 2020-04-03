/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARF_HPP
#define ROCLAPACK_LARF_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, typename U>
rocblas_status rocsolver_larf_template(rocblas_handle handle, const rocblas_side side, const rocblas_int m,
                                        const rocblas_int n, U x, const rocblas_int shiftx, const rocblas_int incx, 
                                        const rocblas_stride stridex, const T* alpha, const rocblas_stride stridep, U A, const rocblas_int shiftA, 
                                        const rocblas_int lda, const rocblas_stride stridea, const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    //constants to use when calling rocablas functions
    T minone = -1;                //constant -1 in host
    T* minoneInt;                 //constant -1 in device
    hipMalloc(&minoneInt, sizeof(T));
    hipMemcpy(minoneInt, &minone, sizeof(T), hipMemcpyHostToDevice);
    T zero = 0;                 //constant 0 in host
    T* zeroInt;                 //constant 0 in device
    hipMalloc(&zeroInt, sizeof(T));
    hipMemcpy(zeroInt, &zero, sizeof(T), hipMemcpyHostToDevice);
    
    //determine side and order of H
    bool leftside = (side == rocblas_side_left);
    rocblas_int order = m;
    rocblas_operation trans = rocblas_operation_none;
    if (leftside) {
        trans = rocblas_operation_transpose;
        order = n;
    }
    
    // (TODO) THIS SHOULD BE DONE WITH THE HANDLE MEMORY ALLOCATOR
    //memory in GPU (workspace)
    T *work;
    hipMalloc(&work, sizeof(T)*order*batch_count);

    // **** FOR NOW, IT DOES NOT DETERMINE "NON-ZERO" DIMENSIONS
    //      OF A AND X, AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****
 
    //compute the matrix vector product  (W=tau*A'*X or W=tau*A*X)
    rocblasCall_gemv<T>(handle, trans, m, n, alpha, stridep, A, shiftA, lda, stridea, 
                    x, shiftx, incx, stridex, cast2constType<T>(zeroInt), 0, 
                    work, 0, 1, order, batch_count);

    //compute the rank-1 update  (A - V*W'  or A - W*V')
    if (leftside) {
        rocblasCall_ger<false,T>(handle, m, n, minoneInt, 0, x, shiftx, incx, stridex,
                             work, 0, 1, order, A, shiftA, lda, stridea, batch_count);
    } else {
        rocblasCall_ger<false,T>(handle, m, n, minoneInt, 0, work, 0 ,1, order, 
                             x, shiftx, incx, stridex, A, shiftA, lda, stridea, batch_count);
    }

    hipFree(minoneInt);
    hipFree(zeroInt);
    hipFree(work);

    return rocblas_status_success;
}

#endif
