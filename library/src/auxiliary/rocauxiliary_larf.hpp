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
#include "ideal_sizes.hpp"
#include "common_device.hpp"
#include <vector>

template <typename T, typename U>
rocblas_status rocsolver_larf_template(rocsolver_handle handle, const rocsolver_side side, const rocsolver_int m,
                                        const rocsolver_int n, U x, const rocblas_int shiftx, const rocsolver_int incx, 
                                        const rocblas_int stridex, const T* alpha, const rocblas_int stridep, U A, const rocblas_int shiftA, 
                                        const rocsolver_int lda, const rocblas_int stridea, const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *xp, *Ap;

    //constants to use when calling rocablas functions
    T minone = -1;                //constant -1 in host
    T* minoneInt;                 //constant -1 in device
    hipMalloc(&minoneInt, sizeof(T));
    hipMemcpy(minoneInt, &minone, sizeof(T), hipMemcpyHostToDevice);
    T zero = 0;                 //constant 0 in host
    T* zeroInt;                 //constant 0 in device
    hipMalloc(&zeroInt, sizeof(T));
    hipMemcpy(zeroInt, &zero, sizeof(T), hipMemcpyHostToDevice);
    
    #ifdef batched
        // **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
        //      BATCH-BLAS FUNCTIONALITY IS ENABLED. ****
        T* xx[batch_count];
        hipMemcpy(xx, x, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
        T* AA[batch_count];
        hipMemcpy(AA, A, batch_count*sizeof(T*), hipMemcpyDeviceToHost);
    #else
        T* xx = x;
        T* AA = A;
    #endif

    //determine side and order of H
    bool leftside = (side == rocblas_side_left);
    rocblas_int order = m;
    rocblas_operation trans = rocblas_operation_none;
    if (leftside) {
        trans = rocblas_operation_transpose;
        order = n;
    }

    // **** FOR NOW, IT DOES NOT DETERMINE "NON-ZERO" DIMENSIONS
    //      OF A AND X, AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****
 
    //memory in GPU (workspace)
    T *workvec;
    hipMalloc(&workvec, sizeof(T)*order*batch_count);

    
    // **** BATCH IS EXECUTED IN A FOR-LOOP UNTIL BATCH-BLAS
    //      FUNCITONALITY IS ENABLED. ALSO ROCBLAS CALLS SHOULD
    //      BE MADE TO THE CORRESPONDING TEMPLATE_FUNCTIONS ****
    
    //compute the matrix vector product  (W=tau*A'*X or W=tau*A*X)
    for (int b=0;b<batch_count;++b) {
        xp = load_ptr_batch<T>(xx,shiftx,b,stridex);
        Ap = load_ptr_batch<T>(AA,shiftA,b,stridea);
        rocblas_gemv(handle, trans, m, n, (alpha + b*stridep), Ap, lda, xp, incx, zeroInt, (workvec + b*order), 1);
    }

    //compute the rank-1 update  (A - V*W'  or A - W*V')
    if (leftside) {
        for (int b=0;b<batch_count;++b) {
            xp = load_ptr_batch<T>(xx,shiftx,b,stridex);
            Ap = load_ptr_batch<T>(AA,shiftA,b,stridea);
            rocblas_ger<false>(handle, m, n, minoneInt, xp, incx, (workvec + b*order), 1, Ap, lda);
        }
    } else {
        for (int b=0;b<batch_count;++b) {
            xp = load_ptr_batch<T>(xx,shiftx,b,stridex);
            Ap = load_ptr_batch<T>(AA,shiftA,b,stridea);
            rocblas_ger<false>(handle, m, n, minoneInt, (workvec + b*order), 1, xp, incx, Ap, lda);
        }
    }

    hipFree(minoneInt);
    hipFree(zeroInt);
    hipFree(workvec);

    return rocblas_status_success;
}

#endif
