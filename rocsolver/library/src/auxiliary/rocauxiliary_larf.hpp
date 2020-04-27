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

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"

template <typename T, bool BATCHED>
void rocsolver_larf_getMemorySize(const rocblas_side side, const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // size of scalars (constants)
    *size_1 = sizeof(T)*3;        

    // size of workspace
    if (side == rocblas_side_left)
        *size_2 = n;
    else
        *size_2 = m;
    *size_2 *= sizeof(T)*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = sizeof(T*)*batch_count;
    else
        *size_3 = 0;
}

template <typename T>
void rocsolver_larf_getMemorySize(const rocblas_side side, const rocblas_int m, const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size)
{
    // size of workspace
    if (side == rocblas_side_left)
        *size = n;
    else
        *size = m;
    *size *= sizeof(T)*batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_larf_template(rocblas_handle handle, const rocblas_side side, const rocblas_int m,
                                        const rocblas_int n, U x, const rocblas_int shiftx, const rocblas_int incx, 
                                        const rocblas_stride stridex, const T* alpha, const rocblas_stride stridep, U A, const rocblas_int shiftA, 
                                        const rocblas_int lda, const rocblas_stride stridea, const rocblas_int batch_count, T* scalars, T* work, T** workArr)
{
    // quick return
    if (n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_device);  

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
 
    //compute the matrix vector product  (W=tau*A'*X or W=tau*A*X)
    rocblasCall_gemv<T>(handle, trans, m, n, alpha, stridep, A, shiftA, lda, stridea, 
                        x, shiftx, incx, stridex, cast2constType<T>(scalars+1), 0, 
                        work, 0, 1, order, batch_count, workArr);

    //compute the rank-1 update  (A - V*W'  or A - W*V')
    if (leftside) {
        rocblasCall_ger<false,T>(handle, m, n, scalars, 0, x, shiftx, incx, stridex,
                             work, 0, 1, order, A, shiftA, lda, stridea, batch_count, workArr);
    } else {
        rocblasCall_ger<false,T>(handle, m, n, scalars, 0, work, 0 ,1, order, 
                             x, shiftx, incx, stridex, A, shiftA, lda, stridea, batch_count, workArr);
    }

    rocblas_set_pointer_mode(handle,old_mode);  
    return rocblas_status_success;
}

#endif
