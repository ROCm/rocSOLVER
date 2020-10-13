/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARF_HPP
#define ROCLAPACK_LARF_HPP

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_larf_getMemorySize(const rocblas_side side,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_Abyx,
                                  size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || m == 0 || !batch_count)
    {
        *size_scalars = 0;
        *size_Abyx = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of temporary result in Householder matrix generation
    if(side == rocblas_side_left)
        *size_Abyx = n;
    else if(side == rocblas_side_right)
        *size_Abyx = m;
    else
        *size_Abyx = max(m, n);
    *size_Abyx *= sizeof(T) * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_larf_argCheck(const rocblas_side side,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int lda,
                                       const rocblas_int incx,
                                       T x,
                                       T A,
                                       U alpha)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || !incx)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((m * n && !A) || (left && m && (!alpha || !x)) || (!left && n && (!alpha || !x)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_larf_template(rocblas_handle handle,
                                       const rocblas_side side,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       U x,
                                       const rocblas_int shiftx,
                                       const rocblas_int incx,
                                       const rocblas_stride stridex,
                                       const T* alpha,
                                       const rocblas_stride stridep,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride stridea,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       T* Abyx,
                                       T** workArr)
{
    // quick return
    if(n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // determine side and order of H
    bool leftside = (side == rocblas_side_left);
    rocblas_int order = m;
    rocblas_operation trans = rocblas_operation_none;
    if(leftside)
    {
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
        order = n;
    }

    // **** FOR NOW, IT DOES NOT DETERMINE "NON-ZERO" DIMENSIONS
    //      OF A AND X, AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****

    // compute the matrix vector product  (W=-A'*X or W=-A*X)
    rocblasCall_gemv<T>(handle, trans, m, n, cast2constType<T>(scalars), 0, A, shiftA, lda, stridea,
                        x, shiftx, incx, stridex, cast2constType<T>(scalars + 1), 0, Abyx, 0, 1,
                        order, batch_count, workArr);

    // compute the rank-1 update  (A + tau*X*W'  or A + tau*W*X')
    if(leftside)
    {
        rocblasCall_ger<COMPLEX, T>(handle, m, n, alpha, stridep, x, shiftx, incx, stridex, Abyx, 0,
                                    1, order, A, shiftA, lda, stridea, batch_count, workArr);
    }
    else
    {
        rocblasCall_ger<COMPLEX, T>(handle, m, n, alpha, stridep, Abyx, 0, 1, order, x, shiftx,
                                    incx, stridex, A, shiftA, lda, stridea, batch_count, workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif
