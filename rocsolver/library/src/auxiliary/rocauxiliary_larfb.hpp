/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2013
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_LARFB_HPP
#define ROCLAPACK_LARFB_HPP

#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U>
__global__ void copymatA1(const rocblas_int ldw,
                          const rocblas_int order,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          T* tmptr)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Wp[i + j * ldw] = Ap[i + j * lda];
    }
}

template <typename T, typename U>
__global__ void addmatA1(const rocblas_int ldw,
                         const rocblas_int order,
                         U A,
                         const rocblas_int shiftA,
                         const rocblas_int lda,
                         const rocblas_stride strideA,
                         T* tmptr)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Ap[i + j * lda] -= Wp[i + j * ldw];
    }
}

template <typename T, bool BATCHED>
void rocsolver_larfb_getMemorySize(const rocblas_side side,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_work,
                                   size_t* size_tmptr,
                                   size_t* size_workArr)
{
    // if quick return, no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_work = 0;
        *size_tmptr = 0;
        *size_workArr = 0;
        return;
    }

    // size of temporary array for computations with
    // triangular part of V
    if(side == rocblas_side_left)
        *size_tmptr = n;
    else
        *size_tmptr = m;
    *size_tmptr *= sizeof(T) * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // size of workspace
    *size_work = 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB * sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_larfb_argCheck(const rocblas_side side,
                                        const rocblas_operation trans,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int ldv,
                                        const rocblas_int ldf,
                                        const rocblas_int lda,
                                        T V,
                                        T A,
                                        U F)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    if(m < 0 || n < 0 || k < 1 || lda < m || ldf < k)
        return rocblas_status_invalid_size;
    if(row && ldv < k)
        return rocblas_status_invalid_size;
    if((!row && left && ldv < m) || (!row && !left && ldv < n))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((left && m && !V) || (!left && n && !V) || (m * n && !A) || !F)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_larfb_template(rocblas_handle handle,
                                        const rocblas_side side,
                                        const rocblas_operation trans,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        T* F,
                                        const rocblas_int shiftF,
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const rocblas_int batch_count,
                                        T* work,
                                        T* tmptr,
                                        T** workArr)
{
    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    T *Vp, *Fp;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T minone = -1;
    T one = 1;

    // determine the side, size of workspace
    // and whether V is trapezoidal
    bool trap;
    bool colwise = (storev == rocblas_column_wise);
    bool forward = (direct == rocblas_forward_direction);
    bool leftside = (side == rocblas_side_left);
    rocblas_operation transt
        = (leftside && trans == rocblas_operation_transpose ? rocblas_operation_conjugate_transpose
                                                            : trans);
    rocblas_operation transp;
    rocblas_fill uploV, uploT;
    rocblas_int order, ldw;
    rocblas_int shift1, shift2;
    size_t offsetA1, offsetA2;
    size_t offsetV1, offsetV2;

    if(leftside)
    {
        order = n;
        ldw = k;
        trap = (m > k);

        if(forward)
        {
            offsetA1 = shiftA;
            offsetA2 = shiftA + idx2D(k, 0, lda);
        }
        else
        {
            offsetA1 = shiftA + idx2D(m - k, 0, lda);
            offsetA2 = shiftA;
        }
    }
    else
    {
        order = k;
        ldw = m;
        trap = (n > k);

        if(forward)
        {
            offsetA1 = shiftA;
            offsetA2 = shiftA + idx2D(0, k, lda);
        }
        else
        {
            offsetA1 = shiftA + idx2D(0, n - k, lda);
            offsetA2 = shiftA;
        }
    }

    if(colwise)
    {
        if(leftside)
            transp = rocblas_operation_conjugate_transpose;
        else
            transp = rocblas_operation_none;

        if(forward)
        {
            uploV = rocblas_fill_lower;
            offsetV1 = shiftV;
            offsetV2 = shiftV + idx2D(k, 0, ldv);
        }
        else
        {
            uploV = rocblas_fill_upper;
            offsetV1 = shiftV + idx2D((leftside ? m - k : n - k), 0, ldv);
            offsetV2 = shiftV;
        }
    }
    else
    {
        if(leftside)
            transp = rocblas_operation_none;
        else
            transp = rocblas_operation_conjugate_transpose;

        if(forward)
        {
            uploV = rocblas_fill_upper;
            offsetV1 = shiftV;
            offsetV2 = shiftV + idx2D(0, k, ldv);
        }
        else
        {
            uploV = rocblas_fill_lower;
            offsetV1 = shiftV + idx2D(0, (leftside ? m - k : n - k), ldv);
            offsetV2 = shiftV;
        }
    }
    rocblas_stride strideW = rocblas_stride(ldw) * order;
    uploT = (forward ? rocblas_fill_upper : rocblas_fill_lower);

    // copy A1 to tmptr
    rocblas_int blocksx = (order - 1) / 32 + 1;
    rocblas_int blocksy = (ldw - 1) / 32 + 1;
    hipLaunchKernelGGL(copymatA1, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0, stream, ldw,
                       order, A, offsetA1, lda, strideA, tmptr);

    // compute: V1' * A1
    //   or    A1 * V1
    rocblasCall_trmm<BATCHED, STRIDED, T>(handle, side, uploV, transp, rocblas_diagonal_unit, ldw,
                                          order, &one, V, offsetV1, ldv, strideV, tmptr, 0, ldw,
                                          strideW, batch_count, work, workArr);

    // compute: V1' * A1 + V2' * A2
    //    or    A1 * V1 + A2 * V2
    if(trap)
    {
        if(leftside)
            rocblasCall_gemm<BATCHED, STRIDED, T>(handle, transp, rocblas_operation_none, ldw,
                                                  order, m - k, &one, V, offsetV2, ldv, strideV, A,
                                                  offsetA2, lda, strideA, &one, tmptr, 0, ldw,
                                                  strideW, batch_count, workArr);
        else
            rocblasCall_gemm<BATCHED, STRIDED, T>(handle, rocblas_operation_none, transp, ldw,
                                                  order, n - k, &one, A, offsetA2, lda, strideA, V,
                                                  offsetV2, ldv, strideV, &one, tmptr, 0, ldw,
                                                  strideW, batch_count, workArr);
    }

    // compute: trans(T) * (V1' * A1 + V2' * A2)
    //    or    (A1 * V1 + A2 * V2) * trans(T)
    rocblasCall_trmm<false, STRIDED, T>(handle, side, uploT, transt, rocblas_diagonal_non_unit, ldw,
                                        order, &one, F, shiftF, ldf, strideF, tmptr, 0, ldw,
                                        strideW, batch_count, work, workArr);

    // compute: A2 - V2 * trans(T) * (V1' * A1 + V2' * A2)
    //    or    A2 - (A1 * V1 + A2 * V2) * trans(T) * V2'
    if(transp == rocblas_operation_none)
        transp = rocblas_operation_conjugate_transpose;
    else
        transp = rocblas_operation_none;

    if(trap)
    {
        if(leftside)
            rocblasCall_gemm<BATCHED, STRIDED, T>(handle, transp, rocblas_operation_none, m - k,
                                                  order, ldw, &minone, V, offsetV2, ldv, strideV,
                                                  tmptr, 0, ldw, strideW, &one, A, offsetA2, lda,
                                                  strideA, batch_count, workArr);
        else
            rocblasCall_gemm<BATCHED, STRIDED, T>(handle, rocblas_operation_none, transp, ldw,
                                                  n - k, order, &minone, tmptr, 0, ldw, strideW, V,
                                                  offsetV2, ldv, strideV, &one, A, offsetA2, lda,
                                                  strideA, batch_count, workArr);
    }

    // compute: V1 * trans(T) * (V1' * A1 + V2' * A2)
    //    or    (A1 * V1 + A2 * V2) * trans(T) * V1'
    rocblasCall_trmm<BATCHED, STRIDED, T>(handle, side, uploV, transp, rocblas_diagonal_unit, ldw,
                                          order, &one, V, offsetV1, ldv, strideV, tmptr, 0, ldw,
                                          strideW, batch_count, work, workArr);

    // compute: A1 - V1 * trans(T) * (V1' * A1 + V2' * A2)
    //    or    A1 - (A1 * V1 + A2 * V2) * trans(T) * V1'
    hipLaunchKernelGGL(addmatA1, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0, stream, ldw,
                       order, A, offsetA1, lda, strideA, tmptr);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif
