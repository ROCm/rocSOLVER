/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGBR_UNGBR_HPP
#define ROCLAPACK_ORGBR_UNGBR_HPP

#include "rocauxiliary_orglq_unglq.hpp"
#include "rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orgbr_ungbr_getMemorySize(const rocblas_storev storev,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_Abyx_tmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_Abyx_tmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    if(storev == rocblas_column_wise)
    {
        // requirements for calling orgqr/ungqr
        if(m >= k)
        {
            rocsolver_orgqr_ungqr_getMemorySize<T, BATCHED>(m, n, k, batch_count, size_scalars,
                                                            size_work, size_Abyx_tmptr, size_trfact,
                                                            size_workArr);
        }
        else
        {
            size_t s1 = sizeof(T) * batch_count * (m - 1) * m / 2;
            size_t s2;
            rocsolver_orgqr_ungqr_getMemorySize<T, BATCHED>(m - 1, m - 1, m - 1, batch_count,
                                                            size_scalars, &s2, size_Abyx_tmptr,
                                                            size_trfact, size_workArr);
            *size_work = max(s1, s2);
        }
    }

    else
    {
        // requirements for calling orglq/unglq
        if(n > k)
        {
            rocsolver_orglq_unglq_getMemorySize<T, BATCHED>(m, n, k, batch_count, size_scalars,
                                                            size_work, size_Abyx_tmptr, size_trfact,
                                                            size_workArr);
        }
        else
        {
            size_t s1 = sizeof(T) * batch_count * (n - 1) * n / 2;
            size_t s2;
            rocsolver_orglq_unglq_getMemorySize<T, BATCHED>(n - 1, n - 1, n - 1, batch_count,
                                                            size_scalars, &s2, size_Abyx_tmptr,
                                                            size_trfact, size_workArr);
            *size_work = max(s1, s2);
        }
    }
}

template <typename T, typename U>
rocblas_status rocsolver_orgbr_argCheck(const rocblas_storev storev,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int lda,
                                        T A,
                                        U ipiv)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    if(m < 0 || n < 0 || k < 0 || lda < m)
        return rocblas_status_invalid_size;
    if(!row && (n > m || n < min(m, k)))
        return rocblas_status_invalid_size;
    if(row && (m > n || m < min(n, k)))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((m * n && !A) || (row && min(n, k) > 0 && !ipiv) || (!row && min(m, k) > 0 && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgbr_ungbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* Abyx_tmptr,
                                              T* trfact,
                                              T** workArr)
{
    // quick return
    if(!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if column-wise, compute orthonormal columns of matrix Q in the
    // bi-diagonalization of a m-by-k matrix A (given by gebrd)
    if(storev == rocblas_column_wise)
    {
        if(m >= k)
        {
            rocsolver_orgqr_ungqr_template<BATCHED, STRIDED, T>(
                handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work,
                Abyx_tmptr, trfact, workArr);
        }
        else
        {
            // shift the householder vectors provided by gebrd as they come below the
            // first subdiagonal
            rocblas_stride strideW = rocblas_stride(m - 1) * m / 2; // number of elements to copy
            rocblas_int ldw = m - 1;
            rocblas_int blocks = (m - 2) / BS + 1;

            // copy
            hipLaunchKernelGGL(copyshift_right<T>, dim3(blocks, blocks, batch_count), dim3(BS, BS), 0,
                               stream, true, m - 1, A, shiftA, lda, strideA, work, 0, ldw, strideW);

            // shift
            hipLaunchKernelGGL(copyshift_right<T>, dim3(blocks, blocks, batch_count), dim3(BS, BS), 0,
                               stream, false, m - 1, A, shiftA, lda, strideA, work, 0, ldw, strideW);

            // result
            rocsolver_orgqr_ungqr_template<BATCHED, STRIDED, T>(
                handle, m - 1, m - 1, m - 1, A, shiftA + idx2D(1, 1, lda), lda, strideA, ipiv,
                strideP, batch_count, scalars, work, Abyx_tmptr, trfact, workArr);
        }
    }

    // if row-wise, compute orthonormal rowss of matrix P' in the
    // bi-diagonalization of a k-by-n matrix A (given by gebrd)
    else
    {
        if(n > k)
        {
            rocsolver_orglq_unglq_template<BATCHED, STRIDED, T>(
                handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work,
                Abyx_tmptr, trfact, workArr);
        }
        else
        {
            // shift the householder vectors provided by gebrd as they come above the
            // first superdiagonal
            rocblas_stride strideW = rocblas_stride(n - 1) * n / 2; // number of elements to copy
            rocblas_int ldw = n - 1;
            rocblas_int blocks = (n - 2) / BS + 1;

            // copy
            hipLaunchKernelGGL(copyshift_down<T>, dim3(blocks, blocks, batch_count), dim3(BS, BS), 0,
                               stream, true, n - 1, A, shiftA, lda, strideA, work, 0, ldw, strideW);

            // shift
            hipLaunchKernelGGL(copyshift_down<T>, dim3(blocks, blocks, batch_count), dim3(BS, BS), 0,
                               stream, false, n - 1, A, shiftA, lda, strideA, work, 0, ldw, strideW);

            // result
            rocsolver_orglq_unglq_template<BATCHED, STRIDED, T>(
                handle, n - 1, n - 1, n - 1, A, shiftA + idx2D(1, 1, lda), lda, strideA, ipiv,
                strideP, batch_count, scalars, work, Abyx_tmptr, trfact, workArr);
        }
    }

    return rocblas_status_success;
}

#endif
