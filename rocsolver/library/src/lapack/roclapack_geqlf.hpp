/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQLF_H
#define ROCLAPACK_GEQLF_H

#include "../auxiliary/rocauxiliary_larfb.hpp"
#include "../auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_geql2.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_geqlf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_trfact = 0;
        *size_diag_tmptr = 0;
        *size_workArr = 0;
        return;
    }

    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        // requirements for a single GEQL2 call
        rocsolver_geql2_getMemorySize<T, BATCHED>(m, n, batch_count, size_scalars, size_work_workArr,
                                                  size_Abyx_norms_trfact, size_diag_tmptr);
        *size_workArr = 0;
    }
    else
    {
        size_t w1, w2, w3, unused, s1, s2;
        rocblas_int jb = GEQxF_GEQx2_BLOCKSIZE;

        // size to store the temporary triangular factor
        *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

        // requirements for calling GEQL2 with sub blocks
        rocsolver_geql2_getMemorySize<T, BATCHED>(m, jb, batch_count, size_scalars, &w1, &s2, &s1);
        *size_Abyx_norms_trfact = max(s2, *size_Abyx_norms_trfact);

        // requirements for calling LARFT
        rocsolver_larft_getMemorySize<T, BATCHED>(m, jb, batch_count, &unused, &w2, size_workArr);

        // requirements for calling LARFB
        rocsolver_larfb_getMemorySize<T, BATCHED>(rocblas_side_left, m, n - jb, jb, batch_count,
                                                  &w3, &s2, &unused);

        *size_work_workArr = max(w1, max(w2, w3));
        *size_diag_tmptr = max(s1, s2);

        // size of workArr is double to accomodate
        // LARFB's TRMM calls in the batched case
        if(BATCHED)
            *size_workArr *= 2;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geqlf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_trfact,
                                        T* diag_tmptr,
                                        T** workArr)
{
    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
        return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                           batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                           diag_tmptr);

    rocblas_int k = min(m, n); // total number of pivots
    rocblas_int nb = GEQxF_GEQx2_BLOCKSIZE;
    rocblas_int ki = ((k - GEQxF_GEQx2_SWITCHSIZE - 1) / nb) * nb;
    rocblas_int kk = min(k, ki + nb);
    rocblas_int jb, j = k - kk + ki;
    rocblas_int mu = m, nu = n;

    rocblas_int ldw = GEQxF_GEQx2_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    while(j >= k - kk)
    {
        // Factor diagonal and subdiagonal blocks
        jb = min(k - j, nb); // number of columns in the block
        rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                    lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                    work_workArr, Abyx_norms_trfact, diag_tmptr);

        // apply transformation to the rest of the matrix
        if(n - k + j > 0)
        {
            // compute block reflector
            rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                                        m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                        lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                                        strideW, batch_count, scalars, (T*)work_workArr, workArr);

            // apply the block reflector
            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_backward_direction, rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
                shiftA + idx2D(0, n - k + j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
                A, shiftA, lda, strideA, batch_count, (T*)work_workArr, diag_tmptr, workArr);
        }
        j -= nb;
        mu = m - k + j + jb;
        nu = n - k + j + jb;
    }

    // factor last block
    if(mu > 0 && nu > 0)
        rocsolver_geql2_template<T>(handle, mu, nu, A, shiftA, lda, strideA, ipiv, strideP,
                                    batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                    diag_tmptr);

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQLF_H */
