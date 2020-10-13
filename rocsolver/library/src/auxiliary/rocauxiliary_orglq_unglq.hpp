/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORGLQ_UNGLQ_HPP
#define ROCLAPACK_ORGLQ_UNGLQ_HPP

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_orgl2_ungl2.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_orglq_unglq_getMemorySize(const rocblas_int m,
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

    size_t s1, s2, s3, unused;
    rocsolver_orgl2_ungl2_getMemorySize<T, BATCHED>(m, n, batch_count, size_scalars,
                                                    size_Abyx_tmptr, size_workArr);

    if(k <= ORGxx_UNGxx_SWITCHSIZE)
    {
        *size_work = 0;
        *size_trfact = 0;
    }

    else
    {
        rocblas_int jb = ORGxx_UNGxx_BLOCKSIZE;
        rocblas_int j = ((k - ORGxx_UNGxx_SWITCHSIZE - 1) / jb) * jb;
        rocblas_int kk = min(k, j + jb);

        // size of workspace is maximum of what is needed by larft and larfb.
        // size of Abyx_tmptr is maximum of what is needed by orgl2/ungl2 and larfb.
        rocsolver_larft_getMemorySize<T, BATCHED>(n, jb, batch_count, &unused, &s1, &unused);
        rocsolver_larfb_getMemorySize<T, BATCHED>(rocblas_side_left, m - jb, n, jb, batch_count,
                                                  &s2, &s3, &unused);

        *size_work = max(s1, s2);
        *size_Abyx_tmptr = *size_Abyx_tmptr >= s3 ? *size_Abyx_tmptr : s3;

        // size of temporary array for triangular factor
        *size_trfact = sizeof(T) * jb * jb * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orglq_unglq_template(rocblas_handle handle,
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

    // if the matrix is small, use the unblocked variant of the algorithm
    if(k <= ORGxx_UNGxx_SWITCHSIZE)
        return rocsolver_orgl2_ungl2_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv,
                                                 strideP, batch_count, scalars, Abyx_tmptr, workArr);

    rocblas_int ldw = ORGxx_UNGxx_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    // start of first blocked block
    rocblas_int jb = ORGxx_UNGxx_BLOCKSIZE;
    rocblas_int j = ((k - ORGxx_UNGxx_SWITCHSIZE - 1) / jb) * jb;

    // start of the unblocked block
    rocblas_int kk = min(k, j + jb);

    rocblas_int blocksy, blocksx;

    // compute the unblockled part and set to zero the
    // corresponding left submatrix
    if(kk < m)
    {
        blocksx = (m - kk - 1) / 32 + 1;
        blocksy = (kk - 1) / 32 + 1;
        hipLaunchKernelGGL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                           stream, m - kk, kk, A, shiftA + idx2D(kk, 0, lda), lda, strideA);

        rocsolver_orgl2_ungl2_template<T>(handle, m - kk, n - kk, k - kk, A,
                                          shiftA + idx2D(kk, kk, lda), lda, strideA, (ipiv + kk),
                                          strideP, batch_count, scalars, Abyx_tmptr, workArr);
    }

    // compute the blocked part
    while(j >= 0)
    {
        // first update the already computed part
        // applying the current block reflector using larft + larfb
        if(j + jb < m)
        {
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_row_wise, n - j,
                                        jb, A, shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j),
                                        strideP, trfact, ldw, strideW, batch_count, scalars, work,
                                        workArr);

            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                rocblas_forward_direction, rocblas_row_wise, m - j - jb, n - j, jb, A,
                shiftA + idx2D(j, j, lda), lda, strideA, trfact, 0, ldw, strideW, A,
                shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count, work, Abyx_tmptr, workArr);
        }

        // now compute the current block and set to zero
        // the corresponding top submatrix
        if(j > 0)
        {
            blocksx = (jb - 1) / 32 + 1;
            blocksy = (j - 1) / 32 + 1;
            hipLaunchKernelGGL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                               stream, jb, j, A, shiftA + idx2D(j, 0, lda), lda, strideA);
        }
        rocsolver_orgl2_ungl2_template<T>(handle, jb, n - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                          strideA, (ipiv + j), strideP, batch_count, scalars,
                                          Abyx_tmptr, workArr);

        j -= jb;
    }

    return rocblas_status_success;
}

#endif
