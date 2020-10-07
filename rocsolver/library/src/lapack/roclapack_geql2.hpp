/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GEQL2_H
#define ROCLAPACK_GEQL2_H

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_geql2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<T, BATCHED>(rocblas_side_left, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(m, batch_count, &w2, &s2);
    *size_work_workArr = max(w1, w2);
    *size_Abyx_norms = max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_geql2_geqlf_argCheck(const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((m * n && !A) || (m * n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_geql2_template(rocblas_handle handle,
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
                                        T* Abyx_norms,
                                        T* diag)
{
    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = min(m, n); // total number of pivots

    for(rocblas_int j = 0; j < dim; j++)
    {
        // generate Householder reflector to work on column j
        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                                 shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                                 strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // insert one in A(m-j-1,n-j-1) tobuild/apply the householder matrix
        hipLaunchKernelGGL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, diag, 0,
                           1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1, true);

        // conjugate tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

        // Apply Householder reflector to the rest of matrix from the left
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                                strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                                (T**)work_workArr);

        // restore original value of A(m-j-1,n-j-1)
        hipLaunchKernelGGL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, diag,
                           0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);

        // restore tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQL2_H */
