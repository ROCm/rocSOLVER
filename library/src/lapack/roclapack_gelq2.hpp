/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <bool BATCHED, typename T>
void rocsolver_gelq2_getMemorySize(const rocblas_int m,
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
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_right, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w2, &s2);
    *size_work_workArr = max(w1, w2);
    *size_Abyx_norms = max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_gelq2_gelqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
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

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m * n && !A) || (m * n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gelq2_template(rocblas_handle handle,
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
    ROCSOLVER_ENTER("gelq2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = min(m, n); // total number of pivots

    for(rocblas_int j = 0; j < dim; ++j)
    {
        // conjugate the jth row of A
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        batch_count);

        // generate Householder reflector to work on row j
        rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(j, min(j + 1, n - 1), lda), lda, strideA,
                                 (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // insert one in A(j,j) tobuild/apply the householder matrix
        ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                                diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

        // Apply Householder reflector to the rest of matrix from the right
        if(j < m - 1)
        {
            rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                    shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                    scalars, Abyx_norms, (T**)work_workArr);
        }

        // restore original value of A(j,j)
        ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                                diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1);

        // restore the jth row of A
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        batch_count);
    }

    return rocblas_status_success;
}
