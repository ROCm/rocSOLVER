/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

template <bool BATCHED, typename T>
void rocsolver_gebd2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_both, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(max(m, n), batch_count, &w2, &s2);
    *size_work_workArr = max(w1, w2);
    *size_Abyx_norms = max(s1, s2);
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_gebd2_gebrd_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tauq,
                                              U taup,
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
    if((m * n && !A) || (m * n && !D) || (m * n && !E) || (m * n && !tauq) || (m * n && !taup))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gebd2_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tauq,
                                        const rocblas_stride strideQ,
                                        T* taup,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms)
{
    ROCSOLVER_ENTER("gebd2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = min(m, n); // total number of pivots

    if(m >= n)
    {
        // generate upper bidiagonal form
        for(rocblas_int j = 0; j < n; j++)
        {
            // generate Householder reflector H(j)
            rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                     shiftA + idx2D(min(j + 1, m - 1), j, lda), 1, strideA,
                                     (tauq + j), strideQ, batch_count, (T*)work_workArr, Abyx_norms);

            // copy A(j,j) to D and insert one to build/apply the householder matrix
            ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, D,
                                    j, strideD, A, shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

            // Apply Householder reflector H(j)
            if(j < n - 1)
            {
                // conjugate tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                        shiftA + idx2D(j, j, lda), 1, strideA, (tauq + j), strideQ,
                                        A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                        scalars, Abyx_norms, (T**)work_workArr);

                // restore tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);
            }

            // restore original value of A(j,j)
            ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                    stream, D, j, strideD, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, 1);

            if(j < n - 1)
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                                lda, strideA, batch_count);

                // generate Householder reflector G(j)
                rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda), A,
                                         shiftA + idx2D(j, min(j + 2, n - 1), lda), lda, strideA,
                                         (taup + j), strideP, batch_count, (T*)work_workArr,
                                         Abyx_norms);

                // copy A(j,j+1) to E and insert one to build/apply the householder
                // matrix
                ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                        stream, E, j, strideE, A, shiftA + idx2D(j, j + 1, lda),
                                        lda, strideA, 1, true);

                // Apply Householder reflector G(j)
                rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j - 1, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, (taup + j),
                                        strideP, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                                lda, strideA, batch_count);

                // restore original value of A(j,j+1)
                ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                        stream, E, j, strideE, A, shiftA + idx2D(j, j + 1, lda),
                                        lda, strideA, 1);
            }
            else
            {
                // zero taup(j)
                ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0,
                                        stream, taup + j, strideP, 1, 0);
            }
        }
    }
    else
    {
        // generate lower bidiagonal form
        for(rocblas_int j = 0; j < m; j++)
        {
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                            strideA, batch_count);

            // generate Householder reflector G(j)
            rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                                     shiftA + idx2D(j, min(j + 1, n - 1), lda), lda, strideA,
                                     (taup + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

            // copy A(j,j) to D and insert one to build/apply the householder matrix
            ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, D,
                                    j, strideD, A, shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

            // Apply Householder reflector G(j)
            if(j < m - 1)
            {
                rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                        shiftA + idx2D(j, j, lda), lda, strideA, (taup + j),
                                        strideP, A, shiftA + idx2D(j + 1, j, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);
            }

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                            strideA, batch_count);

            // restore original value of A(j,j)
            ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                    stream, D, j, strideD, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, 1);

            if(j < m - 1)
            {
                // generate Householder reflector H(j)
                rocsolver_larfg_template(handle, m - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                         shiftA + idx2D(min(j + 2, m - 1), j, lda), 1, strideA,
                                         (tauq + j), strideQ, batch_count, (T*)work_workArr,
                                         Abyx_norms);

                // copy A(j+1,j) to D and insert one to build/apply the householder
                // matrix
                ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                        stream, E, j, strideE, A, shiftA + idx2D(j + 1, j, lda),
                                        lda, strideA, 1, true);

                // conjugate tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                // Apply Householder reflector H(j)
                rocsolver_larf_template(handle, rocblas_side_left, m - j - 1, n - j - 1, A,
                                        shiftA + idx2D(j + 1, j, lda), 1, strideA, (tauq + j),
                                        strideQ, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);

                // restore tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                // restore original value of A(j,j+1)
                ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                        stream, E, j, strideE, A, shiftA + idx2D(j + 1, j, lda),
                                        lda, strideA, 1);
            }
            else
            {
                // zero tauq(j)
                ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0,
                                        stream, tauq + j, strideQ, 1, 0);
            }
        }
    }

    return rocblas_status_success;
}
