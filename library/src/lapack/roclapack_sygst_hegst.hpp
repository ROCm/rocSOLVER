/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_sygs2_hegs2.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_sygst_hegst_getMemorySize(const rocblas_eform itype,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work_x_temp,
                                         size_t* size_workArr_temp_arr,
                                         size_t* size_store_invA,
                                         size_t* size_invA_arr)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_store_invA = 0;
        *size_invA_arr = 0;
        return;
    }

    if(n < xxGST_xxGS2_SWITCHSIZE)
    {
        // requirements for calling a single SYGS2/HEGS2
        rocsolver_sygs2_hegs2_getMemorySize<T, BATCHED>(itype, n, batch_count, size_scalars,
                                                        size_work_x_temp, size_workArr_temp_arr,
                                                        size_store_invA, size_invA_arr);
    }
    else
    {
        rocblas_int kb = xxGST_xxGS2_SWITCHSIZE;
        size_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

        // requirements for calling SYGS2/HEGS2 for the subblocks
        rocsolver_sygs2_hegs2_getMemorySize<T, BATCHED>(itype, kb, batch_count, size_scalars,
                                                        size_work_x_temp, size_workArr_temp_arr,
                                                        size_store_invA, size_invA_arr);

        if(itype == rocblas_eform_ax)
        {
            // extra requirements for calling TRSM
            rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, n - kb, kb, batch_count, &temp1,
                                             &temp2, &temp3, &temp4);
            rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, n - kb, kb, batch_count, &temp5,
                                             &temp6, &temp7, &temp8);

            *size_work_x_temp = max(*size_work_x_temp, max(temp1, temp5));
            *size_workArr_temp_arr = max(*size_workArr_temp_arr, max(temp2, temp6));
            *size_store_invA = max(*size_store_invA, max(temp3, temp7));
            *size_invA_arr = max(*size_invA_arr, max(temp4, temp8));
        }
        else
        {
            // extra requirements for calling TRMM
            temp1 = 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB * sizeof(T) * batch_count;
            *size_work_x_temp = max(*size_work_x_temp, temp1);
        }
    }
}

template <bool BATCHED, bool STRIDED, typename S, typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_sygst_hegst_template(rocblas_handle handle,
                                              const rocblas_eform itype,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              U B,
                                              const rocblas_int shiftB,
                                              const rocblas_int ldb,
                                              const rocblas_stride strideB,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              void* work_x_temp,
                                              void* workArr_temp_arr,
                                              void* store_invA,
                                              void* invA_arr,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("sygst_hegst", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int bs = xxGST_xxGS2_BLOCKSIZE;
    rocblas_int nb = xxGST_xxGS2_SWITCHSIZE;

    // if the matrix is too small, use the unblocked variant of the algorithm
    if(n <= nb)
        return rocsolver_sygs2_hegs2_template<BATCHED, T>(
            handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
            scalars, work_x_temp, workArr_temp_arr, store_invA, invA_arr);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    S s_one = 1;
    T t_one = 1;
    T t_half = 0.5;
    T t_minone = -1;
    T t_minhalf = -0.5;

    if(itype == rocblas_eform_ax)
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute inv(U')*A*inv(U)
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = min(n - k, nb);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    workArr_temp_arr, store_invA, invA_arr);

                if(k + kb < n)
                {
                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_left, uplo, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, kb, n - k - kb, &t_one, B,
                        shiftB + idx2D(k, k, ldb), ldb, strideB, A, shiftA + idx2D(k, k + kb, lda),
                        lda, strideA, batch_count, optim_mem, work_x_temp, workArr_temp_arr,
                        store_invA, invA_arr);

                    rocblasCall_symm_hemm<T>(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                             &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA,
                                             B, shiftB + idx2D(k, k + kb, ldb), ldb, strideB,
                                             &t_one, A, shiftA + idx2D(k, k + kb, lda), lda,
                                             strideA, batch_count);

                    rocblasCall_syr2k_her2k<T>(
                        handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
                        &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
                        shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm<T>(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                             &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA,
                                             B, shiftB + idx2D(k, k + kb, ldb), ldb, strideB,
                                             &t_one, A, shiftA + idx2D(k, k + kb, lda), lda,
                                             strideA, batch_count);

                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_right, uplo, rocblas_operation_none,
                        rocblas_diagonal_non_unit, kb, n - k - kb, &t_one, B,
                        shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_invA, invA_arr);
                }
            }
        }
        else
        {
            // Compute inv(L)*A*inv(L')
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = min(n - k, nb);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    workArr_temp_arr, store_invA, invA_arr);

                if(k + kb < n)
                {
                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_right, uplo, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, n - k - kb, kb, &t_one, B,
                        shiftB + idx2D(k, k, ldb), ldb, strideB, A, shiftA + idx2D(k + kb, k, lda),
                        lda, strideA, batch_count, optim_mem, work_x_temp, workArr_temp_arr,
                        store_invA, invA_arr);

                    rocblasCall_symm_hemm<T>(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                             &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA,
                                             B, shiftB + idx2D(k + kb, k, ldb), ldb, strideB,
                                             &t_one, A, shiftA + idx2D(k + kb, k, lda), lda,
                                             strideA, batch_count);

                    rocblasCall_syr2k_her2k<T>(
                        handle, uplo, rocblas_operation_none, n - k - kb, kb, &t_minone, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, B,
                        shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm<T>(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                             &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA,
                                             B, shiftB + idx2D(k + kb, k, ldb), ldb, strideB,
                                             &t_one, A, shiftA + idx2D(k + kb, k, lda), lda,
                                             strideA, batch_count);

                    rocblasCall_trsm<BATCHED, T>(
                        handle, rocblas_side_left, uplo, rocblas_operation_none,
                        rocblas_diagonal_non_unit, n - k - kb, kb, &t_one, B,
                        shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_invA, invA_arr);
                }
            }
        }
    }
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute U*A*U'
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = min(n - k, nb);

                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, uplo, rocblas_operation_none,
                    rocblas_diagonal_non_unit, k, kb, &t_one, B, shiftB, ldb, strideB, A,
                    shiftA + idx2D(0, k, lda), lda, strideA, batch_count, (T*)work_x_temp,
                    (T**)workArr_temp_arr);

                rocblasCall_symm_hemm<T>(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                         shiftA + idx2D(k, k, lda), lda, strideA, B,
                                         shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                         shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<T>(handle, uplo, rocblas_operation_none, k, kb, &t_one, A,
                                           shiftA + idx2D(0, k, lda), lda, strideA, B,
                                           shiftB + idx2D(0, k, ldb), ldb, strideB, &s_one, A,
                                           shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm<T>(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                         shiftA + idx2D(k, k, lda), lda, strideA, B,
                                         shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                         shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, uplo, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, k, kb, &t_one, B, shiftB + idx2D(k, k, ldb), ldb,
                    strideB, A, shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                    (T*)work_x_temp, (T**)workArr_temp_arr);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    workArr_temp_arr, store_invA, invA_arr);
            }
        }
        else
        {
            // Compute L'*A*L
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = min(n - k, nb);

                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, uplo, rocblas_operation_none,
                    rocblas_diagonal_non_unit, kb, k, &t_one, B, shiftB, ldb, strideB, A,
                    shiftA + idx2D(k, 0, lda), lda, strideA, batch_count, (T*)work_x_temp,
                    (T**)workArr_temp_arr);

                rocblasCall_symm_hemm<T>(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                         shiftA + idx2D(k, k, lda), lda, strideA, B,
                                         shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                         shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<T>(handle, uplo, rocblas_operation_conjugate_transpose, k,
                                           kb, &t_one, A, shiftA + idx2D(k, 0, lda), lda, strideA,
                                           B, shiftB + idx2D(k, 0, ldb), ldb, strideB, &s_one, A,
                                           shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm<T>(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                         shiftA + idx2D(k, k, lda), lda, strideA, B,
                                         shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                         shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, uplo, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, kb, k, &t_one, B, shiftB + idx2D(k, k, ldb), ldb,
                    strideB, A, shiftA + idx2D(k, 0, lda), lda, strideA, batch_count,
                    (T*)work_x_temp, (T**)workArr_temp_arr);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    workArr_temp_arr, store_invA, invA_arr);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
