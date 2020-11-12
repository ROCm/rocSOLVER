/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELS_HPP
#define ROCLAPACK_GELS_HPP

#include "../auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver.h"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_gels_getMemorySize(const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int nrhs,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work_x_temp,
                                  size_t* size_workArr_temp_arr,
                                  size_t* size_diag_trfac_invA,
                                  size_t* size_trfact_workTrmm_invA_arr,
                                  size_t* size_ipiv)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_diag_trfac_invA = 0;
        *size_trfact_workTrmm_invA_arr = 0;
        *size_ipiv = 0;
        return;
    }

    size_t geqrf_scalars, geqrf_work, geqrf_workArr, geqrf_diag, geqrf_trfact;
    rocsolver_geqrf_getMemorySize<T, BATCHED>(m, n, batch_count, &geqrf_scalars, &geqrf_work,
                                              &geqrf_workArr, &geqrf_diag, &geqrf_trfact);

    size_t ormqr_scalars, ormqr_work, ormqr_workArr, ormqr_trfact, ormqr_workTrmm;
    rocsolver_ormqr_unmqr_getMemorySize<T, BATCHED>(rocblas_side_left, m, nrhs, n, batch_count,
                                                    &ormqr_scalars, &ormqr_work, &ormqr_workArr,
                                                    &ormqr_trfact, &ormqr_workTrmm);

    size_t trsm_x_temp, trsm_x_temp_arr, trsm_invA, trsm_invA_arr;
    rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, n, nrhs, batch_count, &trsm_x_temp,
                                     &trsm_x_temp_arr, &trsm_invA, &trsm_invA_arr);

    ROCSOLVER_ASSUME_X(geqrf_scalars == ormqr_scalars, "GEQRF and ORMQR use the same scalars");

    // TODO: rearrange to minimize total size
    *size_scalars = geqrf_scalars;
    *size_work_x_temp = std::max({geqrf_work, ormqr_work, trsm_x_temp});
    *size_workArr_temp_arr = std::max({geqrf_workArr, ormqr_workArr, trsm_x_temp_arr});
    *size_diag_trfac_invA = std::max({geqrf_diag, ormqr_trfact, trsm_invA});
    *size_trfact_workTrmm_invA_arr = std::max({geqrf_trfact, ormqr_workTrmm, trsm_invA_arr});

    const rocblas_int pivot_count_per_batch = std::min(m, n);
    *size_ipiv = sizeof(T) * pivot_count_per_batch * batch_count;
}

template <typename T>
rocblas_status rocsolver_gels_argCheck(rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       T A,
                                       const rocblas_int lda,
                                       T C,
                                       const rocblas_int ldc,
                                       rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:
    // 1. non-supported values
    if(m < n || trans == rocblas_operation_transpose || trans == rocblas_operation_conjugate_transpose)
        return rocblas_status_not_implemented;

    // 2. invalid values
    if(trans != rocblas_operation_none)
        return rocblas_status_invalid_value;

    // 3. invalid size
    if(m < 0 || n < 0 || nrhs < 0 || lda < m || ldc < m || ldc < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 4. invalid pointers
    if((m * n && !A) || ((m * nrhs || n * nrhs) && !C) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gels_template(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       U C,
                                       const rocblas_int shiftC,
                                       const rocblas_int ldc,
                                       const rocblas_stride strideC,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       T* work_x_temp,
                                       T* workArr_temp_arr,
                                       T* diag_trfac_invA,
                                       T** trfact_workTrmm_invA_arr,
                                       T* ipiv,
                                       bool optim_mem)
{
    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if A or C are empty
    // TODO: clear C as needed when underdetermined support is added
    if(m == 0 || n == 0 || nrhs == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // TODO: apply scaling to improve accuracy over a larger range of values

    // note: m >= n
    // compute QR factorization of A
    const rocblas_stride strideP = std::min(m, n);
    rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                               batch_count, scalars, work_x_temp, workArr_temp_arr,
                                               diag_trfac_invA, trfact_workTrmm_invA_arr);
    rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
        handle, rocblas_side_left, rocblas_operation_conjugate_transpose, m, nrhs, n, A, shiftA,
        lda, strideA, ipiv, strideP, C, shiftC, ldc, strideC, batch_count, scalars, (T*)work_x_temp,
        (T*)workArr_temp_arr, (T*)diag_trfac_invA, (T**)trfact_workTrmm_invA_arr);

    // do the equivalent of strtrs
    const rocblas_int check_threads = min(((n - 1) / 64 + 1) * 64, BLOCKSIZE);
    hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, check_threads, 1), 0,
                       stream, n, A, shiftA, lda, strideA, info);
    const T one = 1; // constant 1 in host memory
    // solve U*X = B, overwriting B with X
    rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper,
                                 rocblas_operation_none, rocblas_diagonal_non_unit, n, nrhs, &one,
                                 A, shiftA, lda, strideA, C, shiftC, ldc, strideC, batch_count,
                                 optim_mem, work_x_temp, workArr_temp_arr, diag_trfac_invA,
                                 trfact_workTrmm_invA_arr);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GELS_HPP */
