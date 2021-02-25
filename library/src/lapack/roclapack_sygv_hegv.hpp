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
#include "roclapack_potrf.hpp"
#include "roclapack_syev_heev.hpp"
#include "roclapack_sygst_hegst.hpp"
#include "rocsolver.h"

template <typename T>
__global__ void sygv_update_info(T* info, T* iinfo, const rocblas_int n, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] != 0)
            info[b] += n;
        else
            info[b] = iinfo[b];
    }
}

template <bool BATCHED, typename T, typename S>
void rocsolver_sygv_hegv_getMemorySize(const rocblas_eform itype,
                                       const rocblas_evect jobz,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int batch_count,
                                       size_t* size_scalars,
                                       size_t* size_work1,
                                       size_t* size_work2,
                                       size_t* size_work3,
                                       size_t* size_work4,
                                       size_t* size_pivots_workArr,
                                       size_t* size_iinfo)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots_workArr = 0;
        *size_iinfo = 0;
        return;
    }

    size_t unused, temp1, temp2, temp3, temp4, temp5;

    // requirements for calling POTRF
    rocsolver_potrf_getMemorySize<BATCHED, T>(n, uplo, batch_count, size_scalars, size_work1,
                                              size_work2, size_work3, size_work4,
                                              size_pivots_workArr, size_iinfo);
    *size_iinfo = max(*size_iinfo, sizeof(rocblas_int) * batch_count);

    // requirements for calling SYGST/HEGST
    rocsolver_sygst_hegst_getMemorySize<T, BATCHED>(itype, n, batch_count, &unused, &temp1, &temp2,
                                                    &temp3, &temp4);
    *size_work1 = max(*size_work1, temp1);
    *size_work2 = max(*size_work2, temp2);
    *size_work3 = max(*size_work3, temp3);
    *size_work4 = max(*size_work4, temp4);

    // requirements for calling SYEV/HEEV
    rocsolver_syev_heev_getMemorySize<BATCHED, T, S>(jobz, uplo, n, batch_count, &unused, &temp1,
                                                     &temp2, &temp3, &temp4, &temp5);
    *size_work1 = max(*size_work1, temp1);
    *size_work2 = max(*size_work2, temp2);
    *size_work3 = max(*size_work3, temp3);
    *size_work4 = max(*size_work4, temp4);
    *size_pivots_workArr = max(*size_pivots_workArr, temp5);

    if(jobz == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            // requirements for calling TRSM
            rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, n, n, batch_count, &temp1, &temp2,
                                             &temp3, &temp4);
            *size_work1 = max(*size_work1, temp1);
            *size_work2 = max(*size_work2, temp2);
            *size_work3 = max(*size_work3, temp3);
            *size_work4 = max(*size_work4, temp4);
        }
    }
}

template <typename S, typename T>
rocblas_status rocsolver_sygv_hegv_argCheck(rocblas_handle handle,
                                            const rocblas_eform itype,
                                            const rocblas_evect jobz,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            const rocblas_int lda,
                                            const rocblas_int ldb,
                                            T A,
                                            T B,
                                            S D,
                                            S E,
                                            rocblas_int* info,
                                            const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
        return rocblas_status_invalid_value;
    if(jobz != rocblas_evect_none && jobz != rocblas_evect_original)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !B) || (n && !D) || (n && !E) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_sygv_hegv_template(rocblas_handle handle,
                                            const rocblas_eform itype,
                                            const rocblas_evect jobz,
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
                                            S* D,
                                            const rocblas_stride strideD,
                                            S* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count,
                                            T* scalars,
                                            void* work1,
                                            void* work2,
                                            void* work3,
                                            void* work4,
                                            void* pivots_workArr,
                                            rocblas_int* iinfo,
                                            bool optim_mem)
{
    ROCSOLVER_ENTER("sygv_hegv", "itype:", itype, "jobz:", jobz, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with no errors)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants for rocblas functions calls
    T one = 1;

    // perform Cholesky factorization of B
    rocsolver_potrf_template<BATCHED, S, T>(handle, uplo, n, B, shiftB, ldb, strideB, info,
                                            batch_count, scalars, work1, work2, work3, work4,
                                            (T*)pivots_workArr, iinfo, optim_mem);

    /** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
        A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
        Need to find a way to do this efficiently; for now A will be destroyed in the non
        positive-definite case) **/

    // reduce to standard eigenvalue problem and solve
    rocsolver_sygst_hegst_template<BATCHED, STRIDED, S, T>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        scalars, work1, work2, work3, work4, optim_mem);

    rocsolver_syev_heev_template<BATCHED, STRIDED, T>(
        handle, jobz, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
        scalars, work1, (T*)work2, (T*)work3, (T*)work4, (T**)pivots_workArr);

    // combine info from POTRF with info from SYEV/HEEV
    hipLaunchKernelGGL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n, batch_count);

    /** (TODO: Similarly, if only neig < n eigenvalues converged, TRSMM or TRMM below should not
        work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
        iinfo and set neig = n) **/

    rocblas_int neig = n; //number of converged eigenvalues

    // backtransform eigenvectors
    if(jobz == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            rocblas_operation trans
                = (uplo == rocblas_fill_upper ? rocblas_operation_none
                                              : rocblas_operation_conjugate_transpose);
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, uplo, trans,
                                         rocblas_diagonal_non_unit, n, neig, &one, B, shiftB, ldb,
                                         strideB, A, shiftA, lda, strideA, batch_count, optim_mem,
                                         work1, work2, work3, work4);
        }
        else
        {
            rocblas_operation trans
                = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose
                                              : rocblas_operation_none);
            rocblasCall_trmm<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n, neig, &one, 0,
                B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, (T**)pivots_workArr);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
