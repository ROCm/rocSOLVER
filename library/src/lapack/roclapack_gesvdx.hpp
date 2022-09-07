/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_bdsvdx.hpp"
#include "auxiliary/rocauxiliary_ormbr_unmbr.hpp"
#include "auxiliary/rocauxiliary_ormlq_unmlq.hpp"
#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "roclapack_gesvd.hpp"
#include "rocsolver/rocsolver.h"

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_argCheck(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_srange srange,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_int lda,
                                         const TT vl,
                                         const TT vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         TT* S,
                                         T* U,
                                         const rocblas_int ldu,
                                         T* V,
                                         const rocblas_int ldv,
                                         rocblas_int* ifail,
                                         rocblas_int* info,
                                         const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_singular && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_singular && right_svect != rocblas_svect_none))
        return rocblas_status_invalid_value;
    if(srange != rocblas_srange_all && srange != rocblas_srange_value
       && srange != rocblas_srange_index)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(left_svect == rocblas_svect_singular && ldu < m)
        return rocblas_status_invalid_size;
    if(right_svect == rocblas_svect_singular && ldv < min(m, n))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n * m && !A) || (min(m, n) && !S) || (min(m, n) && !ifail) || (batch_count && !info)
       || (batch_count && !nsv))
        return rocblas_status_invalid_pointer;
    if(left_svect == rocblas_svect_singular && min(m, n) && !U)
        return rocblas_status_invalid_pointer;
    if(right_svect == rocblas_svect_singular && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_gesvdx_getMemorySize(const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int batch_count,
                                    const rocblas_workmode fast_alg,
                                    size_t* size_scalars,
                                    size_t* size_work_workArr,
                                    size_t* size_Abyx_norms_tmptr,
                                    size_t* size_Abyx_norms_trfact_X,
                                    size_t* size_diag_tmptr_Y,
                                    size_t* size_tau,
                                    size_t* size_tempArrayT,
                                    size_t* size_tempArrayC,
                                    size_t* size_workArr)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_tmptr = 0;
        *size_Abyx_norms_trfact_X = 0;
        *size_diag_tmptr_Y = 0;
        *size_tau = 0;
        *size_tempArrayT = 0;
        *size_tempArrayC = 0;
        *size_workArr = 0;
        return;
    }

    size_t w[6] = {};
    size_t a[5] = {};
    size_t x[6] = {};
    size_t y[3] = {};
    size_t unused;

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    //const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    //const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    //const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = min(m, n);
    const rocblas_int kk = max(m, n);
    const rocblas_int nu = leftvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : m);
    const rocblas_int nv = rightvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : n);
    const rocblas_storev storev_lead = row ? rocblas_column_wise : rocblas_row_wise;
    const rocblas_storev storev_other = row ? rocblas_row_wise : rocblas_column_wise;
    const rocblas_side side = row ? rocblas_side_right : rocblas_side_left;
    rocblas_int mn;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // size of array tau to store householder scalars on intermediate
    // orthonormal/unitary matrices
    *size_tau = 2 * sizeof(T) * min(m, n) * batch_count;

    // size of arrays to store temporary copies
    *size_tempArrayT
        = (fast_thinSVD || (thinSVD && leadvO && othervN)) ? sizeof(T) * k * k * batch_count : 0;
    *size_tempArrayC
        = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

    // workspace required for the bidiagonalization
    if(thinSVD)
        rocsolver_gebrd_getMemorySize<BATCHED, T>(k, k, batch_count, size_scalars, &w[0], &a[0],
                                                  &x[0], &y[0]);
    else
        rocsolver_gebrd_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, &w[0], &a[0],
                                                  &x[0], &y[0]);

    // workspace required for the SVD of the bidiagonal form
    rocsolver_bdsqr_getMemorySize<S>(k, nv, nu, 0, batch_count, &w[1]);

    // extra requirements for QR/LQ factorization
    if(thinSVD)
    {
        if(row)
            rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                      &y[1], &unused);
        else
            rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                      &y[1], &unused);
    }

    // extra requirements for orthonormal/unitary matrix generation
    // ormbr
    if(thinSVD && !fast_thinSVD && !leadvN)
        rocsolver_ormbr_unmbr_getMemorySize<BATCHED, T>(storev_lead, side, m, n, k, batch_count,
                                                        &unused, &a[1], &y[2], &x[2], &unused);
    // orgbr
    if(thinSVD)
    {
        if(!othervN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_other, k, k, k, batch_count,
                                                            &unused, &w[3], &a[2], &x[3], &unused);

        if(fast_thinSVD && !leadvN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_lead, k, k, k, batch_count,
                                                            &unused, &w[4], &a[3], &x[4], &unused);
    }
    else
    {
        mn = (row && leftvS) ? n : m;
        if(leftvS || leftvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_column_wise, m, mn, n, batch_count, &unused, &w[3], &a[2], &x[3], &unused);
        else if(leftvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_column_wise, m, k, n, batch_count, &unused, &w[3], &a[2], &x[3], &unused);

        mn = (!row && rightvS) ? m : n;
        if(rightvS || rightvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, mn, n, m, batch_count,
                                                            &unused, &w[4], &a[3], &x[4], &unused);
        else if(rightvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, k, n, m, batch_count,
                                                            &unused, &w[4], &a[3], &x[4], &unused);
    }
    // orgqr/orglq
    if(thinSVD && !leadvN)
    {
        if(leadvA)
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, &unused,
                                                                &w[5], &a[4], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, &unused,
                                                                &w[5], &a[4], &x[5], &unused);
        }
        else
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, n, k, batch_count, &unused,
                                                                &w[5], &a[4], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(m, n, k, batch_count, &unused,
                                                                &w[5], &a[4], &x[5], &unused);
        }
    }

    // get max sizes
    *size_work_workArr = *std::max_element(std::begin(w), std::end(w));
    *size_Abyx_norms_tmptr = *std::max_element(std::begin(a), std::end(a));
    *size_Abyx_norms_trfact_X = *std::max_element(std::begin(x), std::end(x));
    *size_diag_tmptr_Y = *std::max_element(std::begin(y), std::end(y));
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_template(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_srange srange,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         const TT vl,
                                         const TT vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         TT* S,
                                         const rocblas_stride strideS,
                                         T* U,
                                         const rocblas_int ldu,
                                         const rocblas_stride strideU,
                                         T* V,
                                         const rocblas_int ldv,
                                         const rocblas_stride strideV,
                                         rocblas_int* ifail,
                                         const rocblas_stride strideF,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("gesvd", "leftsv:", left_svect, "rightsv:", right_svect, "srange:", srange,
                    "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "vl:", vl, "vu:", vu,
                    "il:", il, "iu:", iu, "ldu:", ldu, "ldv:", ldv, "bc:", batch_count);

    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // quick return
    if(n == 0 || m == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1;
    T zero = 0;

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);

    // auxiliary sizes and variables
    rocblas_fill uplo = row ? rocblas_fill_upper : rocblas_fill_lower;
    const rocblas_svect svect = (leftvS || rightvS) ? rocblas_svect_singular : rocblas_svect_none;
    const rocblas_int thread_count = BS2;
    const rocblas_int k = min(m, n);
    const rocblas_int ldt = k;
    const rocblas_int ldx = thinSVD ? k : m;
    const rocblas_int ldy = thinSVD ? k : n;
    const rocblas_int ldz = 2 * k;
    const rocblas_stride strideD = k;
    const rocblas_stride strideE = k;
    const rocblas_stride strideT = k * k;
    const rocblas_stride strideX = ldx * GEBRD_GEBD2_SWITCHSIZE;
    const rocblas_stride strideY = ldy * GEBRD_GEBD2_SWITCHSIZE;
    const rocblas_stride strideZ = 2 * k * k;
    T* UV;
    rocblas_stride strideUV;
    rocblas_int lduv;
    rocblas_int offZ;
    bool trans,

        // common block sizes and number of threads for internal kernels
        const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;

    /***** 1. bidiagonalization *****/
    /********************************/
    if(thinSVD)
    {
        // apply qr/lq factorization
        local_geqrlq_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, tau, k,
                                                batch_count, scalars, lqrfWS1, lqrfWS2, lqrfWS3,
                                                workArr, row);

        // copy triangular factor
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                lda, strideA, tmpT, 0, ldt, strideT, no_mask{}, uplo);

        // clean triangular factor
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, k, k, tmpT, 0, ldt,
                                strideT, uplo);

        // apply gebrd to triangular factor
        rocsolver_gebrd_template<false, STRIDED>(handle, k, k, tmpT, 0, ldt, strideT, tmpD, strideD,
                                                 tmpE, strideE, tauqp, k, (tauqp + k * batch_count),
                                                 k, tmpX, 0, ldx, strideX, tmpY, 0, ldy, strideY,
                                                 batch_count, scalars, brdWS1, brdWS2);
    }
    else
    {
        // apply gebrd to matrix A
        rocsolver_gebrd_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, tmpD, strideD, tmpE, strideE, tauqp, k,
            (tauqp + k * batch_count), k, tmpX, 0, ldx, strideX, tmpY, 0, ldy, strideY, batch_count,
            scalars, brdWS1, brdWS2);
    }

    /***** 2. solve bidiagonal problem *****/
    /***************************************/
    // compute SVD of bidiagonal matrix
    uplo = thinSVD ? rocblas_fill_upper : uplo;
    rocsolver_bdsvdx_template(handle, uplo, svect, srange, k, tmpD, strideD, tmpE, strideE, vl, vu,
                              il, iu, nsv, S, strideS, tmpZ, 0, ldz, strideZ, ifail, strideF, info,
                              batch_count, svdxWS1, svdxWS2, svdxWS3, svdxWS4, svdxWS5, svdxWS6,
                              svdxWS7, svdxWS8, svdxWS9, svdxWS10, svdxWS11, svdxWS12);

    /***** 3. compute/update left vectors *****/
    /******************************************/
    if(leftvS)
    {
        // For now we work with the k columns in tmpZ, i.e. k vectors
        // (TODO: explore other options like transfering nsv to the host)

        // initialize matrix U
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_m, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, m, k, U, 0, ldu,
                                strideU);

        // copy left vectors to matrix U
        ROCSOLVER_LAUNCH_KERNEL(copy_trans_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, false, k, k, tmpZ,
                                0, ldz, strideZ, U, 0, ldu, strideU);

        if(thinSVD)
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocsolver_ormbr_unmbr_template<false, STRIDED>(
                handle, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, k, k, k,
                tmpT, 0, ldt, strideT, tauqp, k, U, 0, ldu, strideU, batch_count, scalars, mbrWS1,
                mbrWS2, mbrWS3, workArr);

            if(row)
            {
                // apply ormqr (update with transformation from row compression)
                rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
                    handle, rocblas_side_left, rocblas_operation_none, m, k, k, A, shiftA, lda,
                    strideA, tau, k, U, 0, ldu, strideU, batch_count, scalars, mlqrWS1, mlqrWS2,
                    mlqrWS3, workArr);
            }
        }
        else
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocblas_int mm = row ? m : k;
            rocblas_int kk = row ? k : n;
            rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
                handle, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, mm, k, kk,
                A, shiftA, lda, strideA, tauqp, k, U, 0, ldu, strideU, batch_count, scalars, mbrWS1,
                mbrWS2, mbrWS3, workArr);
        }
    }

    /***** 4. compute/update right vectors *****/
    /**********************************************/
    if(rightvS)
    {
        // For now we work with the k columns in tmpZ, i.e. k vectors
        // (TODO: explore other options like transfering nsv to the host)

        // initialize matrix V
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_n, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, k, n, V, 0, ldv,
                                strideV);

        // copy right vectors to matrix V
        ROCSOLVER_LAUNCH_KERNEL(copy_trans_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, true, k, k, tmpZ, k,
                                ldz, strideZ, V, 0, ldv, strideV);

        if(thinSVD)
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocsolver_ormbr_unmbr_template<false, STRIDED>(
                handle, rocblas_row_wise, rocblas_side_right, rocblas_operation_transpose, k, k, k,
                tmpT, 0, ldt, strideT, (tauqp + k * batch_count), k, V, 0, ldv, strideV,
                batch_count, scalars, mbrWS1, mbrWS2, mbrWS3, workArr);

            if(!row)
            {
                // apply ormlq (update with transformation from column compression)
                rocsolver_ormlq_unmlq_template<BATCHED, STRIDED>(
                    handle, rocblas_side_right, rocblas_operation_none, k, n, k, A, shiftA, lda,
                    strideA, tau, k, V, 0, ldv, strideV, batch_count, scalars, mlqrWS1, mlqrWS2,
                    mlqrWS3, workArr);
            }
        }
        else
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocblas_int nn = row ? k : n;
            rocblas_int kk = row ? m : k;
            rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
                handle, rocblas_row_wise, rocblas_side_right, rocblas_operation_transpose, k, nn,
                kk, A, shiftA, lda, strideA, (tauqp + k * batch_count), k, V, 0, ldv, strideV,
                batch_count, scalars, mbrWS1, mbrWS2, mbrWS3, workArr);
        }
    }

    return rocblas_status_success;
}
