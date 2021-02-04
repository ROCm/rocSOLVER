/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_bdsqr.hpp"
#include "../auxiliary/rocauxiliary_orgbr_ungbr.hpp"
#include "../auxiliary/rocauxiliary_ormbr_unmbr.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "roclapack_geqrf.hpp"
#include "roclapack_gelqf.hpp"
#include "rocsolver.h"

/** wrapper to xxGQR/xxGLQ_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_orgqrlq_ungqrlq_template(rocblas_handle handle,
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
                                              T** workArr,
                                              const bool row)
{
    if(row)
        rocsolver_orgqr_ungqr_template<BATCHED,STRIDED>(
                                        handle,m,n,k,A,shiftA,lda,strideA,ipiv,strideP,batch_count,
                                        scalars,work,Abyx_tmptr,trfact,workArr);

    else
        rocsolver_orglq_unglq_template<BATCHED,STRIDED>(
                                        handle,m,n,k,A,shiftA,lda,strideA,ipiv,strideP,batch_count,
                                        scalars,work,Abyx_tmptr,trfact,workArr);
}


/** wrapper to GEQRF/GELQF_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_geqrlq_template(rocblas_handle handle,
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
                          T** workArr,
                          const bool row)
{
    if(row)
        rocsolver_geqrf_template<BATCHED,STRIDED>(handle,m,n,A,shiftA,lda,strideA,ipiv,strideP,batch_count,
                                         scalars,work_workArr,Abyx_norms_trfact,
                                         diag_tmptr,workArr);
    else
        rocsolver_gelqf_template<BATCHED,STRIDED>(handle,m,n,A,shiftA,lda,strideA,ipiv,strideP,batch_count,
                                         scalars,work_workArr,Abyx_norms_trfact,
                                         diag_tmptr,workArr);
}

/** wrapper to BDSQR_TEMPLATE **/
template <typename T, typename TT>
void local_bdsqr_template(rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          TT* D,
                          const rocblas_stride strideD,
                          TT* E,
                          const rocblas_stride strideE,
                          T* V,
                          const rocblas_int shiftV,
                          const rocblas_int ldv,
                          const rocblas_stride strideV,
                          T* U,
                          const rocblas_int shiftU,
                          const rocblas_int ldu,
                          const rocblas_stride strideU,
                          rocblas_int* info,
                          const rocblas_int batch_count,
                          TT* work,
                          T** workArr)
{
    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, 0, D, strideD, E, strideE, V, shiftV, ldv,
                                strideV, U, shiftU, ldu, strideU, (T*)nullptr, 0, 1, 1, info,
                                batch_count, work);
}

/** wrapper to BDSQR_TEMPLATE
    adapts U and V to be of the same type **/
template <typename T, typename TT>
void local_bdsqr_template(rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          TT* D,
                          const rocblas_stride strideD,
                          TT* E,
                          const rocblas_stride strideE,
                          T* const V[],
                          const rocblas_int shiftV,
                          const rocblas_int ldv,
                          const rocblas_stride strideV,
                          T* U,
                          const rocblas_int shiftU,
                          const rocblas_int ldu,
                          const rocblas_stride strideU,
                          rocblas_int* info,
                          const rocblas_int batch_count,
                          TT* work,
                          T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, U, strideU,
                       batch_count);

    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, 0, D, strideD, E, strideE, V, shiftV, ldv,
                                strideV, (T* const*)workArr, shiftU, ldu, strideU,
                                (T* const*)nullptr, 0, 1, 1, info, batch_count, work);
}

/** wrapper to BDSQR_TEMPLATE
    adapts U and V to be of the same type **/
template <typename T, typename TT>
void local_bdsqr_template(rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          TT* D,
                          const rocblas_stride strideD,
                          TT* E,
                          const rocblas_stride strideE,
                          T* V,
                          const rocblas_int shiftV,
                          const rocblas_int ldv,
                          const rocblas_stride strideV,
                          T* const U[],
                          const rocblas_int shiftU,
                          const rocblas_int ldu,
                          const rocblas_stride strideU,
                          rocblas_int* info,
                          const rocblas_int batch_count,
                          TT* work,
                          T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, V, strideV,
                       batch_count);

    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, 0, D, strideD, E, strideE,
                                (T* const*)workArr, shiftV, ldv, strideV, U, shiftU, ldu, strideU,
                                (T* const*)nullptr, 0, 1, 1, info, batch_count, work);
}

/** wrapper to ORMBR_UNMBR_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T>
void local_ormbr_unmbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              T* A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              T* C,
                                              const rocblas_int shiftC,
                                              const rocblas_int ldc,
                                              const rocblas_stride strideC,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* AbyxORwork,
                                              T* diagORtmptr,
                                              T* trfact,
                                              T** workArr)
{
    rocsolver_ormbr_unmbr_template<BATCHED,STRIDED>(handle,storev,side,trans,m,n,k,A,shiftA,lda,strideA,
                                           ipiv,strideP,C,shiftC,ldc,strideC,batch_count,
                                           scalars,AbyxORwork,diagORtmptr,trfact,workArr);
}

/** wrapper to ORMBR_UNMBR_TEMPLATE
    Adapts A and C to be of the same type **/ 
template <bool BATCHED, bool STRIDED, typename T>
void local_ormbr_unmbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              T *const A[],
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              T* C,
                                              const rocblas_int shiftC,
                                              const rocblas_int ldc,
                                              const rocblas_stride strideC,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* AbyxORwork,
                                              T* diagORtmptr,
                                              T* trfact,
                                              T** workArr)
{
    T** CC = workArr + batch_count;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, CC, C, strideC,
                       batch_count);
    
    rocsolver_ormbr_unmbr_template<BATCHED,STRIDED>(handle,storev,side,trans,m,n,k,A,shiftA,lda,strideA,
                                           ipiv,strideP,(T* const*)CC,shiftC,ldc,strideC,batch_count,
                                           scalars,AbyxORwork,diagORtmptr,trfact,workArr);
}

/** wrapper to ORMBR_UNMBR_TEMPLATE
    Adapts A and C to be of the same type **/ 
template <bool BATCHED, bool STRIDED, typename T>
void local_ormbr_unmbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              T* A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              T *const C[],
                                              const rocblas_int shiftC,
                                              const rocblas_int ldc,
                                              const rocblas_stride strideC,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* AbyxORwork,
                                              T* diagORtmptr,
                                              T* trfact,
                                              T** workArr)
{
    T** AA = workArr + batch_count;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    hipLaunchKernelGGL(get_array, dim3(blocks), dim3(256), 0, stream, AA, A, strideA,
                       batch_count);

    rocsolver_ormbr_unmbr_template<BATCHED,STRIDED>(handle,storev,side,trans,m,n,k,(T* const*)AA,shiftA,lda,strideA,
                                           ipiv,strideP,C,shiftC,ldc,strideC,batch_count,
                                           scalars,AbyxORwork,diagORtmptr,trfact,workArr);
}

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        TT* E,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
        && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
           && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none)
       || (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if((right_svect == rocblas_svect_all && ldv < n)
       || (right_svect == rocblas_svect_singular && ldv < min(m, n)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n * m && !A) || (min(m, n) > 1 && !E) || (min(m, n) && !S) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_all && m && !U)
       || (left_svect == rocblas_svect_singular && min(m, n) && !U))
        return rocblas_status_invalid_pointer;
    if((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_gesvd_getMemorySize(const rocblas_svect left_svect,
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

    std::vector<size_t> w(6,0);
    std::vector<size_t> a(5,0);
    std::vector<size_t> x(6,0);
    std::vector<size_t> y(3,0);
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
    const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);
    const bool compressed = (thinSVD && (fast_thinSVD || leadvN || !leadvO || !othervN));
   
    // auxiliary sizes and variables
    const rocblas_int k = min(m, n);
    const rocblas_int kk = max(m, n);
    const rocblas_int nu = leftvN ? 0 : (fast_thinSVD ? k : m);
    const rocblas_int nv = rightvN ? 0 : (fast_thinSVD ? k : n);
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
    *size_tempArrayT = fast_thinSVD ? sizeof(T) * k * k * batch_count : 0;
    *size_tempArrayC = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

    // workspace required for the bidiagonalization
    if(compressed)
        rocsolver_gebrd_getMemorySize<T, BATCHED>(k, k, batch_count, size_scalars, &w[0], &a[0], &x[0], &y[0]);
    else
        rocsolver_gebrd_getMemorySize<T, BATCHED>(m, n, batch_count, size_scalars, &w[0], &a[0], &x[0], &y[0]);

    // workspace required for the SVD of the bidiagonal form
    rocsolver_bdsqr_getMemorySize<S>(k, nv, nu, 0, batch_count, &w[1]);

    // extra requirements for QR/LQ factorization
    if(compressed)
    {
        if(row)
            rocsolver_geqrf_getMemorySize<T, BATCHED>(m,n,batch_count, &unused, &w[2], &x[1], &y[1], &unused);
        else
            rocsolver_gelqf_getMemorySize<T, BATCHED>(m,n,batch_count, &unused, &w[2], &x[1], &y[1], &unused);
    }

    // extra requirements for orthonormal/unitary matrix generation
    // ormbr
    if(thinSVD && !fast_thinSVD && (othervS || othervA || !leadvO))
        rocsolver_ormbr_unmbr_getMemorySize<T, BATCHED>(storev_lead, side, m, n, k, batch_count, 
                                                        &unused, &a[1], &y[2], &x[2], &unused);
    // orgbr
    if(thinSVD)
    {
        if(!othervN)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(storev_other, k, k, k, batch_count, &unused, &w[3],
                                                &a[2], &x[3], &unused);

        if(fast_thinSVD)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(storev_lead, k, k, k, batch_count, &unused, &w[4],
                                                &a[3], &x[4], &unused);
        else if(othervN && leadvO)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(storev_lead, m, n, k, batch_count, &unused, &w[4],
                                                &a[3], &x[4], &unused);
    }
    else
    {
        mn = (row && leftvS) ? n : m;
        if(leftvS || leftvA)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_column_wise, m, mn, n, batch_count, 
                                                &unused, &w[3], &a[2], &x[3], &unused);
        else if(leftvO)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_column_wise, m, k, n, batch_count, 
                                                &unused, &w[3], &a[2], &x[3], &unused);

        mn = (!row && rightvS) ? m : n;
        if(rightvS || rightvA)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_row_wise, mn, n, m, batch_count, 
                                                &unused, &w[4], &a[3], &x[4], &unused);
        else if(rightvO)
            rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_row_wise, k, n, m, batch_count, 
                                                &unused, &w[4], &a[3], &x[4], &unused);
    }

    // orgqr/orglq
    if(thinSVD && !leadvN)
    {
        if(leadvA)
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<T, BATCHED>(kk,kk,k,batch_count,
                                                                &unused, &w[5], &a[4], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<T, BATCHED>(kk,kk,k,batch_count,
                                                                &unused, &w[5], &a[4], &x[5], &unused);
        }
        else
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<T, BATCHED>(m,n,k,batch_count,
                                                                &unused, &w[5], &a[4], &x[5], &unused);
            else
                rocsolver_orglq_unglq_getMemorySize<T, BATCHED>(m,n,k,batch_count,
                                                                &unused, &w[5], &a[4], &x[5], &unused);
        }
    }

    // get max sizes 
    *size_work_workArr = *std::max_element(w.data(),w.data()+6);
    *size_Abyx_norms_tmptr = *std::max_element(a.data(),a.data()+5);
    *size_Abyx_norms_trfact_X = *std::max_element(x.data(),x.data()+6);
    *size_diag_tmptr_Y = *std::max_element(y.data(),y.data()+3);    
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        TT* S,
                                        const rocblas_stride strideS,
                                        T* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        T* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        TT* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_tmptr,
                                        T* Abyx_norms_trfact_X,
                                        T* diag_tmptr_Y,
                                        T* tau,
                                        T* tempArrayT,
                                        T* tempArrayC,
                                        T** workArr)
{
    ROCSOLVER_ENTER("gesvd", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "ldu:", ldu, "ldv:", ldv, "mode:", fast_alg,
                    "bc:", batch_count);

    constexpr bool COMPLEX = is_complex<T>;

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
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    const bool fast_thinSVD = (fast_alg == rocblas_outofplace);
    const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;

    // auxiliary sizes and variables
    const rocblas_int k = min(m, n);
    const rocblas_int kk = max(m, n);
    const rocblas_int shiftX = 0;
    const rocblas_int shiftY = 0;
    const rocblas_int shiftUV = 0;
    const rocblas_int shiftT = 0;
    const rocblas_int shiftC = 0;
    const rocblas_int shiftU = 0;
    const rocblas_int shiftV = 0;
    const rocblas_int ldx = m;
    const rocblas_int ldy = n;
    const rocblas_stride strideX = m * GEBRD_GEBD2_SWITCHSIZE;
    const rocblas_stride strideY = n * GEBRD_GEBD2_SWITCHSIZE;
    T* bufferT = tempArrayT;
    rocblas_int ldt = k;
    rocblas_stride strideT = k * k;
    T* bufferC = tempArrayC;
    rocblas_int ldc = m;
    rocblas_stride strideC = m * n;
    
    T *UV;
    rocblas_int lduv, mn, nu, nv;
    rocblas_int offset_other, offset_lead;
    rocblas_storev storev_other, storev_lead;
    rocblas_stride strideUV;
    rocblas_fill uplo;
    rocblas_side side;
    rocblas_operation trans;

    nu = leftvN ? 0 : m;
    nv = rightvN ? 0 : n;
    if(row)
    {
        UV = U;
        lduv = ldu;
        strideUV = strideU;
        uplo = rocblas_fill_upper;
        storev_other = rocblas_row_wise;
        storev_lead = rocblas_column_wise;
        offset_other = k*batch_count;
        offset_lead = 0;
        if(!fast_thinSVD)
        {
            bufferT = V;
            ldt = ldv;
            strideT = strideV;
        } 
        if(othervS || othervA)
        {
            bufferC = V;
            ldc = ldv;
            strideC = strideV; 
        }
        side = rocblas_side_right;
        trans = rocblas_operation_none;
    }
    else
    {
        UV = V;
        lduv = ldv;
        strideUV = strideV;
        uplo = rocblas_fill_lower;
        storev_other = rocblas_column_wise;
        storev_lead = rocblas_row_wise;
        offset_other = 0;
        offset_lead = k*batch_count;
        if(!fast_thinSVD)
        {
            bufferT = U;
            ldt = ldu;
            strideT = strideU;
        } 
        if(othervS || othervA)
        {
            bufferC = U;
            ldc = ldu;
            strideC = strideU; 
        }
        side = rocblas_side_left;
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
    }

    // common block sizes and number of threads for internal kernels
    constexpr rocblas_int thread_count = 32;
    const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;


    // A thin SVD could be computed for matrices with sufficiently more rows than
    // columns (or columns than rows) by starting with a QR factorization (or LQ
    // factorization) and working with the triangular factor afterwards. When
    // computing a thin SVD, a fast algorithm could be executed by doing some
    // computations out-of-place.

    if(m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m) 
    /*******************************************/
    /********** CASE: CHOOSE THIN-SVD **********/
    /*******************************************/
    {
        if(leadvN)
        /***** SUB-CASE: USE THIN-SVD WITH NO LEAD-DIMENSION VECTORS *****/
        /*****************************************************************/
        {
            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED,STRIDED>(handle,m,n,A,shiftA,lda,strideA,tau,k,batch_count,
                                 scalars,work_workArr,Abyx_norms_trfact_X,
                                 diag_tmptr_Y,workArr,row);

//print_device_matrix<T>(rocblas_cout,"factorized A",m,n,A,lda);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            // N/A

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            hipLaunchKernelGGL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream,
                               k, k, A, shiftA, lda, strideA, uplo);

//print_device_matrix<T>(rocblas_cout,"clean triangular part",m,n,A,lda);
            
            rocsolver_gebrd_template<BATCHED, STRIDED>(
                handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE, tau, k,
                (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX, 
                diag_tmptr_Y, shiftY, ldy, strideY,
                batch_count, scalars, work_workArr, Abyx_norms_tmptr);

//print_device_matrix<T>(rocblas_cout,"diagonalized matrix",m,n,A,lda);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            if(!othervN)
                rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                    handle, storev_other, k, k, k, A, shiftA, lda, strideA, (tau + offset_other), k, 
                    batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);    

//print_device_matrix<T>(rocblas_cout,"right orthogonal matrix from diagonalization",m,n,A,lda);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                local_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, S, strideS, E, strideE, A, shiftA, lda,
                                    strideA, U, shiftU, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
            else
                local_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, S, strideS, E, strideE, V, shiftV, ldv,
                                    strideV, A, shiftA, lda, strideA, info, batch_count,
                                    (TT*)work_workArr, workArr);

//print_device_matrix<TT>(rocblas_cout,"singular values",1,k,S,1);
//print_device_matrix<T>(rocblas_cout,"updated right vectors",m,n,A,lda);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(othervS || othervA)
            {
                mn = row ? n : m;
                hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, mn, mn, A, shiftA, lda,
                               strideA, bufferC, shiftC, ldc, strideC);
            }

//print_device_matrix<T>(rocblas_cout,"final right vectors in V",n,n,V,ldv);
        }  

        else if(fast_thinSVD)
        /***** SUB-CASE: USE FAST (OUT-OF-PLACE) THIN-SVD ALGORITHM *****/
        /****************************************************************/
        {
//printf("\nhola mundo en fast thin SVD..."); fflush(stdout);
            nu = leftvN ? 0 : k;
            nv = rightvN ? 0 : k;

//print_device_matrix<T>(rocblas_cout,"original A",m,n,A,lda);
            
            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED,STRIDED>(handle,m,n,A,shiftA,lda,strideA,tau,k,batch_count,
                                 scalars,work_workArr,Abyx_norms_trfact_X,
                                 diag_tmptr_Y,workArr,row);

            if(leadvA) 
                // copy factorization to U or V when needed
                hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, n, A, shiftA, lda,
                               strideA, UV, shiftUV, lduv, strideUV);

            // copy the triangular part to be used in the bidiagonalization
            hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA, lda,
                               strideA, bufferT, shiftT, ldt, strideT, uplo);

//print_device_matrix<T>(rocblas_cout,"factorized A",m,n,A,lda);
//print_device_matrix<T>(rocblas_cout,"copied triangular part",k,k,bufferT,ldt);
//hipDeviceSynchronize();
//printf("\ncompresion lista..."); fflush(stdout);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            if(leadvA)
                local_orgqrlq_ungqrlq_template<false, STRIDED>(
                    handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau, k, batch_count,
                    scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr, row);
            else
                local_orgqrlq_ungqrlq_template<BATCHED, STRIDED>(
                    handle, m, n, k, A, shiftA, lda, strideA, tau, k, batch_count,
                    scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr, row);

//print_device_matrix<T>(rocblas_cout,"orthogonal matrix from factorization",m,n,A,lda);
//hipDeviceSynchronize();
//printf("\ngenere matriz ortogonal de la compresion..."); fflush(stdout);

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            hipLaunchKernelGGL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream,
                               k, k, bufferT, shiftT, ldt, strideT, uplo);

//print_device_matrix<T>(rocblas_cout,"triangular part ready for diagonalization",k,k,bufferT,ldt);

            rocsolver_gebrd_template<false, STRIDED>(
                handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS, E, strideE, tau, k,
                (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                diag_tmptr_Y, shiftY, ldy, strideY,
                batch_count, scalars, work_workArr, Abyx_norms_tmptr);

//print_device_matrix<T>(rocblas_cout,"triangular part diagonalized",k,k,bufferT,ldt);

            if(!othervN)
                // copy results to generate non-lead vectors if required
                hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, k, bufferT, 
                               shiftT, ldt, strideT, bufferC, shiftC, ldc, strideC);

//print_device_matrix<T>(rocblas_cout,"diagonalization copied into V",n,n,V,ldv);
//hipDeviceSynchronize();
//printf("\nbidiagonalization lista..."); fflush(stdout);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                    handle, storev_lead, k, k, k, bufferT, shiftT, ldt, strideT, (tau + offset_lead), k, 
                    batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);    

            // for the other-side vectors
            if(!othervN)
                rocsolver_orgbr_ungbr_template<false, STRIDED>(
                        handle, storev_other, k, k, k, bufferC, shiftC, ldc, strideC, (tau + offset_other), k, 
                        batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);    

//print_device_matrix<T>(rocblas_cout,"left orthogonal matrix from diagonalization",k,k,bufferT,ldt);
//print_device_matrix<T>(rocblas_cout,"right orthogonal matrix from diagonalization",n,n,V,ldv);
//hipDeviceSynchronize();
//printf("\ngenere matriz ortogonal de la bidiagonalization..."); fflush(stdout);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                local_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, S, strideS, E, strideE, bufferC, shiftC, ldc,
                                    strideC, bufferT, shiftT, ldt, strideT, info, batch_count,
                                    (TT*)work_workArr, workArr);
            else
                local_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, S, strideS, E, strideE, bufferT, shiftT, ldt,
                                    strideT, bufferC, shiftC, ldc, strideC, info, batch_count,
                                    (TT*)work_workArr, workArr);

//print_device_matrix<T>(rocblas_cout,"left orthogonal matrix from diagonalization with bdsqr",k,k,bufferT,ldt);
//print_device_matrix<T>(rocblas_cout,"right orthogonal matrix from diagonalization with bdsqr",n,n,V,ldv);
//print_device_matrix<TT>(rocblas_cout,"singular values",1,k,S,1);
//hipDeviceSynchronize();
//printf("\nSVD lista..."); fflush(stdout);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(leadvO)
            {
                bufferC = tempArrayC;
                ldc = m;
                strideC = m * n;
                
                // update
                if(row)
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,A,shiftA,lda,strideA,bufferT,shiftT,ldt,strideT,
                                    &zero,bufferC,shiftC,ldc,strideC,batch_count,workArr);    
                else    
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,bufferT,shiftT,ldt,strideT,A,shiftA,lda,strideA,
                                    &zero,bufferC,shiftC,ldc,strideC,batch_count,workArr);    
                
                // copy to overwrite A
                hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, n, bufferC,
                               shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else if(leadvS)
            {
                // update
                if(row)
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,A,shiftA,lda,strideA,bufferT,shiftT,ldt,strideT,
                                    &zero,UV,shiftUV,lduv,strideUV,batch_count,workArr);
                else
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,bufferT,shiftT,ldt,strideT,A,shiftA,lda,strideA,
                                    &zero,UV,shiftUV,lduv,strideUV,batch_count,workArr);

                // overwrite A if required
                if(othervO)
                    hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, k, bufferC,
                               shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else
            {
                // update
                if(row)
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,UV,shiftUV,lduv,strideUV,bufferT,shiftT,ldt,strideT,
                                    &zero,A,shiftA,lda,strideA,batch_count,workArr);
                else
                    rocblasCall_gemm<BATCHED,STRIDED>(handle,rocblas_operation_none,rocblas_operation_none,
                                    m,n,k,&one,bufferT,shiftT,ldt,strideT,UV,shiftUV,lduv,strideUV,
                                    &zero,A,shiftA,lda,strideA,batch_count,workArr);

                // copy back to U/V
                hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                               shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

                // overwrite A if required
                if(othervO)
                    hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, k, bufferC,
                               shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }

//print_device_matrix<T>(rocblas_cout,"final left vectors (in A)",m,n,A,lda);
//print_device_matrix<T>(rocblas_cout,"final right vectors (in V)",n,n,V,ldv);
//hipDeviceSynchronize();
//printf("\nupdate de vectores liston..."); fflush(stdout);
        }

        else
        /************ SUB-CASE: USE IN-PLACE THIN-SVD ALGORITHM *******/
        /**************************************************************/
        {
            //*** STAGE 1: Row (or column) compression ***//
            if(!leadvO || !othervN)
            {
                local_geqrlq_template<BATCHED,STRIDED>(handle,m,n,A,shiftA,lda,strideA,tau,k,batch_count,
                                     scalars,work_workArr,Abyx_norms_trfact_X,
                                     diag_tmptr_Y,workArr,row);
                
                if(!leadvO) 
                    // copy factorization to U or V when needed
                    hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, n, A, shiftA, lda,
                               strideA, UV, shiftUV, lduv, strideUV);
                    
                if(othervS || othervA)
                    // copy the triangular part 
                    hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA, lda,
                               strideA, bufferT, shiftT, ldt, strideT, uplo);

                //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
                if(leadvO)
                    local_orgqrlq_ungqrlq_template<BATCHED, STRIDED>(
                        handle, m, n, k, A, shiftA, lda, strideA, tau, k, batch_count,
                        scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr, row);
                else if(leadvA)
                    local_orgqrlq_ungqrlq_template<false, STRIDED>(
                        handle, kk, kk, k, UV, shiftUV, lduv, strideUV, tau, k, batch_count,
                        scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr, row);
                else
                    local_orgqrlq_ungqrlq_template<false, STRIDED>(
                        handle, m, n, k, UV, shiftUV, lduv, strideUV, tau, k, batch_count,
                        scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr, row);
            }
            
            //*** STAGE 3: Bidiagonalization ***//
            if(othervS || othervA) 
            {
                // clean triangular factor
                hipLaunchKernelGGL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream,
                               k, k, bufferT, shiftT, ldt, strideT, uplo);

                rocsolver_gebrd_template<false, STRIDED>(
                    handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS, E, strideE, tau, k,
                    (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                    diag_tmptr_Y, shiftY, ldy, strideY,
                    batch_count, scalars, work_workArr, Abyx_norms_tmptr);

                uplo = rocblas_fill_upper;
            }
            else if(!leadvO)
            {
                // clean triangular factor
                hipLaunchKernelGGL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream,
                               k, k, A, shiftA, lda, strideA, uplo);

                rocsolver_gebrd_template<BATCHED, STRIDED>(
                    handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE, tau, k,
                    (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                    diag_tmptr_Y, shiftY, ldy, strideY,
                    batch_count, scalars, work_workArr, Abyx_norms_tmptr); 

                uplo = rocblas_fill_upper;
            }
            else
                rocsolver_gebrd_template<BATCHED, STRIDED>(
                    handle, m, n, A, shiftA, lda, strideA, S, strideS, E, strideE, tau, k,
                    (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
                    diag_tmptr_Y, shiftY, ldy, strideY,
                    batch_count, scalars, work_workArr, Abyx_norms_tmptr);

//print_device_matrix<T>(rocblas_cout,"diagonalized A",m,n,A,lda);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            if(othervS || othervA)
            {
                if(leadvO)
                    local_ormbr_unmbr_template<BATCHED,STRIDED>(handle,storev_lead,side,trans,
                                        m,n,k,bufferT,shiftT,ldt,strideT,(tau+offset_lead),k,
                                        A,shiftA,lda,strideA,batch_count,
                                        scalars, Abyx_norms_tmptr, diag_tmptr_Y, Abyx_norms_trfact_X,workArr);
                else   
                    local_ormbr_unmbr_template<false,STRIDED>(handle,storev_lead,side,trans,
                                        m,n,k,bufferT,shiftT,ldt,strideT,(tau+offset_lead),k,
                                        UV,shiftUV,lduv,strideUV,batch_count,
                                        scalars, Abyx_norms_tmptr, diag_tmptr_Y, Abyx_norms_trfact_X,workArr);
            }
            else if(!leadvO)
                local_ormbr_unmbr_template<BATCHED,STRIDED>(handle,storev_lead,side,trans,
                                        m,n,k,A,shiftA,lda,strideA,(tau+offset_lead),k,
                                        UV,shiftUV,lduv,strideUV,batch_count,
                                        scalars, Abyx_norms_tmptr, diag_tmptr_Y, Abyx_norms_trfact_X,workArr);               
            else
                rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                        handle, storev_lead, m, n, k, A, shiftA, lda, strideA, (tau + offset_lead), k, batch_count,
                        scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);

//print_device_matrix<T>(rocblas_cout,"orthogonal matrix from diagonalization",m,n,A,lda);

            // for the other-side vectors
            if(othervS || othervA)
                rocsolver_orgbr_ungbr_template<false, STRIDED>(
                        handle, storev_other, k, k, k, bufferT, shiftT, ldt, strideT, (tau + offset_other), k, 
                        batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);    
            else if(othervO)
                rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                        handle, storev_other, k, k, k, A, shiftA, lda, strideA, (tau + offset_other), k, 
                        batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);    

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(!leftvO && !rightvO)
            {
                local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, V, shiftV, ldv,
                                    strideV, U, shiftU, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
            }
            else if(leftvO && !rightvO)
            {
                local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, V, shiftV, ldv,
                                    strideV, A, shiftA, lda, strideA, info, batch_count,
                                    (TT*)work_workArr, workArr);
            }
            else
            {
                local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, A, shiftA,
                                    lda, strideA, U, shiftU, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
            }

//print_device_matrix<TT>(rocblas_cout,"singular values",1,k,S,1);
//print_device_matrix<T>(rocblas_cout,"final left vectors",m,n,A,lda);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            // N/A
        }
    }

    else
    /*********************************************/
    /********** CASE: CHOOSE NORMAL SVD **********/
    /*********************************************/
    { 
        //*** STAGE 1: Row (or column) compression ***//
        // N/A

        //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
        // N/A

        //*** STAGE 3: Bidiagonalization ***//
        rocsolver_gebrd_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, S, strideS, E, strideE, tau, k,
            (tau + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX, 
            diag_tmptr_Y, shiftY, ldy, strideY,
            batch_count, scalars, work_workArr, Abyx_norms_tmptr);

        //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
        if(leftvS || leftvA)
        {
            // copy data to matrix U where orthogonal matrix will be generated
            mn = (row && leftvS) ? n : m;
            hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_m, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, k, A, shiftA, lda,
                               strideA, U, shiftU, ldu, strideU);
            
            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_column_wise, m, mn, n, U, shiftU, ldu, strideU, tau, k, batch_count,
                scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);
        }

        if(rightvS || rightvA)
        {
            // copy data to matrix V where othogonal matrix will be generated
            mn = (!row && rightvS) ? m : n;
            hipLaunchKernelGGL(copy_mat<T>, dim3(blocks_k, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, n, A, shiftA, lda,
                               strideA, V, shiftV, ldv, strideV);

            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_row_wise, mn, n, m, V, shiftV, ldv, strideV, (tau + k * batch_count), k,
                batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);
        }

        if(leftvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_column_wise, m, k, n, A, shiftA, lda, strideA, tau, k, batch_count,
                scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);
        }

        if(rightvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_row_wise, k, n, m, A, shiftA, lda, strideA, (tau + k * batch_count),
                k, batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, Abyx_norms_trfact_X, workArr);
        }

        //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
        if(!leftvO && !rightvO)
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, V, shiftV, ldv,
                                    strideV, U, shiftU, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }

        else if(leftvO && !rightvO)
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, V, shiftV, ldv,
                                    strideV, A, shiftA, lda, strideA, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }

        else
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, S, strideS, E, strideE, A, shiftA,
                                    lda, strideA, U, shiftU, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }
            
        //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
        // N/A
    }

    rocblas_set_pointer_mode(handle, old_mode);
//printf("\nadios mundo..."); fflush(stdout);
    return rocblas_status_success;
}
