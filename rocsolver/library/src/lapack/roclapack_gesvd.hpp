/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GESVD_H
#define ROCLAPACK_GESVD_H

#include "../auxiliary/rocauxiliary_bdsqr.hpp"
#include "../auxiliary/rocauxiliary_orgbr_ungbr.hpp"
#include "common_device.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "rocsolver.h"

/** COPY_ARRAY copies the m-by-n array A into B **/
template <typename T, typename U1, typename U2>
__global__ void copy_array(const rocblas_int m,
                           const rocblas_int n,
                           U1 A,
                           const rocblas_int shiftA,
                           const rocblas_int lda,
                           const rocblas_stride strideA,
                           U2 B,
                           const rocblas_int shiftB,
                           const rocblas_int ldb,
                           const rocblas_stride strideB)
{
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * blocksizey + hipThreadIdx_y;

    if(i < m && j < n)
    {
        T *Ap, *Bp;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
        Bp = load_ptr_batch<T>(B, b, shiftB, strideB);

        Bp[i + j * ldb] = Ap[i + j * lda];
    }
}

/** wrapper to BDSQR_TEMPLATE **/
template <typename T, typename TT>
void local_bdsqr_template(rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          const rocblas_int nc,
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
    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V, shiftV, ldv,
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
                          const rocblas_int nc,
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

    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V, shiftV, ldv,
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
                          const rocblas_int nc,
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

    rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE,
                                (T* const*)workArr, shiftV, ldv, strideV, U, shiftU, ldu, strideU,
                                (T* const*)nullptr, 0, 1, 1, info, batch_count, work);
}

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(const rocblas_svect left_svect,
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
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_tmptr,
                                   size_t* size_X_trfact,
                                   size_t* size_Y,
                                   size_t* size_tau,
                                   size_t* size_workArr)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_tmptr = 0;
        *size_X_trfact = 0;
        *size_Y = 0;
        *size_tau = 0;
        *size_workArr = 0;
        return;
    }

    // booleans used to determine the path that the execution will follow:
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);

    size_t w, s, t, unused;
    rocblas_int k = min(m, n);
    rocblas_int nu = leftvN ? 0 : m;
    rocblas_int nv = rightvN ? 0 : n;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // workspace required for the bidiagonalization
    rocsolver_gebrd_getMemorySize<T, BATCHED>(m, n, batch_count, size_scalars, size_work_workArr,
                                              size_Abyx_norms_tmptr, size_X_trfact, size_Y);

    // worksapce required for the SVD of the bidiagonal form
    rocsolver_bdsqr_getMemorySize<S>(k, nv, nu, 0, batch_count, &w);
    if(w > *size_work_workArr)
        *size_work_workArr = w;

    // workspace required to compute the left singular vectors
    if(!leftvN)
    {
        if(m >= n && (leftvS || leftvO))
            k = n;
        else
            k = m;
        rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_column_wise, m, k, n, batch_count,
                                                        &unused, &w, &s, &t, &unused);
        if(w > *size_work_workArr)
            *size_work_workArr = w;
        if(s > *size_Abyx_norms_tmptr)
            *size_Abyx_norms_tmptr = s;
        if(t > *size_X_trfact)
            *size_X_trfact = t;
    }

    // workspace required to compute the right singular vectors
    if(!rightvN)
    {
        if(n > m && (rightvS || rightvO))
            k = m;
        else
            k = n;
        rocsolver_orgbr_ungbr_getMemorySize<T, BATCHED>(rocblas_row_wise, k, n, m, batch_count,
                                                        &unused, &w, &s, &t, &unused);

        if(w > *size_work_workArr)
            *size_work_workArr = w;
        if(s > *size_Abyx_norms_tmptr)
            *size_Abyx_norms_tmptr = s;
        if(t > *size_X_trfact)
            *size_X_trfact = t;
    }

    // size of array tau to store householder scalars on intermediate
    // orthonormal/unitary matrices
    *size_tau = 2 * sizeof(T) * min(m, n) * batch_count;
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
                                        T* X_trfact,
                                        T* Y,
                                        T* tau,
                                        T** workArr)
{
    constexpr bool COMPLEX = is_complex<T>;

    // quick return
    if(n == 0 || m == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // booleans used to determine the path that the execution will follow:
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    const bool fast_thinSVD = (fast_alg == rocblas_outofplace);

    rocblas_int mn, nu, nv;
    rocblas_fill uplo;
    const rocblas_int k = min(m, n);
    rocblas_stride strideX = m * GEBRD_GEBD2_SWITCHSIZE;
    rocblas_stride strideY = n * GEBRD_GEBD2_SWITCHSIZE;
    rocblas_int shiftX = 0;
    rocblas_int shiftY = 0;
    rocblas_int ldx = m;
    rocblas_int ldy = n;

    // common block sizes and number of threads for internal kernels
    constexpr rocblas_int thread_count = 32;
    const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;

    // A thin SVD could be computed for matrices with sufficiently more rows than
    // columns (or columns that rows) by starting with a QR factorization (or LQ
    // factorization) and working with the triangular factor afterwards. When
    // computing a thin SVD, a fast algorithm could be executed by doing some
    // computations out-of-place.

    // choose Thin-SVD
    if(m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m)
    {
        // (TODO: IMPLEMENT THIN_SVD AND FAST THIN_SVD ALGORITHMS)

        // use fast thin-svd algorithm (this may require larger memory worksapce)
        if(fast_thinSVD)
        {
            return rocblas_status_not_implemented;
        }

        // use normal thin-svd
        else
        { //(!fast_thinSVD)
            return rocblas_status_not_implemented;
        }
    }

    // choose normal SVD
    else
    { // (m < THIN_SVD_SWITCH*n && n < THIN_SVD_SWITCH*m)

        // 1. Bidiagonalize A.
        rocsolver_gebrd_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, S, strideS, E, strideE, tau, k,
            (tau + k * batch_count), k, X_trfact, shiftX, ldx, strideX, Y, shiftY, ldy, strideY,
            batch_count, scalars, work_workArr, Abyx_norms_tmptr);

        // 2. Generate corresponding orthonormal/unitary matrices when required
        if(leftvS || leftvA)
        {
            mn = (m >= n && leftvS) ? n : m;
            hipLaunchKernelGGL(copy_array<T>, dim3(blocks_m, blocks_k, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, m, k, A, shiftA, lda,
                               strideA, U, 0, ldu, strideU);
            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_column_wise, m, mn, n, U, 0, ldu, strideU, tau, k, batch_count,
                scalars, (T*)work_workArr, Abyx_norms_tmptr, X_trfact, workArr);
        }

        if(rightvS || rightvA)
        {
            mn = (n > m && rightvS) ? m : n;
            hipLaunchKernelGGL(copy_array<T>, dim3(blocks_k, blocks_n, batch_count),
                               dim3(thread_count, thread_count, 1), 0, stream, k, n, A, shiftA, lda,
                               strideA, V, 0, ldv, strideV);
            rocsolver_orgbr_ungbr_template<false, STRIDED>(
                handle, rocblas_row_wise, mn, n, m, V, 0, ldv, strideV, (tau + k * batch_count), k,
                batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, X_trfact, workArr);
        }

        if(leftvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_column_wise, m, k, n, A, shiftA, lda, strideA, tau, k, batch_count,
                scalars, (T*)work_workArr, Abyx_norms_tmptr, X_trfact, workArr);
        }

        if(rightvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
                handle, rocblas_row_wise, k, n, m, A, shiftA, lda, strideA, (tau + k * batch_count),
                k, batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr, X_trfact, workArr);
        }

        // 3. compute singular values (and vectors if required) using the
        // bidiagonal form
        uplo = (m >= n) ? rocblas_fill_upper : rocblas_fill_lower;
        nu = leftvN ? 0 : m;
        nv = rightvN ? 0 : n;

        if(!leftvO && !rightvO)
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, 0, ldv,
                                    strideV, U, 0, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }

        else if(leftvO && !rightvO)
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V, 0, ldv,
                                    strideV, A, shiftA, lda, strideA, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }

        else
        {
            local_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A, shiftA,
                                    lda, strideA, U, 0, ldu, strideU, info, batch_count,
                                    (TT*)work_workArr, workArr);
        }
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GESVD_H */
