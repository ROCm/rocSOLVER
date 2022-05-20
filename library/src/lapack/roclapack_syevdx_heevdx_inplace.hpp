/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_ormtr_unmtr.hpp"
#include "auxiliary/rocauxiliary_stebz.hpp"
#include "auxiliary/rocauxiliary_stein.hpp"
#include "rocblas.hpp"
#include "roclapack_syevx_heevx.hpp"
#include "roclapack_sytrd_hetrd.hpp"
#include "rocsolver/rocsolver.h"

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevdx_heevdx_inplace_argCheck(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_erange erange,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        T A,
                                                        const rocblas_int lda,
                                                        const S vl,
                                                        const S vu,
                                                        const rocblas_int il,
                                                        const rocblas_int iu,
                                                        rocblas_int* h_nev,
                                                        S* W,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_original && evect != rocblas_evect_none)
        return rocblas_status_invalid_value;
    if(erange != rocblas_erange_all && erange != rocblas_erange_value
       && erange != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !h_nev) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevdx_heevdx_inplace_getMemorySize(const rocblas_evect evect,
                                                   const rocblas_fill uplo,
                                                   const rocblas_int n,
                                                   const rocblas_int batch_count,
                                                   size_t* size_scalars,
                                                   size_t* size_work1,
                                                   size_t* size_work2,
                                                   size_t* size_work3,
                                                   size_t* size_work4,
                                                   size_t* size_work5,
                                                   size_t* size_work6,
                                                   size_t* size_D,
                                                   size_t* size_E,
                                                   size_t* size_iblock,
                                                   size_t* size_isplit,
                                                   size_t* size_tau,
                                                   size_t* size_nev,
                                                   size_t* size_nsplit_workArr)
{
    // if quick return, set workspace to zero
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_work5 = 0;
        *size_work6 = 0;
        *size_D = 0;
        *size_E = 0;
        *size_iblock = 0;
        *size_isplit = 0;
        *size_tau = 0;
        *size_nev = 0;
        *size_nsplit_workArr = 0;
        return;
    }

    size_t unused;
    size_t a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    size_t c1 = 0, c2 = 0, c3 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &a1, &b1, &c1,
                                                    size_nsplit_workArr);

    // extra requirements for computing the eigenvalues (stebz)
    rocsolver_stebz_getMemorySize<T>(n, batch_count, &a2, &b2, &c2, size_work4, size_work5,
                                     size_work6);

    if(evect == rocblas_evect_original)
    {
        // extra requirements for ormtr/unmtr
        rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                        &unused, &a3, &b3, &c3, &unused);

        // extra requirements for computing the eigenvectors (stein)
        rocsolver_stein_getMemorySize<T, S>(n, batch_count, &a4, &b4);

        // extra space to store A
        *size_work4 = max(*size_work4, sizeof(T) * n * n * batch_count);
    }

    // get max values
    *size_work1 = std::max({a1, a2, a3, a4});
    *size_work2 = std::max({b1, b2, b3, b4});
    *size_work3 = std::max({c1, c2, c3});

    // size of arrays for temporary tridiagonal elements
    *size_D = sizeof(S) * n * batch_count;
    *size_E = sizeof(S) * n * batch_count;

    // size of arrays for temporary submatrix indices
    *size_iblock = sizeof(rocblas_int) * n * batch_count;
    *size_isplit = sizeof(rocblas_int) * n * batch_count;

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;

    // size of array for number of eigenvalues on the device
    *size_nev = sizeof(rocblas_int) * batch_count;

    // size of array for temporary split off block sizes
    *size_nsplit_workArr = max(*size_nsplit_workArr, sizeof(rocblas_int) * batch_count);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevdx_heevdx_inplace_template(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_erange erange,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        U A,
                                                        const rocblas_int shiftA,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        const S vl,
                                                        const S vu,
                                                        const rocblas_int il,
                                                        const rocblas_int iu,
                                                        const S abstol,
                                                        rocblas_int* h_nev,
                                                        S* W,
                                                        const rocblas_stride strideW,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count,
                                                        T* scalars,
                                                        void* work1,
                                                        void* work2,
                                                        void* work3,
                                                        void* work4,
                                                        void* work5,
                                                        void* work6,
                                                        S* D,
                                                        S* E,
                                                        rocblas_int* iblock,
                                                        rocblas_int* isplit,
                                                        T* tau,
                                                        rocblas_int* d_nev,
                                                        void* nsplit_workArr)
{
    ROCSOLVER_ENTER("syevdx_heevdx_inplace", "evect:", evect, "erange:", erange, "uplo:", uplo,
                    "n:", n, "shiftA:", shiftA, "lda:", lda, "vl:", vl, "vu:", vu, "il:", il,
                    "iu:", iu, "abstol:", abstol, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return with info = 0 and nev = 0
    if(n == 0)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

        memset(h_nev, 0, sizeof(rocblas_int) * batch_count);
        return rocblas_status_success;
    }

    // TODO: Scale the matrix

    const rocblas_stride stride = n;

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED, T>(handle, uplo, n, A, shiftA, lda, strideA, D, stride,
                                               E, stride, tau, stride, batch_count, scalars,
                                               (T*)work1, (T*)work2, (T*)work3, (T**)nsplit_workArr);

    // compute eigenvalues
    rocblas_eorder eorder
        = (evect == rocblas_evect_none ? rocblas_eorder_entire : rocblas_eorder_blocks);
    rocsolver_stebz_template<S>(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, 0, stride, E,
                                0, stride, d_nev, (rocblas_int*)nsplit_workArr, W, strideW, iblock,
                                stride, isplit, stride, info, batch_count, (rocblas_int*)work1,
                                (S*)work2, (S*)work3, (S*)work4, (S*)work5, (rocblas_int*)work6);

    if(evect != rocblas_evect_none)
    {
        // kernel dimensions
        rocblas_int blocks1 = (n - 1) / BS1 + 1;
        rocblas_int blocks2 = (n - 1) / BS2 + 1;
        dim3 grid1(1, batch_count, 1);
        dim3 grid2(blocks2, blocks2, batch_count);
        dim3 threads1(BS1, 1, 1);
        dim3 threads2(BS2, BS2, 1);

        // copy A to work4
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, grid2, threads2, 0, stream, copymat_to_buffer, n, n, A,
                                shiftA, lda, strideA, (T*)work4);

        // compute eigenvectors
        rocsolver_stein_template<T>(handle, n, D, 0, stride, E, 0, stride, d_nev, W, 0, strideW,
                                    iblock, stride, isplit, stride, A, shiftA, lda, strideA,
                                    (rocblas_int*)nullptr, 0, info, batch_count, (S*)work1,
                                    (rocblas_int*)work2);

        // apply unitary matrix to eigenvectors
        rocblas_int temp_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
        rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
            handle, rocblas_side_left, uplo, rocblas_operation_none, n, temp_nev, (T*)work4, 0, n,
            n * n, tau, stride, A, shiftA, lda, strideA, batch_count, scalars, (T*)work1, (T*)work2,
            (T*)work3, (T**)nsplit_workArr);

        // sort eigenvalues and eigenvectors
        ROCSOLVER_LAUNCH_KERNEL(syevx_sort_eigs<T>, grid1, threads1, 0, stream, n, d_nev, W,
                                strideW, A, shiftA, lda, strideA, (rocblas_int*)nullptr, 0, info);
    }

    // copy nev from device to host
    if(h_nev)
    {
        hipMemcpyAsync(h_nev, d_nev, sizeof(rocblas_int) * batch_count, hipMemcpyDeviceToHost,
                       stream);
        hipStreamSynchronize(stream);
    }

    return rocblas_status_success;
}
