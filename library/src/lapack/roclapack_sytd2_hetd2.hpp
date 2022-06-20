/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/** set_tau kernel copies to tau the corresponding Householder scalars **/
template <typename T>
ROCSOLVER_KERNEL void
    set_tau(const rocblas_int batch_count, T* tmptau, T* tau, const rocblas_stride strideP)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* t = tau + b * strideP;

        t[0] = tmptau[b];
    }
}

/** set_tridiag kernel copies results to set tridiagonal form in A, diagonal elements in D
    and off-diagonal elements in E **/
template <typename T, typename S, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D,
                                  const rocblas_stride strideD,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    bool lower = (uplo == rocblas_fill_lower);

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        d[i] = a[i + i * lda];

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}

template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D,
                                  const rocblas_stride strideD,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    bool lower = (uplo == rocblas_fill_lower);

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        d[i] = a[i + i * lda].real();

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}

template <bool BATCHED, typename T>
void rocsolver_sytd2_hetd2_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_tmptau = 0;
        *size_workArr = 0;
        return;
    }

    size_t w_temp;

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of array to store temporary householder scalars
    *size_tmptau = sizeof(T) * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements to call LARFG
    rocsolver_larfg_getMemorySize<T>(n, batch_count, size_work, size_norms);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w_temp);
    *size_work = std::max(*size_work, w_temp);
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tau,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !D) || (n && !E) || (n && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_sytd2_hetd2_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              T* tau,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* norms,
                                              T* tmptau,
                                              T** workArr)
{
    ROCSOLVER_ENTER("sytd2_hetd2", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // configure kernels
    rocblas_int blocks = (n - 1) / BS1 + 1;
    dim3 grid_n(blocks, batch_count);
    dim3 threads(BS1, 1, 1);
    blocks = (batch_count - 1) / BS1 + 1;
    dim3 grid_b(blocks, 1);

    rocblas_stride stridet = 1; //stride for tmptau

    if(uplo == rocblas_fill_lower)
    {
        // reduce the lower part of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < n - 1; ++j)
        {
            // 1. generate Householder reflector to annihilate A(j+2:n-1,j)
            rocsolver_larfg_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j + 1, j, lda), A,
                                        shiftA + idx2D(min(j + 2, n - 1), j, lda), 1, strideA,
                                        tmptau, stridet, batch_count, work, norms);

            // 2. copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, E + j, strideE);

            // 3. overwrite tau with w = tmptau*A*v - 1/2*tmptau*(tmptau*v'*A*v)*v
            // a. compute tmptau*A*v -> tau
            rocblasCall_symv_hemv<T>(handle, uplo, n - 1 - j, tmptau, stridet, A,
                                     shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                                     shiftA + idx2D(j + 1, j, lda), 1, strideA, scalars + 1, 0, tau,
                                     j, 1, strideP, batch_count, work, workArr);

            // b. compute scalar tmptau*v'*A*v=tau'*v -> norms
            rocblasCall_dot<COMPLEX, T>(handle, n - 1 - j, tau, j, 1, strideP, A,
                                        shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count,
                                        norms, work, workArr);

            // c. finally update tau as an axpy: -1/2*tmptau*norms*v + tau -> tau
            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, n - 1 - j, norms,
                                    tmptau, stridet, A, shiftA + idx2D(j + 1, j, lda), strideA, tau,
                                    j, strideP);

            // 4. apply the Householder reflector to A as a rank-2 update:
            // A = A - v*w' - w*v'
            rocblasCall_syr2_her2<T>(handle, uplo, n - 1 - j, scalars, A,
                                     shiftA + idx2D(j + 1, j, lda), 1, strideA, tau, j, 1, strideP,
                                     A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                     batch_count, workArr);

            // 5. Save the used housedholder scalar
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                    tau + j, strideP);
        }
    }

    else
    {
        // reduce the upper part of A
        // main loop running backwards (for each column)
        for(rocblas_int j = n - 1; j > 0; --j)
        {
            // 1. generate Householder reflector to annihilate A(0:j-2,j)
            rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, tmptau, 1,
                                        batch_count, work, norms);

            // 2. copy to E(j-1) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j - 1, j, lda), strideA, E + j - 1, strideE);

            // 3. overwrite tau with w = tmptau*A*v - 1/2*tmptau*tmptau*(v'*A*v*)v
            // a. compute tmptau*A*v -> tau
            rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, stridet, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                     1, strideP, batch_count, work, workArr);

            // b. compute scalar tmptau*v'*A*v=tau'*v -> norms
            rocblasCall_dot<COMPLEX, T>(handle, j, tau, 0, 1, strideP, A, shiftA + idx2D(0, j, lda),
                                        1, strideA, batch_count, norms, work, workArr);

            // c. finally update tau as an axpy: -1/2*tmptau*norms*v + tau -> tau
            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, j, norms, tmptau,
                                    stridet, A, shiftA + idx2D(0, j, lda), strideA, tau, 0, strideP);

            // 4. apply the Householder reflector to A as a rank-2 update:
            // A = A - v*w' - w*v'
            rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                     strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                     batch_count, workArr);

            // 5. Save the used housedholder scalar
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                    tau + j - 1, strideP);
        }
    }

    // Copy results (set tridiagonal form in A)
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid_n, threads, 0, stream, uplo, n, A, shiftA, lda,
                            strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
