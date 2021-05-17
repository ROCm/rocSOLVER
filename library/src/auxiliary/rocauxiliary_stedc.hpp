/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

/****************************************************************************
  TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH). MORE PARALLELISM CAN BE INTRODUCED IN THE FUTURE IN AT LEAST TWO
  WAYS:
  1. the split diagonal blocks can be worked in parallel as they are
  independent
  2. for each block, multiple threads can accelerate some of the reductions
  and vector operations of the DC algorithm (defaltion, solve and merge processes).
***************************************************************************/

/** STEDC_KERNEL implements the main loop of the DC algorithm
    to compute the eigenvalues/eigenvectors of a symmetric tridiagonal
    matrix given by D and E **/
template <typename T, typename S, typename U>
__global__ void stedc_kernel(const rocblas_int n,
                             S* DD,
                             const rocblas_stride strideD,
                             S* EE,
                             const rocblas_stride strideE,
                             U CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             rocblas_int* iinfo,
                             S* WW,
                             const S eps,
                             const S ssfmin,
                             const S ssfmax)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T* C;
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* info = iinfo + bid;

    rocblas_int k = 0; //position where the next independent block starts
    S tol; //tolerance. If an element of E is <= tol we have an independent block
    rocblas_int bs; //size of an independent block

    // main loop
    while(k < n)
    {
        // Split next independent block
        bs = 1;
        for(rocblas_int j = k; j < n - 1; ++j)
        {
            tol = eps * sqrt(abs(D[j])) * sqrt(abs(D[j + 1]));
            if(abs(E[j]) < tol)
                break;
            bs++;
        }

        // if block is too small, solve it with steqr
        if(true) //(TODO: should be if(bs <= STEDC_MIN_DC_SIZE) once DC is implemented)
        {
            S* W = WW + bid * (2 * bs - 2);
            run_steqr(bs, D + k, E + k, C + k + k * ldc, ldc, info, W, 30 * bs, eps, ssfmin, ssfmax);
        }

        else
        {
            // TODO: here goes the implementation of DC algorithm to work with large independent blocks.
        }

        k += bs;
    }
}

/** This local gemm adapts rocblas_gemm to multiply complex*real, and
    overwrite result: A = A*B **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // temp = A*B
    rocblasCall_gemm<BATCHED, STRIDED, T>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, A, shiftA, lda,
        strideA, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // A = temp
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / 32 + 1;
    hipLaunchKernelGGL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0, stream, n,
                       n, A, shiftA, lda, strideA, temp, shiftT, ldt, strideT, rocblas_fill_full,
                       copymat_from_buffer);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, std::enable_if_t<is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A -> work; work*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // work = real(A)
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / 32 + 1;
    hipLaunchKernelGGL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                       stream, n, n, A, shiftA, lda, strideA, work, shiftT, ldt, strideT,
                       rocblas_fill_full, copymat_to_buffer);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // real(A) = temp
    hipLaunchKernelGGL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                       stream, n, n, A, shiftA, lda, strideA, temp, shiftT, ldt, strideT,
                       rocblas_fill_full, copymat_from_buffer);

    // work = imag(A)
    hipLaunchKernelGGL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                       stream, n, n, A, shiftA, lda, strideA, work, shiftT, ldt, strideT,
                       rocblas_fill_full, copymat_to_buffer);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // imag(A) = temp
    hipLaunchKernelGGL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                       stream, n, n, A, shiftA, lda, strideA, temp, shiftT, ldt, strideT,
                       rocblas_fill_full, copymat_from_buffer);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack,
                                   size_t* size_tempvect,
                                   size_t* size_tempgemm,
                                   size_t* size_workArr)
{
    constexpr bool COMPLEX = is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        return;
    }

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
    }

    // if size is too small, use steqr
    else if(n <= STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        size_t s1, s2;

        // requirements for steqr of small independent blocks
        // (TODO: Size should be STEDC_MIN_DC_SIZE when DC method is implemented)
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &s1);

        // extra requirements for original eigenvectors of small independent blocks
        if(evect != rocblas_evect_tridiagonal)
        {
            *size_tempvect = n * n * batch_count * sizeof(S);
            *size_tempgemm = n * n * batch_count * sizeof(S);
            if(COMPLEX)
                s2 = n * n * batch_count * sizeof(S);
            else
                s2 = 0;
            if(BATCHED && !COMPLEX)
                *size_workArr = sizeof(S*) * batch_count;
            else
                *size_workArr = 0;
        }
        else
        {
            *size_tempvect = 0;
            *size_tempgemm = 0;
            *size_workArr = 0;
            s2 = 0;
        }

        *size_work_stack = max(s1, s2);
    }
}

template <typename T, typename S>
rocblas_status rocsolver_stedc_argCheck(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S D,
                                        S E,
                                        T C,
                                        const rocblas_int ldc,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_none && evect != rocblas_evect_tridiagonal
       && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(evect != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || (evect != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedc_template(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        void* work_stack,
                                        S* tempvect,
                                        S* tempgemm,
                                        S** workArr)
{
    ROCSOLVER_ENTER("stedc", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info = 0
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1 && evect != rocblas_evect_none)
        hipLaunchKernelGGL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
                           strideC, n, 1);
    if(n <= 1)
        return rocblas_status_success;

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_template<S>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                    batch_count, (rocblas_int*)work_stack);
    }

    // if size is too small, use steqr
    else if(n <= STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        rocblas_int blocks = (n - 1) / 32 + 1;

        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);

        // if eigenvectors of tridiagonal matrix are required, compute them directly in C
        if(evect == rocblas_evect_tridiagonal)
        {
            // initialize identity matrix in C
            hipLaunchKernelGGL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                               stream, n, n, C, shiftC, ldc, strideC);

            // execute divide and conquer kernel
            hipLaunchKernelGGL((stedc_kernel<T>), dim3(batch_count), dim3(1), 0, stream, n,
                               D + shiftD, strideD, E + shiftE, strideE, C, shiftC, ldc, strideC,
                               info, (S*)work_stack, eps, ssfmin, ssfmax);
        }

        // otherwise, an additional gemm will be required to update C
        else
        {
            rocblas_int ldt = n;
            rocblas_stride strideT = n * n;

            // initialize identity matrix in tempvect
            hipLaunchKernelGGL(init_ident<S>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                               stream, n, n, tempvect, 0, ldt, strideT);

            // execute divide and conquer kernel with tempvect
            hipLaunchKernelGGL((stedc_kernel<S>), dim3(batch_count), dim3(1), 0, stream, n,
                               D + shiftD, strideD, E + shiftE, strideE, tempvect, 0, ldt, strideT,
                               info, (S*)work_stack, eps, ssfmin, ssfmax);

            // update eigenvectors C <- C*tempvect
            local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                            (S*)work_stack, 0, ldt, strideT, batch_count, workArr);
        }
    }

    return rocblas_status_success;
}
