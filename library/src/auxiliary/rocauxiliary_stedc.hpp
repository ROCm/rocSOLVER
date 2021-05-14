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
__global__ void stedc_kernel(const rocblas_evect evect,
                             const rocblas_int n,
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
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* info = iinfo + bid;
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);

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

        //printf("k: %d, bs: %d\n",k,bs);

        // if block is too small, solve it with steqr
        if(bs <= 100000) //STEDC_MIN_DC_SIZE)
        {
            // if computing vectors of tridiagonal matrix simply use steqr
            if(evect == rocblas_evect_tridiagonal)
            {
                rocblas_stride strideW = 2 * bs - 2;
                S* work = WW + (bid * strideW);

                run_steqr(bs, D + k, E + k, C + k + k * ldc, ldc, info, work, 30 * bs, eps, ssfmin,
                          ssfmax);
            }

            // otherwise, an extra gemm will be required to update C
            else
            {
            }
        }

        // otherwise solve it with divide and conquer method
        else
        {
        }

        k += bs;
    }
}

template <typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack)
{
    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        return;
    }

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
    }

    // if size is too small, use steqr
    else if(n <= STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
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

template <typename T, typename S, typename U>
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
                                        void* work_stack)
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

    // Initialize identity matrix
    if(evect == rocblas_evect_tridiagonal)
    {
        rocblas_int blocks = (n - 1) / 32 + 1;
        hipLaunchKernelGGL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                           stream, n, n, C, shiftC, ldc, strideC);
    }

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
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);

        hipLaunchKernelGGL((stedc_kernel<T>), dim3(batch_count), dim3(1), 0, stream, evect, n,
                           D + shiftD, strideD, E + shiftE, strideE, C, shiftC, ldc, strideC, info,
                           (S*)work_stack, eps, ssfmin, ssfmax);
    }

    return rocblas_status_success;
}
