/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.10.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

#define SPLIT_THDS 256
#define IBISEC_BLKS 128
#define IBISEC_THDS 256

/************** Kernels and device functions *********************************/
/*****************************************************************************/

/** This kernel deals with the case n = 1
    (one split block and a single eigenvalue which is the element in D) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void stebz_case1_kernel(const rocblas_eval_range range,
                                         const T vlow,
                                         const T vup,
                                         U DA,
                                         const rocblas_int shiftD,
                                         const rocblas_stride strideD,
                                         rocblas_int* nev,
                                         rocblas_int* nsplit,
                                         T* WA,
                                         const rocblas_stride strideW,
                                         rocblas_int* IBA,
                                         const rocblas_stride strideIB,
                                         rocblas_int* ISA,
                                         const rocblas_stride strideIS,
                                         const rocblas_int batch_count)
{
    int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(bid < batch_count)
    {
        // select bacth instance
        T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
        T* W = WA + bid * strideW;
        rocblas_int* IB = IBA + bid * strideIB;
        rocblas_int* IS = ISA + bid * strideIS;

        // one split block
        nsplit[bid] = 1;
        IS[0] = 1;

        // check if diagonal element is in range and return
        T d = D[0];
        if(range == rocblas_range_value && (d <= vlow || d > vup))
        {
            nev[bid] = 0;
        }
        else
        {
            nev[bid] = 1;
            W[0] = d;
            IB[0] = 1;
        }
    }
}

/** This kernel splits the matrix in independent blocks and prepares things
    for the computations in the iterative bisection **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SPLIT_THDS)
    stebz_spliting_kernel(const rocblas_eval_range range,
                          const rocblas_int n,
                          const T vlow,
                          const T vup,
                          const rocblas_int ilow,
                          const rocblas_int iup,
                          U DA,
                          const rocblas_int shiftD,
                          const rocblas_int strideD,
                          U EA,
                          const rocblas_int shiftE,
                          const rocblas_int strideE,
                          rocblas_int* nsplit,
                          rocblas_int* ISA,
                          const rocblas_stride strideIS,
                          rocblas_int* tmpISA,
                          T* pivmin,
                          T* EsqrA,
                          T* boundsA,
                          T eps,
                          T sfmin)
{
    // batch instance
    const int bid = hipBlockIdx_y;
    T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
    T* E = load_ptr_batch<T>(EA, bid, shiftE, strideE);
    rocblas_int* IS = ISA + bid * strideIS;
    T* Esqr = EsqrA + bid * (n - 1);
    T* bounds = boundsA + 2 * bid;

    // shared memory setup for iamax.
    // (sidx also temporarily stores the number of blocks found by each thread)
    __shared__ T sval[SPLIT_THDS];
    __shared__ rocblas_int sidx[SPLIT_THDS];

    // the number of elements worked by this thread is nn
    const int tid = hipThreadIdx_x;
    rocblas_int nn = (n - 1) / SPLIT_THDS;
    if(tid < n - 1 - nn * SPLIT_THDS)
        nn++;
    sidx[tid] = nn;
    __syncthreads();

    // thus, this thread offset is:
    rocblas_int offset = 0;
    for(int i = 0; i < tid; ++i)
        offset += sidx[i];

    // this thread find its split-off blocks
    // tmpIS stores the block indices found by this thread
    rocblas_int* tmpIS = tmpISA + (bid * n) + offset;
    T tmp;
    rocblas_int j;
    rocblas_int tmpns = 0; //temporary number of blocks found
    for(rocblas_int i = 0; i < nn; ++i)
    {
        j = i + offset;

        tmp = E[j];
        tmp *= tmp;
        Esqr[j] = tmp;

        if(std::abs(D[j] * D[j + 1]) * eps * eps + sfmin > tmp)
        {
            tmpIS[tmpns] = j;
            tmpns++;
        }
    }
    sidx[tid] = tmpns;
    __syncthreads();

    // find split-off blocks in entire matrix
    offset = 0;
    for(int i = 0; i < tid; ++i)
        offset += sidx[i];
    for(int i = 0; i < tmpns; ++i)
        IS[i + offset] = tmpIS[i] + 1;

    // total number of split blocks
    if(tid == SPLIT_THDS - 1)
    {
        offset += tmpns;
        nsplit[bid] = offset + 1;
        IS[offset] = n;
    }
    __syncthreads();

    // find max squared off-diagonal element
    iamax<SPLIT_THDS>(tid, n - 1, Esqr, 1, sval, sidx);
    __syncthreads();

    if(tid == 0)
    {
        // compute pivmin (minimum value that can be pivot in sturm sequence)
        pivmin[bid] = sval[0] * sfmin;

        // Find upper and lower bounds vl and vu of the absolute interval (vl, vu] where
        // the eigenavlues will be searched. vl and vu are set to zero when looking for
        // all the eigenvalues in matrix,
        T vl = 0;
        T vu = 0;
        if(range == rocblas_range_index)
        {
        }
        else if(range == rocblas_range_value)
        {
            vl = vlow;
            vu = vup;
        }
        bounds[0] = vl;
        bounds[1] = vu;
    }
}

/** This is the main kernel that implements the iterative bisection.
    Each thread works with as many non-converged intervals as needed on each iteration.
    Each thread-block is working with as many split-off blocks as needed to cover
    the entire matrix **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(IBISEC_THDS)
    stebz_bisection_kernel(const rocblas_eval_range range,
                           const rocblas_int n,
                           U DA,
                           const rocblas_int shiftD,
                           const rocblas_int strideD,
                           U EA,
                           const rocblas_int shiftE,
                           const rocblas_int strideE,
                           rocblas_int* nev,
                           rocblas_int* nsplit,
                           T* WA,
                           const rocblas_stride strideW,
                           rocblas_int* IBA,
                           const rocblas_stride strideIB,
                           rocblas_int* ISA,
                           const rocblas_stride strideIS,
                           rocblas_int* info,
                           rocblas_int* tmpnevA,
                           T* pivmin,
                           T* EsqrA,
                           T* boundsA,
                           T eps,
                           T sfmin)
{
    // batch instance
    const int bid = hipBlockIdx_y;
    T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
    T* E = load_ptr_batch<T>(EA, bid, shiftE, strideE);
    T* W = WA + bid * strideW;
    rocblas_int* IB = IBA + bid * strideIB;
    rocblas_int* IS = ISA + bid * strideIS;
    T* Esqr = EsqrA + bid * (n - 1);
    T* bounds = boundsA + 2 * bid;
    rocblas_int nofb = nsplit[bid];
    T pmin = pivmin[bid];
    rocblas_int* tmpnev = tmpnevA + bid * n;

    const int tid = hipThreadIdx_x;
    const int sbid = hipBlockIdx_x;
    rocblas_int bin, bout, bdim;
    T tmp;

    // loop over idependent split blocks
    for(int b = sbid; b < nofb; b += IBISEC_BLKS)
    {
        // initialize number of eigenvalues for current split block
        tmpnev[sbid] = 0;

        // find dimension and indices of current split block
        bin = (b == 0) ? 0 : IS[b - 1];
        bout = IS[b] - 1;
        bdim = bout - bin + 1;

        // if current split block has dimension 1, quick return
        if(bdim == 1)
        {
            if(tid == 0)
            {
                tmp = D[bin];
                if((range == rocblas_range_all)
                   || (bounds[0] < tmp - pmin && bounds[1] >= tmp - pmin))
                {
                    W[bin] = tmp;
                    tmpnev[sbid] = 1;
                    IB[bin] = sbid + 1;
                }
            }
        }

        // otherwise do iterative bisection
        else
        {
            __syncthreads();
        }
    }
}

/** This kernel synthetize the results from all the independent
    split blocks of a given matrix **/
template <typename T>
ROCSOLVER_KERNEL void stebz_synthesis_kernel(const rocblas_int n,
                                             rocblas_int* nev,
                                             rocblas_int* nsplit,
                                             T* WA,
                                             const rocblas_stride strideW,
                                             rocblas_int* IBA,
                                             const rocblas_stride strideIB,
                                             rocblas_int* ISA,
                                             const rocblas_stride strideIS,
                                             const rocblas_int batch_count,
                                             rocblas_int* tmpnevA)
{
    int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(bid < batch_count)
    {
        // select bacth instance
        T* W = WA + bid * strideW;
        rocblas_int* IB = IBA + bid * strideIB;
        rocblas_int* IS = ISA + bid * strideIS;
        rocblas_int nofb = nsplit[bid];
        rocblas_int* tmpnev = tmpnevA + bid * n;

        rocblas_int bin;
        rocblas_int nn = 0;

        // re-arrange W and IB
        for(int b = 0; b < nofb; ++b)
        {
            bin = (b == 0) ? 0 : IS[b - 1];
            for(int bb = 0; bb < tmpnev[b]; ++bb)
            {
                W[nn] = W[bin + bb];
                IB[nn] = IB[bin + bb];
                nn++;
            }
        }

        // total number of eigenvalues found
        nev[bid] = nn;
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

// Helper to calculate workspace size requirements
template <typename T>
void rocsolver_stebz_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work,
                                   size_t* size_pivmin,
                                   size_t* size_Esqr,
                                   size_t* size_bounds)
{
    // if quick return no workspace needed
    //    if(n == 0 || !batch_count)
    //    {
    //        *size_stack = 0;
    //        return;
    //    }

    *size_work = sizeof(rocblas_int) * n * batch_count;
    *size_pivmin = sizeof(T) * batch_count;
    *size_Esqr = sizeof(T) * (n - 1) * batch_count;
    *size_bounds = sizeof(T) * 2 * batch_count;
}

// Helper to check argument correctnesss
template <typename T>
rocblas_status rocsolver_stebz_argCheck(rocblas_handle handle,
                                        const rocblas_eval_range range,
                                        const rocblas_eval_order order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        T* D,
                                        T* E,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        rocblas_int* IB,
                                        rocblas_int* IS,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(range != rocblas_range_all && range != rocblas_range_value && range != rocblas_range_index)
        return rocblas_status_invalid_value;
    if(order != rocblas_order_blocks && order != rocblas_order_entire)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(range == rocblas_range_value && vlow >= vup)
        return rocblas_status_invalid_size;
    if(range == rocblas_range_index && (ilow < 1 || iup < 0))
        return rocblas_status_invalid_size;
    if(range == rocblas_range_index && (iup > n || (n > 0 && ilow > iup)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !W || !IB || !IS)) || (n > 1 && !E) || !info || !nev || !nsplit)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// stebz template function implementation
template <typename T, typename U>
rocblas_status rocsolver_stebz_template(rocblas_handle handle,
                                        const rocblas_eval_range range,
                                        const rocblas_eval_order order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        const T abstol,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* IB,
                                        const rocblas_stride strideIB,
                                        rocblas_int* IS,
                                        const rocblas_stride strideIS,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* work,
                                        T* pivmin,
                                        T* Esqr,
                                        T* bounds)
{
    ROCSOLVER_ENTER("stebz", "range:", range, "order:", order, "n:", n, "vlow:", vlow, "vup:", vup,
                    "ilow:", ilow, "iup:", iup, "abstol:", abstol, "shiftD:", shiftD,
                    "shiftE:", shiftE, "bc:", batch_count);

    // quick return (no batch)
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = nev = nsplit = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nsplit, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return (dimension zero)
    if(n == 0)
        return rocblas_status_success;

    // quick return (dimension 1)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(stebz_case1_kernel<T>, gridReset, threads, 0, stream, range, vlow,
                                vup, D, shiftD, strideD, nev, nsplit, W, strideW, IB, strideIB, IS,
                                strideIS, batch_count);

        return rocblas_status_success;
    }

    // numerics constants:
    // machine epsilon
    T eps = get_epsilon<T>();
    // smallest safe real (i.e. 1/sfmin does not overflow)
    T sfmin = get_safemin<T>();
    // relative tolerance for evaluating when and eigenvalue interval is small
    // enough to consider it as converged
    T rtol = 2 * eps;

    // split matrix into independent blocks and prepare for iterative bisection
    ROCSOLVER_LAUNCH_KERNEL(stebz_spliting_kernel<T>, dim3(1, batch_count), dim3(SPLIT_THDS), 0,
                            stream, range, n, vlow, vup, ilow, iup, D, shiftD, strideD, E, shiftE,
                            strideE, nsplit, IS, strideIS, work, pivmin, Esqr, bounds, eps, sfmin);

    // Implement iterative bisection on each split block.
    // The next kernel has IBISEC_BLKS thread-blocks with IBISEC_THDS threads.
    // Each thread works with as many non-converged intervals as needed on each iteration.
    // Each thread-block is working with as many split-off blocks as needed to cover
    // the entire matrix.

    /** (TODO: in the future, we can evaluate if transfering nsplit -the number of
        split-off blocks- into the host, to launch exactly that amount of thread-blocks,
        could give better performance) **/

    ROCSOLVER_LAUNCH_KERNEL(stebz_bisection_kernel<T>, dim3(IBISEC_BLKS, batch_count),
                            dim3(IBISEC_THDS), 0, stream, range, n, D, shiftD, strideD, E, shiftE,
                            strideE, nev, nsplit, W, strideW, IB, strideIB, IS, strideIS, info,
                            work, pivmin, Esqr, bounds, eps, sfmin);

    // Finally, synthetize the results from all the split blocks
    ROCSOLVER_LAUNCH_KERNEL(stebz_synthesis_kernel<T>, gridReset, threads, 0, stream, n, nev,
                            nsplit, W, strideW, IB, strideIB, IS, strideIS, batch_count, work);

    print_device_matrix<T>(std::cout, "D", 1, n, D, 1);
    print_device_matrix<T>(std::cout, "E", 1, n - 1, E, 1);
    print_device_matrix<rocblas_int>(std::cout, "nsplit", 1, 1, nsplit, 1);
    print_device_matrix<rocblas_int>(std::cout, "nev", 1, 1, nev, 1);
    print_device_matrix<T>(std::cout, "W", 1, n, W, 1);
    print_device_matrix<rocblas_int>(std::cout, "IS", 1, n, IS, 1);
    print_device_matrix<rocblas_int>(std::cout, "IB", 1, n, IB, 1);
    print_device_matrix<rocblas_int>(std::cout, "work", 1, n, work, 1);

    return rocblas_status_success;
}
