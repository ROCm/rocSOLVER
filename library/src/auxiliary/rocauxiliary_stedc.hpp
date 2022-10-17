/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/** STEDC_NUM_LEV returns the ideal number of times a matrix (or split block)
    will be divided during the divide phase of divide & conquer algorithm **/
__host__ __device__ inline rocblas_int stedc_num_levs(const rocblas_int n)
{
    return 2;
}

/** STEDC_SPLIT finds independent blocks in the tridiagonal matrix
    given by D and E. (The independent blocks can then be solved in
    parallel by the DC algorithm) **/
template <typename S>
ROCSOLVER_KERNEL void stedc_split(const rocblas_int n,
                                  S* DD,
                                  const rocblas_stride strideD,
                                  S* EE,
                                  const rocblas_stride strideE,
                                  rocblas_int* splitsA,
                                  const S eps)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* splits = splitsA + bid * (2 * n + 2);

    rocblas_int k = 0; //position where the last block starts
    S tol; //tolerance. If an element of E is <= tol we have an independent block
    rocblas_int bs; //size of an independent block
    rocblas_int nb = 1; //number of blocks
    splits[0] = 0; //positions where each block begings

    // main loop
    /*    while(k < n)
    {
        bs = 1;
        for(rocblas_int j = k; j < n - 1; ++j)
        {
            tol = eps * sqrt(abs(D[j])) * sqrt(abs(D[j + 1]));
            if(abs(E[j]) < tol)
            {
                // Split next independent block
                // save its location in matrix
                splits[nb] = j+1;
                nb++;
                break;
            }
            bs++;
        }
        k += bs;
    }*/
    splits[nb] = n;
    splits[n + 1] = nb; //also save the number of split blocks
}

/** STEDC_KERNEL implements the main loop of the DC algorithm
    to compute the eigenvalues/eigenvectors of the symmetric tridiagonal
    submatrices **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void stedc_kernel(const rocblas_int n,
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
                                   rocblas_int* splitsA,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax)
{
    rocblas_int bid = hipBlockIdx_y; // batch instance id
    rocblas_int sid = hipBlockIdx_x; // split block id
    rocblas_int tid = hipThreadIdx_x; // thread id
    rocblas_int bdim = hipBlockDim_x; // number of threads

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T* C;
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* info = iinfo + bid;
    rocblas_int* splits = splitsA + bid * (2 * n + 2);
    rocblas_int* idd = splits + n + 2;
    S* W = WW + (bid * 2 * n);

    // info of split blocks
    rocblas_int nb = splits[n + 1]; // total number of blocks
    rocblas_int bs; // size of split block
    rocblas_int p1; // begining of split block

    // info of sub-blocks
    rocblas_int p2; // begining of sub-block
    rocblas_int blks; // number of sub-blocks
    rocblas_int levs; // number of level of division
    S p;

    // shared temp arrays
    extern __shared__ rocblas_int lmem[];
    rocblas_int* ns = lmem; // shares the sub-blocks sizes
    rocblas_int* ps = ns + bdim; // shares the sub-blocks initial positions

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    for(int k = sid; k < nb; k += STEDC_NUM_SPLIT_BLKS)
    {
        // Select current split block
        p1 = splits[k];
        p2 = splits[k + 1];
        bs = p2 - p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levs(bs);
        blks = 1 << levs;

        // if split block is too small, solve it with steqr
        if(blks == 1)
        {
            if(tid == 0)
            {
                run_steqr(bs, D + p1, E + p1, C + p1 + p1 * ldc, ldc, info, W + p1 * 2, 30 * bs,
                          eps, ssfmin, ssfmax, false);
            }
        }

        // otherwise, divide & conquer with blks threads (one thread per sub-block)
        else
        {
            /************************* 1. divide phase *************************/
            // (artificially divide split block into blks sub-blocks
            // find initial positions of each sub-blocks)
            ns[tid] = 0;

            // find sub-block sizes
            if(tid == 0)
            {
                ns[0] = bs;
                rocblas_int t, t2;
                for(int i = 0; i < levs; ++i)
                {
                    for(int j = (1 << i); j > 0; --j)
                    {
                        t = ns[j - 1];
                        t2 = t / 2;
                        ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                        ns[j * 2 - 2] = t2;
                    }
                }
            }
            __syncthreads();

            // find begining of this thread sub-block and update D elements
            p2 = 0;
            for(int i = 0; i < tid; ++i)
                p2 += ns[i];
            p2 += p1;
            ps[tid] = p2;
            if(tid > 0 && tid < blks)
            {
                // perform sub-block division
                p = E[p2 - 1];
                D[p2] -= p;
                D[p2 - 1] -= p;
            }
            __syncthreads();
            /******************************************************************/
            //if(tid==0)
            //{
            //for(int v=0;v<blks;++v)
            //    printf("%d ",ps[v]);
            //printf("\n");
            //}

            /************************* 2. solve phase *************************/
            // (solve the blks sub-blocks in parallel)
            //            run_steqr(ns[tid], D + p2, E + p2, C + p2 + p2 * ldc, ldc, info,
            //                        W + p2 * 2, 30 * bs, eps, ssfmin, ssfmax, false);
            //            __syncthreads();
            /******************************************************************/

            /************************* 3. Merge phase *************************/
            // (merge results of the blks sub-blocks to generate
            //  the solution of the entire split block)
            T* z;
            for(int k = 0; k < levs; ++k)
            {
                // +++++++++++++++++++++++++++++++++++
                // a. find rank-1 modification components (z and p) for this merge
                // (threads with iam < bd work with components above the merge point;
                //  threads with iam >= bd work below the merge point)
                rocblas_int bd = 1 << k;
                rocblas_int iam = tid % (bd << 1);
                if(iam < bd && tid < blks)
                {
                    rocblas_int sz = ns[tid];
                    for(int j = 1; j < bd - iam; ++j)
                        sz += ns[tid + j];
                    // with this, all threads involved in a merge (above merge point)
                    // will point to the same row of C and the same off-diag element
                    z = C + p2 - 1 + sz;
                    p = E[p2 - 1 + sz];
                }
                else if(iam >= bd && tid < blks)
                {
                    rocblas_int sz = 0;
                    for(int j = 0; j < iam - bd; ++j)
                        sz += ns[tid - j - 1];
                    // with this, all threads involved in a merge (below merge point)
                    // will point to the same row of C and the same off-diag element
                    z = C + p2 - sz;
                    p = E[p2 - sz - 1];
                }
                __syncthreads();
                // +++++++++++++++++++++++++++++++++++

                // +++++++++++++++++++++++++++++++++++
                // b. calculate deflation tolerance
                // tol = 8 * eps * (max diagonal or off-diagonal element participating in merge)
                S val;
                S tol = 0.00000000001;
                // +++++++++++++++++++++++++++++++++++

                // +++++++++++++++++++++++++++++++++++
                // c. deflate enigenvalues
                // first deflate each thread sub-block
                for(int i = 0; i < ns[tid]; ++i)
                {
                    //                    if(abs(p * z[(p2 + i) * ldc]) <= tol)
                    //                    {
                    //                        // deflated ev because component in z is zero
                    //                        idd[p2 + i] = 0;
                    //                    }
                    //                    else
                    //                    {
                    rocblas_int jj = 1;
                    val = D[p2 + i];
                    for(int j = 0; j < i; ++j)
                    {
                        if(idd[p2 + j] == 1 && abs(D[p2 + j] - val) <= tol)
                        {
                            // deflated ev because it is repeated
                            idd[p2 + i] = 0;
                            break;
                        }
                        jj++;
                    }
                    if(jj > i)
                    {
                        // non-deflated ev
                        idd[p2 + i] = 1;
                    }
                    //                    }
                }
                __syncthreads();

                // then compare with other sub-blocks participating in this merge
                // (follows a simple, reduction-like process)
                for(int ii = 0; ii <= k; ++ii)
                {
                    rocblas_int div = 1 << (ii + 1);
                    if(iam % div == div - 1) //actual number of threads is halved each time
                    {
                        // find limits
                        rocblas_int inb = (1 << ii) - 1;
                        rocblas_int inc = div - 1;
                        rocblas_int countb = ns[tid];
                        rocblas_int countc = 0;
                        for(int i = inc; i > inb; --i)
                            countc += ns[tid - i];
                        for(int i = inb; i > 0; --i)
                            countb += ns[tid - i];
                        inb = ps[tid - inb];
                        inc = ps[tid - inc];

                        // perform comparisons
                        for(int i = 0; i < countb; ++i)
                        {
                            if(idd[inb + i] == 1)
                            {
                                val = D[inb + i];
                                for(int j = 0; j < countc; ++j)
                                {
                                    if(idd[inc + j] == 1 && abs(D[inc + j] - val) <= tol)
                                    {
                                        // deflated ev because it is repeated
                                        idd[inb + i] = 0;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                // (at this point, if idd[i] = 0, it means that D[i] has been deflated
                // +++++++++++++++++++++++++++++++++++

                // +++++++++++++++++++++++++++++++++++
                // d. Find secular equations using the non-deflated values
            }
            /**********************************************/
        }
        __syncthreads();
    }
}

/** STEDC_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void stedc_sort(const rocblas_int n,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 U CC,
                                 const rocblas_int shiftC,
                                 const rocblas_int ldc,
                                 const rocblas_stride strideC)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T* C;
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* D = DD + (bid * strideD);

    rocblas_int l, m;
    S p;

    // Sort eigenvalues and eigenvectors by selection sort
    for(int ii = 1; ii < n; ii++)
    {
        l = ii - 1;
        m = l;
        p = D[l];
        for(int j = ii; j < n; j++)
        {
            if(D[j] < p)
            {
                m = j;
                p = D[j];
            }
        }
        if(m != l)
        {
            D[m] = D[l];
            D[l] = p;
            swapvect(n, C + 0 + l * ldc, 1, C + 0 + m * ldc, 1);
        }
    }
}

/** This local gemm adapts rocblas_gemm to multiply complex*real, and
    overwrite result: A = A*B **/
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
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
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                            stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, temp);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
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
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // real(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    // work = imag(A)
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // imag(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack,
                                   size_t* size_tempvect,
                                   size_t* size_tempgemm,
                                   size_t* size_splits,
                                   size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        return;
    }

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
    }

    // if size is too small, use steqr
    else if(n <= 0) //STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        size_t s1, s2;

        // requirements for steqr of small independent blocks
        // (TODO: Size should be STEDC_MIN_DC_SIZE when DC method is implemented)
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &s1);

        // extra requirements for original eigenvectors of small independent blocks
        //        if(evect != rocblas_evect_tridiagonal)
        //        {
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
        //        }
        //        else
        //        {
        //            *size_tempvect = 0;
        //            *size_tempgemm = 0;
        //            *size_workArr = 0;
        //            s2 = 0;
        //        }
        *size_work_stack = max(s1, s2);

        // size for split blocks positions
        *size_splits = sizeof(rocblas_int) * (2 * n + 2) * batch_count;
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
                                        rocblas_int* splits,
                                        S** workArr)
{
    ROCSOLVER_ENTER("stedc", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    print_device_matrix(std::cout, "D", 1, n, D, 1);
    print_device_matrix(std::cout, "E", 1, n, E, 1);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1 && evect != rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
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
    else if(n <= 0) //STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);
        rocblas_int blocksn = (n - 1) / BS2 + 1;

        // find independent split blocks in matrix
        ROCSOLVER_LAUNCH_KERNEL(stedc_split, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
                                strideD, E + shiftE, strideE, splits, eps);

        // initialize identity matrix in C if required
        if(evect == rocblas_evect_tridiagonal)
            ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocksn, blocksn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

        // initialize identity matrix in tempvect
        rocblas_int ldt = n;
        rocblas_stride strideT = n * n;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, tempvect, 0, ldt, strideT);

        // find max number of sub-blocks to consider during the divide phase
        rocblas_int maxblks = 1 << stedc_num_levs(n);
        size_t lmemsize = sizeof(rocblas_int) * 2 * maxblks;

        // execute divide and conquer kernel with tempvect
        ROCSOLVER_LAUNCH_KERNEL((stedc_kernel<S>), dim3(STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(maxblks), lmemsize, stream, n, D + shiftD, strideD, E + shiftE,
                                strideE, tempvect, 0, ldt, strideT, info, (S*)work_stack, splits,
                                eps, ssfmin, ssfmax);

        // update eigenvectors C <- C*tempvect
        local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                        (S*)work_stack, 0, ldt, strideT, batch_count, workArr);

        print_device_matrix(std::cout, "Dfin", 1, n, D, 1);
        print_device_matrix(std::cout, "idd", 1, n, splits + n + 2, 1);

        // finally sort eigenvalues and eigenvectors
        ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, C, shiftC, ldc, strideC);
    }

    return rocblas_status_success;
}
