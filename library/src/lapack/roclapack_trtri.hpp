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
#include "rocblas.hpp"
#include "rocsolver.h"

#ifdef OPTIMAL
template <rocblas_int DIM, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(WAVESIZE) trti2_kernel_small(const rocblas_fill uplo,
                                                                     const rocblas_diagonal diagtype,
                                                                     U AA,
                                                                     const rocblas_int shiftA,
                                                                     const rocblas_int lda,
                                                                     const rocblas_stride strideA)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= DIM)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);

    // read corresponding row from global memory in local array
    T rA[DIM];
#pragma unroll
    for(int j = 0; j < DIM; ++j)
        rA[j] = A[i + j * lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    __shared__ T diag[DIM];
    T temp;

    // diagonal element
    const bool unit = (diagtype == rocblas_diagonal_unit);
    if(unit)
    {
        diag[i] = -1.0;
    }
    else
    {
        rA[i] = 1.0 / rA[i];
        diag[i] = -rA[i];
    }

    // compute element i of each column j
    if(uplo == rocblas_fill_upper)
    {
#pragma unroll
        for(rocblas_int j = 1; j < DIM; j++)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i < j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = i + 1; ii < j; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
        }
    }
    else
    {
#pragma unroll
        for(rocblas_int j = DIM - 2; j >= 0; j--)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i > j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = j + 1; ii < i; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
        }
    }

    // write results to global memory from local array
#pragma unroll
    for(int j = 0; j < DIM; j++)
        A[i + j * lda] = rA[j];
}

template <typename T, typename U>
void trti2_run_small(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_diagonal diag,
                     const rocblas_int n,
                     U A,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const rocblas_int batch_count)
{
#define RUN_TRTI2_SMALL(DIM)                                                                \
    hipLaunchKernelGGL((trti2_kernel_small<DIM, T>), grid, block, 0, stream, uplo, diag, A, \
                       shiftA, lda, strideA)

    dim3 grid(batch_count, 1, 1);
    dim3 block(WAVESIZE, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch(n)
    {
    case 1: RUN_TRTI2_SMALL(1); break;
    case 2: RUN_TRTI2_SMALL(2); break;
    case 3: RUN_TRTI2_SMALL(3); break;
    case 4: RUN_TRTI2_SMALL(4); break;
    case 5: RUN_TRTI2_SMALL(5); break;
    case 6: RUN_TRTI2_SMALL(6); break;
    case 7: RUN_TRTI2_SMALL(7); break;
    case 8: RUN_TRTI2_SMALL(8); break;
    case 9: RUN_TRTI2_SMALL(9); break;
    case 10: RUN_TRTI2_SMALL(10); break;
    case 11: RUN_TRTI2_SMALL(11); break;
    case 12: RUN_TRTI2_SMALL(12); break;
    case 13: RUN_TRTI2_SMALL(13); break;
    case 14: RUN_TRTI2_SMALL(14); break;
    case 15: RUN_TRTI2_SMALL(15); break;
    case 16: RUN_TRTI2_SMALL(16); break;
    case 17: RUN_TRTI2_SMALL(17); break;
    case 18: RUN_TRTI2_SMALL(18); break;
    case 19: RUN_TRTI2_SMALL(19); break;
    case 20: RUN_TRTI2_SMALL(20); break;
    case 21: RUN_TRTI2_SMALL(21); break;
    case 22: RUN_TRTI2_SMALL(22); break;
    case 23: RUN_TRTI2_SMALL(23); break;
    case 24: RUN_TRTI2_SMALL(24); break;
    case 25: RUN_TRTI2_SMALL(25); break;
    case 26: RUN_TRTI2_SMALL(26); break;
    case 27: RUN_TRTI2_SMALL(27); break;
    case 28: RUN_TRTI2_SMALL(28); break;
    case 29: RUN_TRTI2_SMALL(29); break;
    case 30: RUN_TRTI2_SMALL(30); break;
    case 31: RUN_TRTI2_SMALL(31); break;
    case 32: RUN_TRTI2_SMALL(32); break;
    case 33: RUN_TRTI2_SMALL(33); break;
    case 34: RUN_TRTI2_SMALL(34); break;
    case 35: RUN_TRTI2_SMALL(35); break;
    case 36: RUN_TRTI2_SMALL(36); break;
    case 37: RUN_TRTI2_SMALL(37); break;
    case 38: RUN_TRTI2_SMALL(38); break;
    case 39: RUN_TRTI2_SMALL(39); break;
    case 40: RUN_TRTI2_SMALL(40); break;
    case 41: RUN_TRTI2_SMALL(41); break;
    case 42: RUN_TRTI2_SMALL(42); break;
    case 43: RUN_TRTI2_SMALL(43); break;
    case 44: RUN_TRTI2_SMALL(44); break;
    case 45: RUN_TRTI2_SMALL(45); break;
    case 46: RUN_TRTI2_SMALL(46); break;
    case 47: RUN_TRTI2_SMALL(47); break;
    case 48: RUN_TRTI2_SMALL(48); break;
    case 49: RUN_TRTI2_SMALL(49); break;
    case 50: RUN_TRTI2_SMALL(50); break;
    case 51: RUN_TRTI2_SMALL(51); break;
    case 52: RUN_TRTI2_SMALL(52); break;
    case 53: RUN_TRTI2_SMALL(53); break;
    case 54: RUN_TRTI2_SMALL(54); break;
    case 55: RUN_TRTI2_SMALL(55); break;
    case 56: RUN_TRTI2_SMALL(56); break;
    case 57: RUN_TRTI2_SMALL(57); break;
    case 58: RUN_TRTI2_SMALL(58); break;
    case 59: RUN_TRTI2_SMALL(59); break;
    case 60: RUN_TRTI2_SMALL(60); break;
    case 61: RUN_TRTI2_SMALL(61); break;
    case 62: RUN_TRTI2_SMALL(62); break;
    case 63: RUN_TRTI2_SMALL(63); break;
    case 64: RUN_TRTI2_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }
}
#endif // OPTIMAL

template <typename T, typename U>
ROCSOLVER_KERNEL void invdiag(const rocblas_diagonal diag,
                              const rocblas_int n,
                              U A,
                              const rocblas_int shiftA,
                              const rocblas_int lda,
                              const rocblas_stride strideA,
                              T* alphas)
{
    int b = hipBlockIdx_y;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* d = alphas + b * n;

        if(a[i + i * lda] != 0 && diag == rocblas_diagonal_non_unit)
        {
            a[i + i * lda] = 1.0 / a[i + i * lda];
            d[i] = -a[i + i * lda];
        }
        else
            d[i] = -1.0;
    }
}

template <bool ISBATCHED>
rocblas_int trtri_get_blksize(const rocblas_int dim)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int size[] = {TRTRI_BATCH_BLKSIZES};
        rocblas_int intervals[] = {TRTRI_BATCH_INTERVALS};
        rocblas_int max = TRTRI_BATCH_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }
    else
    {
        rocblas_int size[] = {TRTRI_BLKSIZES};
        rocblas_int intervals[] = {TRTRI_INTERVALS};
        rocblas_int max = TRTRI_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }

    return blk;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_trtri_getMemorySize(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_tmpcopy,
                                   size_t* size_workArr)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_tmpcopy = 0;
        *size_workArr = 0;
        return;
    }

    // get block size
    rocblas_int blk = trtri_get_blksize<ISBATCHED>(n);

    // size of temporary array required for copies
    if(diag == rocblas_diagonal_unit && blk > 0)
        *size_tmpcopy = 0;
    else
        *size_tmpcopy = n * n * sizeof(T) * batch_count;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    size_t w1a, w1b, w3a, w3b;

    // requirements for TRTI2
    rocblas_int nn = (blk == 1) ? n : blk;
#ifdef OPTIMAL
    if(nn <= WAVESIZE)
    {
        // if very small size, no workspace needed
        w1a = 0;
        w3a = 0;
    }
    else
    {
        // size for TRMV
        w1a = nn * sizeof(T) * batch_count;
        // size for alphas
        w3a = nn * sizeof(T) * batch_count;
    }
#else
    // size for TRMV
    w1a = nn * sizeof(T) * batch_count;
    // size for alphas
    w3a = nn * sizeof(T) * batch_count;
#endif

    if(blk == 0)
    {
        // requirements for calling rocBLAS TRTRI
        rocblasCall_trtri_mem<BATCHED, T>(n, batch_count, size_work1, size_work2);
        *size_work3 = 0;
        *size_work4 = 0;
    }
    else if(blk == 1)
    {
        *size_work1 = w1a;
        *size_work2 = 0;
        *size_work3 = w3a;
        *size_work4 = 0;
    }
    else
    {
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, n, blk, batch_count, &w1b, size_work2,
                                         &w3b, size_work4);
        *size_work1 = max(w1a, w1b);
        *size_work3 = max(w3a, w3b);
    }
}

template <typename T>
rocblas_status rocsolver_trtri_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void trti2(rocblas_handle handle,
           const rocblas_fill uplo,
           const rocblas_diagonal diag,
           const rocblas_int n,
           U A,
           const rocblas_int shiftA,
           const rocblas_int lda,
           const rocblas_stride strideA,
           const rocblas_int batch_count,
           T* work,
           T* alphas)
{
#ifdef OPTIMAL
    // if very small size, use optimized kernel
    if(n <= WAVESIZE)
    {
        trti2_run_small<T>(handle, uplo, diag, n, A, shiftA, lda, strideA, batch_count);
        return;
    }
#endif

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_stride stdw = rocblas_stride(n);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // inverse of the diagonal (reciprocals)
    rocblas_int blocks = (n - 1) / 32 + 1;
    hipLaunchKernelGGL(invdiag<T>, dim3(blocks, batch_count), dim3(32, 1), 0, stream, diag, n, A,
                       shiftA, lda, strideA, alphas);

    if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int j = 1; j < n; ++j)
        {
            rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, j, A, shiftA, lda,
                                strideA, A, shiftA + idx2D(0, j, lda), 1, strideA, work, stdw,
                                batch_count);

            rocblasCall_scal<T>(handle, j, alphas + j, stdw, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count);
        }
    }
    else //rocblas_fill_lower
    {
        for(rocblas_int j = n - 2; j >= 0; --j)
        {
            rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, n - j - 1, A,
                                shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                                shiftA + idx2D(j + 1, j, lda), 1, strideA, work, stdw, batch_count);

            rocblasCall_scal<T>(handle, n - j - 1, alphas + j, stdw, A,
                                shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trtri_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* tmpcopy,
                                        T** workArr,
                                        const bool optim_mem)
{
    ROCSOLVER_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // start with info = 0
    rocblas_int blocks = (batch_count - 1) / 32 + 1;
    hipLaunchKernelGGL(reset_info, dim3(blocks, 1, 1), dim3(32, 1, 1), 0, stream, info, batch_count,
                       0);

    // quick return if no dimensions
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    blocks = (n - 1) / 32 + 1;
    rocblas_int ldw = n;
    rocblas_stride strideW = n * n;

    // check for singularities if non-unit diagonal
    if(diag == rocblas_diagonal_non_unit)
    {
        hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, 64, 1), 0, stream,
                           n, A, shiftA, lda, strideA, info);
    }

    // get block size
    rocblas_int blk = trtri_get_blksize<ISBATCHED>(n);
    rocblas_int jb;

    if(diag == rocblas_diagonal_non_unit && blk > 0)
    {
        // save copy of A to restore it in cases where info is nonzero
        hipLaunchKernelGGL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                           stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, tmpcopy, info_mask(info));
    }

    if(blk == 0)
    {
        // simply use rocblas_trtri
        rocblasCall_trtri<BATCHED, STRIDED, T>(handle, uplo, diag, n, A, shiftA, lda, strideA,
                                               tmpcopy, 0, ldw, strideW, batch_count, (T*)work1,
                                               (T**)work2, workArr);

        // copy result to A if info is zero
        hipLaunchKernelGGL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                           stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,
                           info_mask(info, info_mask::negate), uplo, diag);
    }

    else if(blk == 1)
    {
        // use the unblocked algorithm
        trti2<T>(handle, uplo, diag, n, A, shiftA, lda, strideA, batch_count, (T*)work1, (T*)work3);
    }

    else
    {
        // use blocked algorithm with block size blk
        if(uplo == rocblas_fill_upper)
        {
            for(rocblas_int j = 0; j < n; j += blk)
            {
                jb = min(n - j, blk);

                // update current block column
                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, uplo, rocblas_operation_none, diag, j, jb, &one, 0, A,
                    shiftA, lda, strideA, A, shiftA + idx2D(0, j, lda), lda, strideA, batch_count);

                rocblasCall_trsm<BATCHED, T>(
                    handle, rocblas_side_right, uplo, rocblas_operation_none, diag, j, jb, &minone,
                    A, shiftA + idx2D(j, j, lda), lda, strideA, A, shiftA + idx2D(0, j, lda), lda,
                    strideA, batch_count, optim_mem, work1, work2, work3, work4);

                trti2<T>(handle, uplo, diag, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                         batch_count, (T*)work1, (T*)work3);
            }
        }
        else // rocblas_fill_lower
        {
            rocblas_int nn = ((n - 1) / blk) * blk + 1;
            for(rocblas_int j = nn - 1; j >= 0; j -= blk)
            {
                jb = min(n - j, blk);

                // update current block column
                rocblasCall_trmm<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, uplo, rocblas_operation_none, diag, n - j - jb, jb,
                    &one, 0, A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, A,
                    shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count);

                rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_right, uplo,
                                             rocblas_operation_none, diag, n - j - jb, jb, &minone,
                                             A, shiftA + idx2D(j, j, lda), lda, strideA, A,
                                             shiftA + idx2D(j + jb, j, lda), lda, strideA,
                                             batch_count, optim_mem, work1, work2, work3, work4);

                // inverse of current diagonal block
                trti2<T>(handle, uplo, diag, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                         batch_count, (T*)work1, (T*)work3);
            }
        }
    }

    if(diag == rocblas_diagonal_non_unit && blk > 0)
    {
        // restore A in cases where info is nonzero
        hipLaunchKernelGGL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                           stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, tmpcopy, info_mask(info));
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
