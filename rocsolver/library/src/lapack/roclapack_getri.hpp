/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRI_H
#define ROCLAPACK_GETRI_H

#include "rocblas.hpp"
#include "rocblas_device_functions.hpp"
#include "rocsolver.h"

#ifdef OPTIMAL
template <rocblas_int DIM, typename T, typename U>
__global__ void __launch_bounds__(WAVESIZE) getri_kernel_small(U AA,
                                                               const rocblas_int shiftA,
                                                               const rocblas_int lda,
                                                               const rocblas_stride strideA,
                                                               rocblas_int* ipivA,
                                                               const rocblas_int shiftP,
                                                               const rocblas_stride strideP,
                                                               rocblas_int* info)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= DIM)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
    rocblas_int* ipiv = load_ptr_batch<rocblas_int>(ipivA, b, shiftP, strideP);

    // read corresponding row from global memory in local array
    T rA[DIM];
#pragma unroll
    for(int j = 0; j < DIM; ++j)
        rA[j] = A[i + j * lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    __shared__ T diag[DIM];
    __shared__ rocblas_int _info;
    T temp;
    rocblas_int jp;

    // compute info
    if(i == 0)
        _info = 0;
    __syncthreads();
    if(rA[i] == 0)
    {
        rocblas_int _info_temp = _info;
        while(_info_temp == 0 || _info_temp > i + 1)
            _info_temp = atomicCAS(&_info, _info_temp, i + 1);
    }
    __syncthreads();

    if(i == 0)
        info[b] = _info;
    if(_info != 0)
        return;

    //--- TRTRI ---

    // diagonal element
    rA[i] = 1.0 / rA[i];

// compute element i of each column j
#pragma unroll
    for(rocblas_int j = 1; j < DIM; j++)
    {
        // share current column and diagonal
        common[i] = rA[j];
        diag[i] = rA[i];
        __syncthreads();

        if(i < j)
        {
            temp = 0;

            for(rocblas_int ii = i; ii < j; ii++)
                temp += rA[ii] * common[ii];

            rA[j] = -diag[j] * temp;
        }
        __syncthreads();
    }

    //--- GETRI ---

#pragma unroll
    for(rocblas_int j = DIM - 2; j >= 0; j--)
    {
        // extract lower triangular column (copy_and_zero)
        if(i > j)
        {
            common[i] = rA[j];
            rA[j] = 0;
        }
        __syncthreads();

        // update column j (gemv)
        temp = 0;

        for(rocblas_int ii = j + 1; ii < DIM; ii++)
            temp += rA[ii] * common[ii];

        rA[j] -= temp;
        __syncthreads();
    }

// apply pivots (getri_pivot)
#pragma unroll
    for(rocblas_int j = DIM - 2; j >= 0; j--)
    {
        jp = ipiv[j] - 1;
        if(jp != j)
        {
            temp = rA[j];
            rA[j] = rA[jp];
            rA[jp] = temp;
        }
    }

// write results to global memory from local array
#pragma unroll
    for(int j = 0; j < DIM; j++)
        A[i + j * lda] = rA[j];
}

template <typename T, typename U>
rocblas_status getri_run_small(rocblas_handle handle,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count)
{
#define RUN_GETRI_SMALL(DIM)                                                                 \
    hipLaunchKernelGGL((getri_kernel_small<DIM, T>), grid, block, 0, stream, A, shiftA, lda, \
                       strideA, ipiv, shiftP, strideP, info)

    dim3 grid(batch_count, 1, 1);
    dim3 block(WAVESIZE, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch(n)
    {
    case 1: RUN_GETRI_SMALL(1); break;
    case 2: RUN_GETRI_SMALL(2); break;
    case 3: RUN_GETRI_SMALL(3); break;
    case 4: RUN_GETRI_SMALL(4); break;
    case 5: RUN_GETRI_SMALL(5); break;
    case 6: RUN_GETRI_SMALL(6); break;
    case 7: RUN_GETRI_SMALL(7); break;
    case 8: RUN_GETRI_SMALL(8); break;
    case 9: RUN_GETRI_SMALL(9); break;
    case 10: RUN_GETRI_SMALL(10); break;
    case 11: RUN_GETRI_SMALL(11); break;
    case 12: RUN_GETRI_SMALL(12); break;
    case 13: RUN_GETRI_SMALL(13); break;
    case 14: RUN_GETRI_SMALL(14); break;
    case 15: RUN_GETRI_SMALL(15); break;
    case 16: RUN_GETRI_SMALL(16); break;
    case 17: RUN_GETRI_SMALL(17); break;
    case 18: RUN_GETRI_SMALL(18); break;
    case 19: RUN_GETRI_SMALL(19); break;
    case 20: RUN_GETRI_SMALL(20); break;
    case 21: RUN_GETRI_SMALL(21); break;
    case 22: RUN_GETRI_SMALL(22); break;
    case 23: RUN_GETRI_SMALL(23); break;
    case 24: RUN_GETRI_SMALL(24); break;
    case 25: RUN_GETRI_SMALL(25); break;
    case 26: RUN_GETRI_SMALL(26); break;
    case 27: RUN_GETRI_SMALL(27); break;
    case 28: RUN_GETRI_SMALL(28); break;
    case 29: RUN_GETRI_SMALL(29); break;
    case 30: RUN_GETRI_SMALL(30); break;
    case 31: RUN_GETRI_SMALL(31); break;
    case 32: RUN_GETRI_SMALL(32); break;
    case 33: RUN_GETRI_SMALL(33); break;
    case 34: RUN_GETRI_SMALL(34); break;
    case 35: RUN_GETRI_SMALL(35); break;
    case 36: RUN_GETRI_SMALL(36); break;
    case 37: RUN_GETRI_SMALL(37); break;
    case 38: RUN_GETRI_SMALL(38); break;
    case 39: RUN_GETRI_SMALL(39); break;
    case 40: RUN_GETRI_SMALL(40); break;
    case 41: RUN_GETRI_SMALL(41); break;
    case 42: RUN_GETRI_SMALL(42); break;
    case 43: RUN_GETRI_SMALL(43); break;
    case 44: RUN_GETRI_SMALL(44); break;
    case 45: RUN_GETRI_SMALL(45); break;
    case 46: RUN_GETRI_SMALL(46); break;
    case 47: RUN_GETRI_SMALL(47); break;
    case 48: RUN_GETRI_SMALL(48); break;
    case 49: RUN_GETRI_SMALL(49); break;
    case 50: RUN_GETRI_SMALL(50); break;
    case 51: RUN_GETRI_SMALL(51); break;
    case 52: RUN_GETRI_SMALL(52); break;
    case 53: RUN_GETRI_SMALL(53); break;
    case 54: RUN_GETRI_SMALL(54); break;
    case 55: RUN_GETRI_SMALL(55); break;
    case 56: RUN_GETRI_SMALL(56); break;
    case 57: RUN_GETRI_SMALL(57); break;
    case 58: RUN_GETRI_SMALL(58); break;
    case 59: RUN_GETRI_SMALL(59); break;
    case 60: RUN_GETRI_SMALL(60); break;
    case 61: RUN_GETRI_SMALL(61); break;
    case 62: RUN_GETRI_SMALL(62); break;
    case 63: RUN_GETRI_SMALL(63); break;
    case 64: RUN_GETRI_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }

    return rocblas_status_success;
}
#endif // OPTIMAL

template <typename T>
__device__ void copy_and_zero(const rocblas_int m,
                              const rocblas_int n,
                              T* a,
                              const rocblas_int lda,
                              T* w,
                              const rocblas_int ldw)
{
    // Copies the lower triangular part of the matrix to the workspace and then
    // replaces it with zeroes
    int i, j;
    for(int k = hipThreadIdx_y; k < m * n; k += hipBlockDim_y)
    {
        i = k % m;
        j = k / m;
        if(i > j)
        {
            w[i + j * ldw] = a[i + j * lda];
            a[i + j * lda] = 0;
        }
    }
    __syncthreads();
}

template <typename T>
__device__ void zero_work(const rocblas_int m, const rocblas_int n, T* w, const rocblas_int ldw)
{
    // Zeroes the workspace so that calls to gemm and trsm do not alter the matrix
    // (used for singular matrices)
    int i, j;
    for(int k = hipThreadIdx_y; k < m * n; k += hipBlockDim_y)
    {
        i = k % m;
        j = k / m;
        w[i + j * ldw] = 0;
    }
    __syncthreads();
}

template <typename T>
__device__ void getri_pivot(const rocblas_int n, T* a, const rocblas_int lda, rocblas_int* p)
{
    // Applies the pivots specified in ipiv to the inverted matrix
    rocblas_int jp;
    T temp;
    for(rocblas_int j = n - 2; j >= 0; --j)
    {
        jp = p[j] - 1;
        if(jp != j)
        {
            for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
            {
                temp = a[i + j * lda];
                a[i + j * lda] = a[i + jp * lda];
                a[i + jp * lda] = temp;
            }
            __syncthreads();
        }
    }
}

template <typename T, typename U>
__global__ void getri_check_singularity(const rocblas_int n,
                                        U A,
                                        const rocblas_int shifta,
                                        const rocblas_int lda,
                                        const rocblas_stride stridea,
                                        rocblas_int* info)
{
    // Checks for singularities in the matrix and updates info to indicate where
    // the first singularity (if any) occurs
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    __shared__ rocblas_int _info;

    if(hipThreadIdx_y == 0)
        _info = 0;
    __syncthreads();

    for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if(a[i + i * lda] == 0)
        {
            rocblas_int _info_temp = _info;
            while(_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
    }
    __syncthreads();

    if(hipThreadIdx_y == 0)
        info[b] = _info;
}

template <bool COPYALL, bool INPLACE, typename T, typename U, typename V>
__global__ void getri_trtri_update(const rocblas_int n,
                                   U A,
                                   const rocblas_int shifta,
                                   const rocblas_int lda,
                                   const rocblas_stride stridea,
                                   const V W,
                                   const rocblas_int shiftw,
                                   const rocblas_int ldw,
                                   const rocblas_stride stridew,
                                   rocblas_int* info)
{
    // Processes the output of rocblas_trtri so that it matches the output of
    // vanilla LAPACK
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    T* a = load_ptr_batch<T>(A, b, shifta, stridea);
    T* w = load_ptr_batch<T>(W, b, shiftw, stridew);

    // In-place: if A is singular, do not change A; otherwise, copy upper
    // triangular inverse. Out-of-place: if A is singular, restore A; otherwise,
    // copy lower triangular original
    bool copy = COPYALL || (info[b] == 0 && INPLACE && i <= j)
        || (info[b] == 0 && !INPLACE && i > j) || (info[b] != 0 && !INPLACE);
    if(i < n && j < n && copy)
    {
        a[i + j * lda] = w[i + j * ldw];
    }
}

template <typename T, typename U, typename V>
__global__ void getri_kernel(const rocblas_int n,
                             U A,
                             const rocblas_int shiftA,
                             const rocblas_int lda,
                             const rocblas_stride strideA,
                             rocblas_int* ipiv,
                             const rocblas_int shiftP,
                             const rocblas_stride strideP,
                             rocblas_int* info,
                             V work,
                             const rocblas_stride strideW)
{
    // Do-everything getri kernel (excepting the call to trtri) for small- and
    // mid-size matrices
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* w = load_ptr_batch<T>(work, b, 0, strideW);
    rocblas_int* p = load_ptr_batch<rocblas_int>(ipiv, b, shiftP, strideP);

    if(info[b] != 0)
        return;

    T minone = -1;
    T one = 1;
    if(n <= GETRI_SWITCHSIZE_MID)
    {
        // use unblocked version
        for(rocblas_int j = n - 2; j >= 0; --j)
        {
            copy_and_zero(n - j, 1, a + j + j * lda, lda, w + j, n);
            gemv_kernel(n, n - j - 1, &minone, a + (j + 1) * lda, lda, w + j + 1, 1, &one,
                        a + j * lda, 1);
        }
    }
    else
    {
        // use blocked version
        rocblas_int jb, nb = GETRI_BLOCKSIZE;
        rocblas_int nn = ((n - 1) / nb) * nb + 1;
        for(rocblas_int j = nn - 1; j >= 0; j -= nb)
        {
            jb = min(n - j, nb);

            copy_and_zero(n - j, jb, a + j + j * lda, lda, w + j, n);

            if(j + jb < n)
                gemm_kernel(n, jb, n - j - jb, &minone, a + (j + jb) * lda, lda, w + j + jb, n,
                            &one, a + j * lda, lda);

            trsm_kernel_right_lower(rocblas_diagonal_unit, n, jb, &one, w + j, n, a + j * lda, lda);
        }
    }

    getri_pivot(n, a, lda, p);
}

template <typename T, typename U, typename V>
__global__ void getri_kernel_large1(const rocblas_int n,
                                    const rocblas_int j,
                                    const rocblas_int jb,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    rocblas_int* info,
                                    V work,
                                    const rocblas_stride strideW)
{
    // Helper kernel for large-size matrices. Preps the matrix for calls to
    // gemm and trsm.
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* w = load_ptr_batch<T>(work, b, 0, strideW);

    if(info[b] != 0)
        zero_work(n - j, jb, w + j, n);
    else
        copy_and_zero(n - j, jb, a + j + j * lda, lda, w + j, n);
}

template <typename T, typename U>
__global__ void getri_kernel_large2(const rocblas_int n,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    rocblas_int* ipiv,
                                    const rocblas_int shiftP,
                                    const rocblas_stride strideP,
                                    rocblas_int* info)
{
    // Helper kernel for large-size matrices. Applies the pivots to the inverted
    // matrix.
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    rocblas_int* p = load_ptr_batch<rocblas_int>(ipiv, b, shiftP, strideP);

    if(info[b] == 0)
        getri_pivot(n, a, lda, p);
}

template <bool BATCHED, bool INPLACE, typename T>
void rocsolver_getri_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_tmpcopy,
                                   size_t* size_workArr)
{
    // if quick return, no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_tmpcopy = 0;
        *size_workArr = 0;
        return;
    }

#ifdef OPTIMAL
    // if very small size, no workspace needed
    if(n <= WAVESIZE)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_tmpcopy = 0;
        *size_workArr = 0;
        return;
    }
#endif

    // size of scalars for calling rocblas
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    size_t w1a, w1b, w2a, w2b, t1, t2;

    // requirements for calling TRSM
    if(n <= GETRI_SWITCHSIZE_LARGE)
    {
        w1a = 0;
        w2a = 0;
        *size_work3 = 0;
        *size_work4 = 0;
    }
    else
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, n, GETRI_BLOCKSIZE, batch_count, &w1a,
                                         &w2a, size_work3, size_work4);

    // requirements for calling TRTRI
    rocblasCall_trtri_mem<BATCHED, T>(n, batch_count, &w1b, &w2b);

    *size_work1 = max(w1a, w1b);
    *size_work2 = max(w2a, w2b);

    // size of temporary array required for copies
    t1 = (INPLACE ? n * n : 0) * sizeof(T) * batch_count;
    t2 = (n <= GETRI_SWITCHSIZE_MID ? n : n * GETRI_BLOCKSIZE) * sizeof(T) * batch_count;
    *size_tmpcopy = max(t1, t2);
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        rocblas_int* ipiv,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n,
                                        const rocblas_int lda,
                                        const rocblas_int ldc,
                                        T A,
                                        T C,
                                        rocblas_int* ipiv,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || ldc < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !A) || (n && !C) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getri_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U A1,
                                        const rocblas_int shiftA1,
                                        const rocblas_int lda1,
                                        const rocblas_stride strideA1,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_int shiftP,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* tmpcopy,
                                        T** workArr,
                                        bool optim_mem)
{
    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        rocblas_int blocks = (batch_count - 1) / 32 + 1;
        hipLaunchKernelGGL(reset_info, dim3(blocks, 1, 1), dim3(32, 1, 1), 0, stream, info,
                           batch_count, 0);
        return rocblas_status_success;
    }

    rocblas_int blocks = (n - 1) / 32 + 1;
    rocblas_int threads = min(((n - 1) / 64 + 1) * 64, BLOCKSIZE);

#ifdef OPTIMAL
    // if very small size, use optimized inversion kernel
    if(n <= WAVESIZE)
    {
        if(A1 != nullptr)
            hipLaunchKernelGGL((getri_trtri_update<true, false, T>),
                               dim3(batch_count, blocks, blocks), dim3(1, 32, 32), 0, stream, n, A,
                               shiftA, lda, strideA, A1, shiftA1, lda1, strideA1, nullptr);

        return getri_run_small<T>(handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                                  batch_count);
    }
#endif

    rocblas_int ldw = n;
    rocblas_stride strideW = n * n;

    if(A1 == nullptr) // in-place trtri
    {
        // check for singularities
        hipLaunchKernelGGL(getri_check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                           0, stream, n, A, shiftA, lda, strideA, info);

        // compute inv(U)
        rocblasCall_trtri<BATCHED, STRIDED, T>(handle, rocblas_fill_upper, rocblas_diagonal_non_unit,
                                               n, A, shiftA, lda, strideA, tmpcopy, 0, ldw, strideW,
                                               batch_count, (T*)work1, (T**)work2, workArr);

        // copy inv(U) to A
        hipLaunchKernelGGL((getri_trtri_update<false, true, T>), dim3(batch_count, blocks, blocks),
                           dim3(1, 32, 32), 0, stream, n, A, shiftA, lda, strideA, tmpcopy, 0, ldw,
                           strideW, info);
    }
    else // out-of-place trtri
    {
        // check for singularities
        hipLaunchKernelGGL(getri_check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                           0, stream, n, A1, shiftA1, lda1, strideA1, info);

        // compute inv(U)
        rocblasCall_trtri<BATCHED, STRIDED, T>(handle, rocblas_fill_upper, rocblas_diagonal_non_unit,
                                               n, A1, shiftA1, lda1, strideA1, A, shiftA, lda,
                                               strideA, batch_count, (T*)work1, (T**)work2, workArr);

        // restore lower triangular part of A
        hipLaunchKernelGGL((getri_trtri_update<false, false, T>), dim3(batch_count, blocks, blocks),
                           dim3(1, 32, 32), 0, stream, n, A, shiftA, lda, strideA, A1, shiftA1,
                           lda1, strideA1, info);
    }

    // at this point: if A is singular, then it contains the original triangular
    // factors L and U (i.e. it is unmodified); otherwise, it contains L and
    // inv(U)

    strideW = (n <= GETRI_SWITCHSIZE_MID ? n : n * GETRI_BLOCKSIZE);
    if(n <= GETRI_SWITCHSIZE_LARGE)
    {
        hipLaunchKernelGGL(getri_kernel<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1), 0, stream,
                           n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, tmpcopy, strideW);
    }
    else
    {
        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

        T minone = -1;
        T one = 1;
        rocblas_int jb, nb = GETRI_BLOCKSIZE;

        rocblas_int nn = ((n - 1) / nb) * nb + 1;
        for(rocblas_int j = nn - 1; j >= 0; j -= nb)
        {
            jb = min(n - j, nb);

            hipLaunchKernelGGL(getri_kernel_large1<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                               0, stream, n, j, jb, A, shiftA, lda, strideA, info, tmpcopy, strideW);

            if(j + jb < n)
                rocblasCall_gemm<BATCHED, STRIDED>(
                    handle, rocblas_operation_none, rocblas_operation_none, n, jb, n - j - jb,
                    &minone, A, shiftA + idx2D(0, j + jb, lda), lda, strideA, tmpcopy, j + jb, ldw,
                    strideW, &one, A, shiftA + idx2D(0, j, lda), lda, strideA, batch_count, workArr);

            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_right, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_unit, n, jb, &one,
                                         tmpcopy, j, ldw, strideW, A, shiftA + idx2D(0, j, lda),
                                         lda, strideA, batch_count, optim_mem, work1, work2, work3,
                                         work4, workArr);
        }

        hipLaunchKernelGGL(getri_kernel_large2<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1), 0,
                           stream, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info);

        rocblas_set_pointer_mode(handle, old_mode);
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRI_H */
