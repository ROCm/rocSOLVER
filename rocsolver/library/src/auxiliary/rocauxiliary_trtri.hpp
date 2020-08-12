/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_TRTRI_H
#define ROCLAPACK_TRTRI_H

#include "rocblas.hpp"
#include "rocblas_device_functions.hpp"
#include "rocsolver.h"

#ifdef OPTIMAL
template <rocblas_int DIM, typename T>
__device__ rocblas_int trtri_impl_small_upper(const rocblas_diagonal diag, T *rA, T *common, T *common_diag,
                                              rocblas_int* info)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    // shared memory (for communication between threads in group)
    __shared__ rocblas_int _info;
    T temp;
    
    // compute info
    if (i == 0)
        _info = 0;
    __syncthreads();
    if (rA[i] == 0)
    {
        rocblas_int _info_temp = _info;
        while (_info_temp == 0 || _info_temp > i + 1)
            _info_temp = atomicCAS(&_info, _info_temp, i + 1);
    }
    __syncthreads();

    if (i == 0)
        info[b] = _info;
    if (_info != 0)
        return _info;
    
    // diagonal element
    common_diag[i] = (diag == rocblas_diagonal_non_unit ? (rA[i] = 1.0 / rA[i]) : 1);
    
    // compute element i of each column j
    #pragma unroll
    for (rocblas_int j = 1; j < DIM; j++)
    {
        // share current column
        common[i] = rA[j];
        __syncthreads();
        
        if (i < j)
        {
            temp = (diag == rocblas_diagonal_non_unit ? rA[i] : 1) * common[i];

            for (rocblas_int ii = i+1; ii < j; ii++)
                temp += rA[ii] * common[ii];

            rA[j] = -common_diag[j] * temp;
        }
        __syncthreads();
    }

    return _info;
}
template <rocblas_int DIM, typename T>
__device__ rocblas_int trtri_impl_small_lower(const rocblas_diagonal diag, T *rA, T *common, T *common_diag,
                                              rocblas_int* info)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    // shared memory (for communication between threads in group)
    __shared__ rocblas_int _info;
    T temp;
    
    // compute info
    if (i == 0)
        _info = 0;
    __syncthreads();
    if (rA[i] == 0)
    {
        rocblas_int _info_temp = _info;
        while (_info_temp == 0 || _info_temp > i + 1)
            _info_temp = atomicCAS(&_info, _info_temp, i + 1);
    }
    __syncthreads();

    if (i == 0)
        info[b] = _info;
    if (_info != 0)
        return _info;
    
    // diagonal element
    common_diag[i] = (diag == rocblas_diagonal_non_unit ? (rA[i] = 1.0 / rA[i]) : 1);
    
    // compute element i of each column j
    #pragma unroll
    for (rocblas_int j = DIM-1; j >= 0; j--)
    {
        // share current column
        common[i] = rA[j];
        __syncthreads();
        
        if (i > j)
        {
            temp = (diag == rocblas_diagonal_non_unit ? rA[i] : 1) * common[i];

            for (rocblas_int ii = i-1; ii > j; ii--)
                temp += rA[ii] * common[ii];

            rA[j] = -common_diag[j] * temp;
        }
        __syncthreads();
    }

    return _info;
}

template <rocblas_int DIM, typename T, typename U>
__global__ void __launch_bounds__(WAVESIZE)
trtri_kernel_small(const rocblas_fill uplo, const rocblas_diagonal diag, U AA, const rocblas_int shiftA,
                   const rocblas_int lda, const rocblas_stride strideA, rocblas_int* info)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if (i >= DIM)
        return;
    
    // batch instance
    T* A = load_ptr_batch<T>(AA,b,shiftA,strideA);
       
    // read corresponding row from global memory in local array
    T rA[DIM];
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        rA[j] = A[i + j*lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    __shared__ T common_diag[DIM];
    
    rocblas_int _info = (uplo == rocblas_fill_upper ?
        trtri_impl_small_upper<DIM,T>(diag, rA, common, common_diag, info):
        trtri_impl_small_lower<DIM,T>(diag, rA, common, common_diag, info));
    if (_info != 0)
        return;

    // write results to global memory from local array
    #pragma unroll
    for (int j = 0; j < DIM; j++)
        A[i + j*lda] = rA[j];
}

template <typename T, typename U>
rocblas_status trtri_run_small(rocblas_handle handle, const rocblas_fill uplo, const rocblas_diagonal diag,
                               const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                               const rocblas_stride strideA, rocblas_int* info, const rocblas_int batch_count)
{
    #define RUN_TRTRI_SMALL(DIM)                                                         \
        hipLaunchKernelGGL((trtri_kernel_small<DIM,T>), grid, block, 0, stream,          \
                           uplo, diag, A, shiftA, lda, strideA, info)
    
    dim3 grid(batch_count,1,1);
    dim3 block(WAVESIZE,1,1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch (n) {
        case  1: RUN_TRTRI_SMALL( 1); break;
        case  2: RUN_TRTRI_SMALL( 2); break;
        case  3: RUN_TRTRI_SMALL( 3); break;
        case  4: RUN_TRTRI_SMALL( 4); break;
        case  5: RUN_TRTRI_SMALL( 5); break;
        case  6: RUN_TRTRI_SMALL( 6); break;
        case  7: RUN_TRTRI_SMALL( 7); break;
        case  8: RUN_TRTRI_SMALL( 8); break;
        case  9: RUN_TRTRI_SMALL( 9); break;
        case 10: RUN_TRTRI_SMALL(10); break;
        case 11: RUN_TRTRI_SMALL(11); break;
        case 12: RUN_TRTRI_SMALL(12); break;
        case 13: RUN_TRTRI_SMALL(13); break;
        case 14: RUN_TRTRI_SMALL(14); break;
        case 15: RUN_TRTRI_SMALL(15); break;
        case 16: RUN_TRTRI_SMALL(16); break;
        case 17: RUN_TRTRI_SMALL(17); break;
        case 18: RUN_TRTRI_SMALL(18); break;
        case 19: RUN_TRTRI_SMALL(19); break;
        case 20: RUN_TRTRI_SMALL(20); break;
        case 21: RUN_TRTRI_SMALL(21); break;
        case 22: RUN_TRTRI_SMALL(22); break;
        case 23: RUN_TRTRI_SMALL(23); break;
        case 24: RUN_TRTRI_SMALL(24); break;
        case 25: RUN_TRTRI_SMALL(25); break;
        case 26: RUN_TRTRI_SMALL(26); break;
        case 27: RUN_TRTRI_SMALL(27); break;
        case 28: RUN_TRTRI_SMALL(28); break;
        case 29: RUN_TRTRI_SMALL(29); break;
        case 30: RUN_TRTRI_SMALL(30); break;
        case 31: RUN_TRTRI_SMALL(31); break;
        case 32: RUN_TRTRI_SMALL(32); break;
        case 33: RUN_TRTRI_SMALL(33); break;
        case 34: RUN_TRTRI_SMALL(34); break;
        case 35: RUN_TRTRI_SMALL(35); break;
        case 36: RUN_TRTRI_SMALL(36); break;
        case 37: RUN_TRTRI_SMALL(37); break;
        case 38: RUN_TRTRI_SMALL(38); break;
        case 39: RUN_TRTRI_SMALL(39); break;
        case 40: RUN_TRTRI_SMALL(40); break;
        case 41: RUN_TRTRI_SMALL(41); break;
        case 42: RUN_TRTRI_SMALL(42); break;
        case 43: RUN_TRTRI_SMALL(43); break;
        case 44: RUN_TRTRI_SMALL(44); break;
        case 45: RUN_TRTRI_SMALL(45); break;
        case 46: RUN_TRTRI_SMALL(46); break;
        case 47: RUN_TRTRI_SMALL(47); break;
        case 48: RUN_TRTRI_SMALL(48); break;
        case 49: RUN_TRTRI_SMALL(49); break;
        case 50: RUN_TRTRI_SMALL(50); break;
        case 51: RUN_TRTRI_SMALL(51); break;
        case 52: RUN_TRTRI_SMALL(52); break;
        case 53: RUN_TRTRI_SMALL(53); break;
        case 54: RUN_TRTRI_SMALL(54); break;
        case 55: RUN_TRTRI_SMALL(55); break;
        case 56: RUN_TRTRI_SMALL(56); break;
        case 57: RUN_TRTRI_SMALL(57); break;
        case 58: RUN_TRTRI_SMALL(58); break;
        case 59: RUN_TRTRI_SMALL(59); break;
        case 60: RUN_TRTRI_SMALL(60); break;
        case 61: RUN_TRTRI_SMALL(61); break;
        case 62: RUN_TRTRI_SMALL(62); break;
        case 63: RUN_TRTRI_SMALL(63); break;
        case 64: RUN_TRTRI_SMALL(64); break;
        default: __builtin_unreachable();
    }
    
    return rocblas_status_success;
}
#endif //OPTIMAL

template <typename T>
__device__ void trtri_check_singularity(const rocblas_diagonal diag, const rocblas_int n, T *a,
                                        const rocblas_int lda, rocblas_int *info)
{
    // check for singularities
    int b = hipBlockIdx_x;

    if (diag == rocblas_diagonal_unit)
    {
        if (hipThreadIdx_x == 0)
            info[b] = 0;
        __syncthreads();
        return;
    }

    __shared__ rocblas_int _info;
    
    // compute info
    if (hipThreadIdx_y == 0)
        _info = 0;
    __syncthreads();
    for (int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if (a[i + i * lda] == 0)
        {
            rocblas_int _info_temp = _info;
            while (_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
    }
    __syncthreads();

    if (hipThreadIdx_y == 0)
        info[b] = _info;
    __syncthreads();
}

template <typename T>
__device__ void trtri_unblk_upper(const rocblas_diagonal diag, const rocblas_int n, T *a, const rocblas_int lda,
                                  rocblas_int *info, T *w)
{
    // unblocked trtri kernel assuming upper triangular matrix
    int i = hipThreadIdx_y;
    if (i >= n)
        return;

    // diagonal element
    if (diag == rocblas_diagonal_non_unit)
    {
        a[i + i * lda] = 1.0 / a[i + i * lda];
        __syncthreads();
    }
    
    // compute element i of each column j
    T ajj, aij;
    for (rocblas_int j = 1; j < n; j++)
    {
        ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);

        if (i < j)
            w[i] = a[i + j * lda];
        __syncthreads();
        
        if (i < j)
        {
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for (rocblas_int ii = i+1; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trtri_unblk_lower(const rocblas_diagonal diag, const rocblas_int n, T *a, const rocblas_int lda,
                                  rocblas_int *info, T *w)
{
    // unblocked trtri kernel assuming lower triangular matrix
    int i = hipThreadIdx_y;
    if (i >= n)
        return;

    // diagonal element
    if (diag == rocblas_diagonal_non_unit)
    {
        a[i + i * lda] = 1.0 / a[i + i * lda];
        __syncthreads();
    }
    
    // compute element i of each column j
    T ajj, aij;
    for (rocblas_int j = n-2; j >= 0; j--)
    {
        ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);

        if (i > j)
            w[i] = a[i + j * lda];
        __syncthreads();
        
        if (i > j)
        {
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for (rocblas_int ii = i-1; ii > j; ii--)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}


template <typename T, typename U, typename V>
__global__ void trtri_kernel_upper(const rocblas_diagonal diag, const rocblas_int n,
                                   U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                   rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    trtri_check_singularity(diag, n, a, lda, info);
    if (info[b] != 0)
        return;

    if (n <= TRTRI_SWITCHSIZE_MID)
        // use unblocked version
        trtri_unblk_upper(diag, n, a, lda, info, w);
    else
    {
        // use blocked version
        T minone = -1;
        T one = 1;
        rocblas_int jb, nb = TRTRI_BLOCKSIZE;
        
        for (rocblas_int j = 0; j < n; j += nb)
        {
            jb = min(n-j, nb);

            trmm_kernel_left_upper(diag, j, jb, &one, a, lda, a + j*lda, lda, w);
            trsm_kernel_right_upper(diag, j, jb, &minone, a + j+j*lda, lda, a + j*lda, lda);
            trtri_unblk_upper(diag, jb, a + j+j*lda, lda, info, w);
        }
    }
}

template <typename T, typename U, typename V>
__global__ void trtri_kernel_lower(const rocblas_diagonal diag, const rocblas_int n,
                                   U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                   rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    trtri_check_singularity(diag, n, a, lda, info);
    if (info[b] != 0)
        return;

    if (n <= TRTRI_SWITCHSIZE_MID)
        // use unblocked version
        trtri_unblk_lower(diag, n, a, lda, info, w);
    else
    {
        // use blocked version
        T minone = -1;
        T one = 1;
        rocblas_int jb, nb = TRTRI_BLOCKSIZE;
        
        rocblas_int nn = ((n - 1)/nb)*nb + 1;
        for (rocblas_int j = nn-1; j >= 0; j -= nb)
        {
            jb = min(n-j, nb);

            trmm_kernel_left_lower(diag, n-j-jb, jb, &one, a + (j+jb)+(j+jb)*lda, lda, a + (j+jb)+j*lda, lda, w);
            trsm_kernel_right_lower(diag, n-j-jb, jb, &minone, a + j+j*lda, lda, a + (j+jb)+j*lda, lda);
            trtri_unblk_lower(diag, jb, a + j+j*lda, lda, info, w);
        }
    }
}

template <typename T, typename U, typename V>
__global__ void trtri_kernel_large_upper(const rocblas_diagonal diag, const rocblas_int n, const rocblas_int j, const rocblas_int jb,
                                         U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                         rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n * TRTRI_BLOCKSIZE;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    if (j == 0)
        trtri_check_singularity(diag, n, a, lda, info);
    
    if (info[b] != 0)
    {
        // if A is singular, we want it to remain unaltered by trmm
        int idx = hipThreadIdx_y;

        int jj = j + idx;
        if (j > 0 && jj < n)
        {
            // restore original entries of A
            for (int i = 0; i < j; i++)
                a[i + jj * lda] = w[i + idx * n];
        }

        jj = j + TRTRI_BLOCKSIZE + idx;
        if (jj < n)
        {
            // save original entries of A
            for (int i = 0; i < j + TRTRI_BLOCKSIZE; i++)
                w[i + idx * n] = a[i + jj * lda];
        }
    }
    else
    {
        T minone = -1;
        trsm_kernel_right_upper(diag, j, jb, &minone, a + j+j*lda, lda, a + j*lda, lda);
        trtri_unblk_upper(diag, jb, a + j+j*lda, lda, info, w);
    }
}

template <typename T, typename U, typename V>
__global__ void trtri_kernel_large_lower(const rocblas_diagonal diag, const rocblas_int n, const rocblas_int j, const rocblas_int jb,
                                         U A, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                                         rocblas_int *info, V work)
{
    int b = hipBlockIdx_x;

    rocblas_stride strideW = n * TRTRI_BLOCKSIZE;
    T* a = load_ptr_batch<T>(A,b,shiftA,strideA);
    T* w = load_ptr_batch<T>(work,b,0,strideW);

    if (j == 0)
        trtri_check_singularity(diag, n, a, lda, info);
    
    if (info[b] != 0)
    {
        // if A is singular, we want it to remain unaltered by trmm
        int idx = hipThreadIdx_y;

        int jj = j + idx;
        if (j+jb < n && jj < n)
        {
            // restore original entries of A
            for (int i = n-1; i > j; i--)
                a[i + jj * lda] = w[i + idx * n];
        }

        jj = j - TRTRI_BLOCKSIZE + idx;
        if (jj >= 0)
        {
            // save original entries of A
            for (int i = n-1; i > j - TRTRI_BLOCKSIZE; i--)
                w[i + idx * n] = a[i + jj * lda];
        }
    }
    else
    {
        T minone = -1;
        trsm_kernel_right_lower(diag, n-j-jb, jb, &minone, a + j+j*lda, lda, a + (j+jb)+j*lda, lda);
        trtri_unblk_lower(diag, jb, a + j+j*lda, lda, info, w);
    }
}


template <bool BATCHED, typename T>
void rocsolver_trtri_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    // for scalars
    *size_1 = sizeof(T)*3;

    // for workspace
    if (n <= TRTRI_SWITCHSIZE_LARGE)
        *size_2 = n;
    else
        *size_2 = n * TRTRI_BLOCKSIZE + 2 * ROCBLAS_TRMM_NB * ROCBLAS_TRMM_NB;
    *size_2 *= sizeof(T)*batch_count;

    // size of array of pointers to workspace
    if (BATCHED)
        *size_3 = 3 * sizeof(T*) * batch_count;
    else
        *size_3 = 0;
}

template <typename T>
rocblas_status rocsolver_trtri_argCheck(const rocblas_fill uplo, const rocblas_diagonal diag, const rocblas_int n,
                                        const rocblas_int lda, T A, rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if (diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;
    
    // 2. invalid size
    if (n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trtri_template(rocblas_handle handle, const rocblas_fill uplo, const rocblas_diagonal diag,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                                        const rocblas_stride strideA, rocblas_int *info,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr)
{
    // quick return if zero instances in batch
    if (batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if (n == 0)
    {
        rocblas_int blocks = (batch_count - 1)/32 + 1;
        hipLaunchKernelGGL(reset_info, dim3(blocks,1,1), dim3(32,1,1), 0, stream,
                           info, batch_count, 0);
        return rocblas_status_success;
    }
    
    #ifdef OPTIMAL
    // if very small size, use optimized inversion kernel
    if (n <= WAVESIZE)
        return trtri_run_small<T>(handle,uplo,diag,n,A,shiftA,lda,strideA,info,batch_count);

    #endif

    rocblas_int threads = min(((n - 1)/64 + 1) * 64, TRTRI_BLOCKSIZE);
    
    if (n <= TRTRI_SWITCHSIZE_LARGE)
    {
        if (uplo == rocblas_fill_upper)
            hipLaunchKernelGGL(trtri_kernel_upper<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                               diag, n, A, shiftA, lda, strideA, info, work);
        else
            hipLaunchKernelGGL(trtri_kernel_lower<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                               diag, n, A, shiftA, lda, strideA, info, work);
    }
    else
    {
        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle,&old_mode);
        rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

        T one = 1;
        rocblas_int jb, nb = TRTRI_BLOCKSIZE;
        rocblas_int shiftW = batch_count * n * TRTRI_BLOCKSIZE;

        if (uplo == rocblas_fill_upper)
        {
            for (rocblas_int j = 0; j < n; j += nb)
            {
                jb = min(n-j, nb);
                
                rocblasCall_trmm<BATCHED,STRIDED,T>(handle, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                                                    diag, j, jb, &one, A, shiftA, lda, strideA,
                                                    A, shiftA + idx2D(0,j,lda), lda, strideA, batch_count, work + shiftW, workArr);

                hipLaunchKernelGGL(trtri_kernel_large_upper<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                                diag, n, j, jb, A, shiftA, lda, strideA, info, work);
            }
        }
        else
        {
            rocblas_int nn = ((n - 1)/nb)*nb + 1;
            for (rocblas_int j = nn-1; j >= 0; j -= nb)
            {
                jb = min(n-j, nb);
                
                rocblasCall_trmm<BATCHED,STRIDED,T>(handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                                                    diag, n-j-jb, jb, &one, A, shiftA + idx2D(j+jb,j+jb,lda), lda, strideA,
                                                    A, shiftA + idx2D(j+jb,j,lda), lda, strideA, batch_count, work + shiftW, workArr);

                hipLaunchKernelGGL(trtri_kernel_large_lower<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                                diag, n, j, jb, A, shiftA, lda, strideA, info, work);
            }
        }
        

        rocblas_set_pointer_mode(handle,old_mode);
    }

    return rocblas_status_success;
}

#endif /* ROCLAPACK_TRTRI_H */
