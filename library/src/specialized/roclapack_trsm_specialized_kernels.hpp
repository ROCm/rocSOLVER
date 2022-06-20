/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

/** Constants for block size of trsm **/
// clang-format off
#define TRSM_NUMROWS_REAL 12
#define TRSM_NUMCOLS_REAL 16
#define TRSM_INTERVALSROW_REAL                                          \
    40, 56, 80, 112, 144, 176, 208, 240, 288, 352, 480
#define TRSM_INTERVALSCOL_REAL                                          \
    448, 768, 960, 1152, 1408, 1920, 2304, 2816, 3840, 4096, 4736,      \
    4992, 5888, 7680, 9728
#define TRSM_BLKSIZES_REAL                                              \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 24, 24, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1, 32, 32, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1, 48, 48, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1, 80, 80, 80, 56, 56, 40, 40, 40, 32, 32, 32},    \
    {1,  1,  1,  1, 80, 80, 80, 80, 80, 48, 48, 48, 40, 32,  0,  0},    \
    {1,  1,  1, 80, 80, 80, 80, 80, 56, 56, 32, 32, 32, 32,  0,  0},    \
    {1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}

#define TRSM_NUMROWS_COMPLEX 10
#define TRSM_NUMCOLS_COMPLEX 12
#define TRSM_INTERVALSROW_COMPLEX                                       \
    40, 56, 80, 112, 144, 208, 240, 288, 480
#define TRSM_INTERVALSCOL_COMPLEX                                       \
    704, 960, 1344, 1920, 2304, 2816, 3200, 3840, 4864, 5888, 7680
#define TRSM_BLKSIZES_COMPLEX                                           \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                               \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 24, 24},                            \
    {1, 1, 1, 1, 1, 1, 1, 1, 32, 32, 32, 32},                           \
    {1, 1, 1, 1, 1, 72, 72, 56, 48, 32, 32, 32},                        \
    {1, 1, 1, 1, 64, 64, 64, 64, 48, 32, 32, 32},                       \
    {1, 1, 1, 80, 80, 80, 64, 64, 48, 32, 32, 32},                      \
    {1, 1, 80, 80, 80, 80, 64, 64, 40, 40, 32, 32},                     \
    {1, 1, 72, 72, 64, 64, 64, 64, 32, 32, 32, 0},                      \
    {1, 80, 80, 80, 80, 80, 64, 64, 48, 40, 32, 0},                     \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define TRSM_BATCH_NUMROWS_REAL 11
#define TRSM_BATCH_NUMCOLS_REAL 17
#define TRSM_BATCH_INTERVALSROW_REAL                                        \
    20, 28, 40, 80, 112, 176, 208, 288, 352, 480
#define TRSM_BATCH_INTERVALSCOL_REAL                                        \
    6, 10, 12, 22, 28, 30, 36, 42, 46, 50, 60, 96, 432, 928, 960, 1472
#define TRSM_BATCH_BLKSIZES_REAL                                            \
    { 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1, 24, 24, 24, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},   \
    {48, 48, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 16, 16, 16, 16, 16},   \
    {64, 64, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 24, 24, 24, 24, 24},   \
    {64, 64, 32, 32, 24, 24, 24, 24, 32, 32, 32, 32, 32, 24, 24, 24, 24},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 40, 40, 40, 32, 32, 24, 24, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32,  0},   \
    {64, 64, 64, 32, 32, 32, 48, 48, 48, 48, 40, 32, 32, 32,  0,  0,  0}

#define TRSM_BATCH_NUMROWS_COMPLEX 10
#define TRSM_BATCH_NUMCOLS_COMPLEX 16
#define TRSM_BATCH_INTERVALSROW_COMPLEX                                     \
    20, 28, 40, 56, 80, 112, 144, 176, 480
#define TRSM_BATCH_INTERVALSCOL_COMPLEX                                     \
    4, 12, 16, 28, 32, 40, 48, 50, 60, 72, 88, 176, 232, 400, 464
#define TRSM_BATCH_BLKSIZES_COMPLEX                                         \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  8,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1, 16, 16, 16, 16,  1,  1,  1, 16, 16, 16, 16, 16},        \
    {1,  1,  1, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},        \
    {1,  1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 48, 48, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 80, 80, 56, 56, 32, 32},        \
    {1, 64, 32, 32, 32, 64, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32},        \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 64, 64, 48, 48, 48}
// clang-format on

/** Forward and Backward subtitution kernel launchers **/
#define FORWARD_SUBSTITUTIONS                                                                     \
    if(conj)                                                                                      \
    {                                                                                             \
        if(isunit)                                                                                \
            ROCSOLVER_LAUNCH_KERNEL(conj_unit_forward_substitution_kernel<T>, grid, threads,      \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,       \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);              \
        else                                                                                      \
            ROCSOLVER_LAUNCH_KERNEL(conj_nonunit_forward_substitution_kernel<T>, grid, threads,   \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,       \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);              \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        if(isunit)                                                                                \
            ROCSOLVER_LAUNCH_KERNEL(unit_forward_substitution_kernel<T>, grid, threads, lmemsize, \
                                    stream, nx, ny, A, lda1, lda2, shiftA + offA, strideA, B,     \
                                    ldb1, ldb2, shiftB + offB, strideB);                          \
        else                                                                                      \
            ROCSOLVER_LAUNCH_KERNEL(nonunit_forward_substitution_kernel<T>, grid, threads,        \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,       \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);              \
    }

#define BACKWARD_SUBSTITUTIONS                                                                     \
    if(conj)                                                                                       \
    {                                                                                              \
        if(isunit)                                                                                 \
            ROCSOLVER_LAUNCH_KERNEL(conj_unit_backward_substitution_kernel<T>, grid, threads,      \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,        \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);               \
        else                                                                                       \
            ROCSOLVER_LAUNCH_KERNEL(conj_nonunit_backward_substitution_kernel<T>, grid, threads,   \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,        \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);               \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(isunit)                                                                                 \
            ROCSOLVER_LAUNCH_KERNEL(unit_backward_substitution_kernel<T>, grid, threads, lmemsize, \
                                    stream, nx, ny, A, lda1, lda2, shiftA + offA, strideA, B,      \
                                    ldb1, ldb2, shiftB + offB, strideB);                           \
        else                                                                                       \
            ROCSOLVER_LAUNCH_KERNEL(nonunit_backward_substitution_kernel<T>, grid, threads,        \
                                    lmemsize, stream, nx, ny, A, lda1, lda2, shiftA + offA,        \
                                    strideA, B, ldb1, ldb2, shiftB + offB, strideB);               \
    }

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

// **************** forward substitution kernels ************************//
///////////////////////////////////////////////////////////////////////////
/** The following kernels implement forward substitution for lower triangular L
    or upper triangular U matrices in the form
    LX = B
    U'X = B
    B = XU
    B = XL'

    nx is the number of variables and ny the number of right/left-hand-sides.
    Whether B is accessed by rows (left-hand-sides) or columns (right-hand-sides) is
    determined by the values of ldb1 and ldb2. Whether L/U is transposed or not is
    determined by the values of lda1 and lda2.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in y to cover all the 'ny' right/left-hand-sides (columns/rows of B).
    There should be only one group in x with hipBlockDim_x = nx.
    Size of shared memory per group should be:
    lmemsize = hipBlockDim_y * sizeof(T);

    There are 4 different forward substitution kernels; each one deals with
    a combination of unit and conjugate. In the non-unit case, the kernels DO NOT
    verify whether the diagonal element of L/U is non-zero.**/
template <typename T, typename U>
ROCSOLVER_KERNEL void unit_forward_substitution_kernel(const rocblas_int nx,
                                                       const rocblas_int ny,
                                                       U AA,
                                                       const rocblas_int lda1,
                                                       const rocblas_int lda2,
                                                       const rocblas_int shiftA,
                                                       const rocblas_stride strideA,
                                                       U BB,
                                                       const rocblas_int ldb1,
                                                       const rocblas_int ldb2,
                                                       const rocblas_int shiftB,
                                                       const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x > k) ? A[ida + k * lda2] * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_unit_forward_substitution_kernel(const rocblas_int nx,
                                                            const rocblas_int ny,
                                                            U AA,
                                                            const rocblas_int lda1,
                                                            const rocblas_int lda2,
                                                            const rocblas_int shiftA,
                                                            const rocblas_stride strideA,
                                                            U BB,
                                                            const rocblas_int ldb1,
                                                            const rocblas_int ldb2,
                                                            const rocblas_int shiftB,
                                                            const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x > k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void nonunit_forward_substitution_kernel(const rocblas_int nx,
                                                          const rocblas_int ny,
                                                          U AA,
                                                          const rocblas_int lda1,
                                                          const rocblas_int lda2,
                                                          const rocblas_int shiftA,
                                                          const rocblas_stride strideA,
                                                          U BB,
                                                          const rocblas_int ldb1,
                                                          const rocblas_int ldb2,
                                                          const rocblas_int shiftB,
                                                          const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / A[x * (lda1 + lda2)];
                b[ty] = c;
            }
            __syncthreads();

            c -= (x > k) ? A[ida + k * lda2] * b[ty] : 0;
        }
        if(x == nx - 1)
            c = c / A[x * (lda1 + lda2)];

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_nonunit_forward_substitution_kernel(const rocblas_int nx,
                                                               const rocblas_int ny,
                                                               U AA,
                                                               const rocblas_int lda1,
                                                               const rocblas_int lda2,
                                                               const rocblas_int shiftA,
                                                               const rocblas_stride strideA,
                                                               U BB,
                                                               const rocblas_int ldb1,
                                                               const rocblas_int ldb2,
                                                               const rocblas_int shiftB,
                                                               const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / conj(A[x * (lda1 + lda2)]);
                b[ty] = c;
            }
            __syncthreads();

            c -= (x > k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }
        if(x == nx - 1)
            c = c / conj(A[x * (lda1 + lda2)]);

        // move results back to global
        B[idb] = c;
    }
}

// **************** backward substitution kernels ************************//
////////////////////////////////////////////////////////////////////////////
/** The following kernels implement backward substitution for lower triangular L
    or upper triangular U matrices in the form
    L'X = B
    UX = B
    B = XU'
    B = XL

    nx is the number of variables and ny the number of right/left-hand-sides.
    Whether B is accessed by rows (left-hand-sides) or columns (right-hand-sides) is
    determined by the values of ldb1 and ldb2. Whether L/U is transposed or not is
    determined by the values of lda1 and lda2.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in y to cover all the 'ny' right/left-hand-sides (columns/rows of B).
    There should be only one group in x with hipBlockDim_x = nx.
    Size of shared memory per group should be:
    lmemsize = hipBlockDim_y * sizeof(T);

    There are 4 different backward substitution kernels; each one deals with
    a combination of unit and conjugate. In the non-unit case, the kernels DO NOT
    verify whether the diagonal element of L/U is non-zero.**/
template <typename T, typename U>
ROCSOLVER_KERNEL void unit_backward_substitution_kernel(const rocblas_int nx,
                                                        const rocblas_int ny,
                                                        U AA,
                                                        const rocblas_int lda1,
                                                        const rocblas_int lda2,
                                                        const rocblas_int shiftA,
                                                        const rocblas_stride strideA,
                                                        U BB,
                                                        const rocblas_int ldb1,
                                                        const rocblas_int ldb2,
                                                        const rocblas_int shiftB,
                                                        const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x < k) ? A[ida + k * lda2] * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_unit_backward_substitution_kernel(const rocblas_int nx,
                                                             const rocblas_int ny,
                                                             U AA,
                                                             const rocblas_int lda1,
                                                             const rocblas_int lda2,
                                                             const rocblas_int shiftA,
                                                             const rocblas_stride strideA,
                                                             U BB,
                                                             const rocblas_int ldb1,
                                                             const rocblas_int ldb2,
                                                             const rocblas_int shiftB,
                                                             const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x < k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void nonunit_backward_substitution_kernel(const rocblas_int nx,
                                                           const rocblas_int ny,
                                                           U AA,
                                                           const rocblas_int lda1,
                                                           const rocblas_int lda2,
                                                           const rocblas_int shiftA,
                                                           const rocblas_stride strideA,
                                                           U BB,
                                                           const rocblas_int ldb1,
                                                           const rocblas_int ldb2,
                                                           const rocblas_int shiftB,
                                                           const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / A[x * (lda1 + lda2)];
                b[ty] = c;
            }
            __syncthreads();

            c -= (x < k) ? A[ida + k * lda2] * b[ty] : 0;
        }
        if(x == 0)
            c = c / A[x * (lda1 + lda2)];

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_nonunit_backward_substitution_kernel(const rocblas_int nx,
                                                                const rocblas_int ny,
                                                                U AA,
                                                                const rocblas_int lda1,
                                                                const rocblas_int lda2,
                                                                const rocblas_int shiftA,
                                                                const rocblas_stride strideA,
                                                                U BB,
                                                                const rocblas_int ldb1,
                                                                const rocblas_int ldb2,
                                                                const rocblas_int shiftB,
                                                                const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / conj(A[x * (lda1 + lda2)]);
                b[ty] = c;
            }
            __syncthreads();

            c -= (x < k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }
        if(x == 0)
            c = c / conj(A[x * (lda1 + lda2)]);

        // move results back to global
        B[idb] = c;
    }
}

/*************************************************************
    Launchers of specilized  kernels
*************************************************************/

/** This function returns the block size for the internal
    (blocked) trsm implementation **/
template <bool ISBATCHED, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_int rocsolver_trsm_blksize(const rocblas_int m, const rocblas_int n)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int M = TRSM_BATCH_NUMROWS_REAL - 1;
        rocblas_int N = TRSM_BATCH_NUMCOLS_REAL - 1;
        rocblas_int intervalsM[] = {TRSM_BATCH_INTERVALSROW_REAL};
        rocblas_int intervalsN[] = {TRSM_BATCH_INTERVALSCOL_REAL};
        rocblas_int size[][TRSM_BATCH_NUMCOLS_REAL] = {TRSM_BATCH_BLKSIZES_REAL};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }
    else
    {
        rocblas_int M = TRSM_NUMROWS_REAL - 1;
        rocblas_int N = TRSM_NUMCOLS_REAL - 1;
        rocblas_int intervalsM[] = {TRSM_INTERVALSROW_REAL};
        rocblas_int intervalsN[] = {TRSM_INTERVALSCOL_REAL};
        rocblas_int size[][TRSM_NUMCOLS_REAL] = {TRSM_BLKSIZES_REAL};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }

    if(blk == 1)
        blk = std::min(m, 512);

    return blk;
}

/** complex type version **/
template <bool ISBATCHED, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_int rocsolver_trsm_blksize(const rocblas_int m, const rocblas_int n)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int M = TRSM_BATCH_NUMROWS_COMPLEX - 1;
        rocblas_int N = TRSM_BATCH_NUMCOLS_COMPLEX - 1;
        rocblas_int intervalsM[] = {TRSM_BATCH_INTERVALSROW_COMPLEX};
        rocblas_int intervalsN[] = {TRSM_BATCH_INTERVALSCOL_COMPLEX};
        rocblas_int size[][TRSM_BATCH_NUMCOLS_COMPLEX] = {TRSM_BATCH_BLKSIZES_COMPLEX};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }
    else
    {
        rocblas_int M = TRSM_NUMROWS_COMPLEX - 1;
        rocblas_int N = TRSM_NUMCOLS_COMPLEX - 1;
        rocblas_int intervalsM[] = {TRSM_INTERVALSROW_COMPLEX};
        rocblas_int intervalsN[] = {TRSM_INTERVALSCOL_COMPLEX};
        rocblas_int size[][TRSM_NUMCOLS_COMPLEX] = {TRSM_BLKSIZES_COMPLEX};
        blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
    }

    if(blk == 1)
        blk = std::min(m, 512);

    return blk;
}

/** This function determine workspace size for the internal trsm **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_trsm_mem(const rocblas_side side,
                        const rocblas_operation trans,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int batch_count,
                        size_t* size_work1,
                        size_t* size_work2,
                        size_t* size_work3,
                        size_t* size_work4,
                        bool* optim_mem,
                        bool inblocked)
{
    // always allocate all required memory for TRSM optimal performance
    *optim_mem = true;

    rocblas_int mm = m;

    if(!inblocked)
    {
        static constexpr bool ISBATCHED = BATCHED || STRIDED;

        // determine type of system and block size
        const bool isleft = (side == rocblas_side_left);
        rocblas_int blk = isleft ? rocsolver_trsm_blksize<ISBATCHED, T>(m, n)
                                 : rocsolver_trsm_blksize<ISBATCHED, T>(n, m);

        if(blk > 0)
        {
            *size_work1 = 0;
            *size_work2 = 0;
            *size_work3 = 0;
            *size_work4 = 0;
            return;
        }
        else
            mm = m;
    }
    else
    {
        // inblocked = true when called from inside blocked algorithms like GETRF.

        // (Note: rocblas TRSM workspace size is less than expected when the number of rows is multiple of 128.
        //  For this reason, when trying to set up a workspace that fits all the TRSM calls for m <= blk,
        //  blk cannot be multiple of 128.)
        //    rocblas_int mm = (blk % 128 != 0) ? blk : blk + 1;
        mm = (m % 128 != 0) ? m : m + 1;
    }

    rocblasCall_trsm_mem<BATCHED, T>(side, trans, mm, n, batch_count, size_work1, size_work2,
                                     size_work3, size_work4);
}

/** Internal TRSM (lower case):
    Optimized function that solves systems
    B <- LX = B,
    B <- L'X = B,
    B <- B = XL, or
    B <- B = XL'

    This is blocked implementation that calls the internal forward/backward subtitution kernels
    to solve the diagonal blocks, and uses gemm to update the right/left -hand-sides **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_lower(rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_diagonal diag,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4)
{
    ROCSOLVER_ENTER("trsm_lower", "side:", side, "trans:", trans, "diag:", diag, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine type of system
    const bool isleft = (side == rocblas_side_left);
    const bool notrans = (trans == rocblas_operation_none);
    const bool isunit = (diag == rocblas_diagonal_unit);
    const bool conj = (trans == rocblas_operation_conjugate_transpose);
    rocblas_int lda1, lda2, ldb1, ldb2, offA, offB, nx, ny, j = 0;

    // determine block size
    rocblas_int blk = isleft ? rocsolver_trsm_blksize<ISBATCHED, T>(m, n)
                             : rocsolver_trsm_blksize<ISBATCHED, T>(n, m);

    if(blk == 0)
    {
        rocblasCall_trsm<BATCHED, T>(handle, side, rocblas_fill_lower, trans, diag, m, n, &one, A,
                                     shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                     optim_mem, work1, work2, work3, work4);
        return;
    }

    // ****** MAIN LOOP ***********
    if(isleft)
    {
        // prepare kernels
        nx = blk;
        ny = n;
        dimx = nx;
        dimy = 1024 / dimx;
        blocks = (ny - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimy * sizeof(T);

        if(notrans) // forward case: LX = B
        {
            lda1 = 1;
            lda2 = lda;
            ldb1 = 1;
            ldb2 = ldb;

            while(j < m - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(j, j, lda);
                offB = idx2D(j, 0, ldb);
                FORWARD_SUBSTITUTIONS;

                // update right hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m - nextpiv, n, blk,
                    &minone, A, shiftA + idx2D(nextpiv, j, lda), lda, strideA, B,
                    shiftB + idx2D(j, 0, ldb), ldb, strideB, &one, B,
                    shiftB + idx2D(nextpiv, 0, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = m - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = idx2D(j, j, lda);
            offB = idx2D(j, 0, ldb);
            FORWARD_SUBSTITUTIONS;
        }

        else // backward case: L'X = B
        {
            lda1 = lda;
            lda2 = 1;
            ldb1 = 1;
            ldb2 = ldb;

            while(j < m - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(m - nextpiv, m - nextpiv, lda);
                offB = idx2D(m - nextpiv, 0, ldb);
                BACKWARD_SUBSTITUTIONS;

                // update right hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, trans, rocblas_operation_none, m - nextpiv, n, blk, &minone, A,
                    shiftA + idx2D(m - nextpiv, 0, lda), lda, strideA, B,
                    shiftB + idx2D(m - nextpiv, 0, ldb), ldb, strideB, &one, B,
                    shiftB + idx2D(0, 0, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = m - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = 0;
            offB = 0;
            BACKWARD_SUBSTITUTIONS;
        }
    }

    else
    {
        // prepare kernels
        nx = blk;
        ny = m;
        dimx = nx;
        dimy = 1024 / dimx;
        blocks = (ny - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimy * sizeof(T);

        if(notrans) // backward case: B = XL
        {
            lda1 = lda;
            lda2 = 1;
            ldb1 = ldb;
            ldb2 = 1;

            while(j < n - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(n - nextpiv, n - nextpiv, lda);
                offB = idx2D(0, n - nextpiv, ldb);
                BACKWARD_SUBSTITUTIONS;

                // update left hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m, n - nextpiv, blk,
                    &minone, B, shiftB + idx2D(0, n - nextpiv, ldb), ldb, strideB, A,
                    shiftA + idx2D(n - nextpiv, 0, lda), lda, strideA, &one, B,
                    shiftB + idx2D(0, 0, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = n - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = 0;
            offB = 0;
            BACKWARD_SUBSTITUTIONS;
        }

        else // forward case: B = XL'
        {
            lda1 = 1;
            lda2 = lda;
            ldb1 = ldb;
            ldb2 = 1;

            while(j < n - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(j, j, lda);
                offB = idx2D(0, j, ldb);
                FORWARD_SUBSTITUTIONS;

                // update left hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, trans, m, n - nextpiv, blk, &minone, B,
                    shiftB + idx2D(0, j, ldb), ldb, strideB, A, shiftA + idx2D(nextpiv, j, lda),
                    lda, strideA, &one, B, shiftB + idx2D(0, nextpiv, ldb), ldb, strideB,
                    batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = n - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = idx2D(j, j, lda);
            offB = idx2D(0, j, ldb);
            FORWARD_SUBSTITUTIONS;
        }
    }
}

/** Internal TRSM (upper case):
    Optimized function that solves systems
    B <- UX = B,
    B <- U'X = B,
    B <- B = XU, or
    B <- B = XU'
    This is blocked implementation that calls the internal forward/backward subtitution kernels
    to solve the diagonal blocks, and uses gemm to update the right/left -hand-sides **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_upper(rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_diagonal diag,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4)
{
    ROCSOLVER_ENTER("trsm_upper", "side:", side, "trans:", trans, "diag:", diag, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    rocblas_int dimx, dimy, blocks, nextpiv;
    dim3 grid, threads;
    size_t lmemsize;

    // determine type of system
    const bool isleft = (side == rocblas_side_left);
    const bool notrans = (trans == rocblas_operation_none);
    const bool isunit = (diag == rocblas_diagonal_unit);
    const bool conj = (trans == rocblas_operation_conjugate_transpose);
    rocblas_int lda1, lda2, ldb1, ldb2, offA, offB, nx, ny, j = 0;

    // determine block size
    rocblas_int blk = isleft ? rocsolver_trsm_blksize<ISBATCHED, T>(m, n)
                             : rocsolver_trsm_blksize<ISBATCHED, T>(n, m);

    if(blk == 0)
    {
        rocblasCall_trsm<BATCHED, T>(handle, side, rocblas_fill_upper, trans, diag, m, n, &one, A,
                                     shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                     optim_mem, work1, work2, work3, work4);
        return;
    }

    // ****** MAIN LOOP ***********
    if(isleft)
    {
        // prepare kernels
        nx = blk;
        ny = n;
        dimx = nx;
        dimy = 1024 / dimx;
        blocks = (ny - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimy * sizeof(T);

        if(!notrans) // forward case: U'X = B
        {
            lda1 = lda;
            lda2 = 1;
            ldb1 = 1;
            ldb2 = ldb;

            while(j < m - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(j, j, lda);
                offB = idx2D(j, 0, ldb);
                FORWARD_SUBSTITUTIONS;

                // update right hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, trans, rocblas_operation_none, m - nextpiv, n, blk, &minone, A,
                    shiftA + idx2D(j, nextpiv, lda), lda, strideA, B, shiftB + idx2D(j, 0, ldb),
                    ldb, strideB, &one, B, shiftB + idx2D(nextpiv, 0, ldb), ldb, strideB,
                    batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = m - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = idx2D(j, j, lda);
            offB = idx2D(j, 0, ldb);
            FORWARD_SUBSTITUTIONS;
        }

        else // backward case: UX = B
        {
            lda1 = 1;
            lda2 = lda;
            ldb1 = 1;
            ldb2 = ldb;

            while(j < m - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(m - nextpiv, m - nextpiv, lda);
                offB = idx2D(m - nextpiv, 0, ldb);
                BACKWARD_SUBSTITUTIONS;

                // update right hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m - nextpiv, n, blk,
                    &minone, A, shiftA + idx2D(0, m - nextpiv, lda), lda, strideA, B,
                    shiftB + idx2D(m - nextpiv, 0, ldb), ldb, strideB, &one, B,
                    shiftB + idx2D(0, 0, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = m - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = 0;
            offB = 0;
            BACKWARD_SUBSTITUTIONS;
        }
    }

    else
    {
        // prepare kernels
        nx = blk;
        ny = m;
        dimx = nx;
        dimy = 1024 / dimx;
        blocks = (ny - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimy * sizeof(T);

        if(!notrans) // backward case: B = XU'
        {
            lda1 = 1;
            lda2 = lda;
            ldb1 = ldb;
            ldb2 = 1;

            while(j < n - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(n - nextpiv, n - nextpiv, lda);
                offB = idx2D(0, n - nextpiv, ldb);
                BACKWARD_SUBSTITUTIONS;

                // update left hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, trans, m, n - nextpiv, blk, &minone, B,
                    shiftB + idx2D(0, n - nextpiv, ldb), ldb, strideB, A,
                    shiftA + idx2D(0, n - nextpiv, lda), lda, strideA, &one, B,
                    shiftB + idx2D(0, 0, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = n - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = 0;
            offB = 0;
            BACKWARD_SUBSTITUTIONS;
        }

        else // forward case: B = XU
        {
            lda1 = lda;
            lda2 = 1;
            ldb1 = ldb;
            ldb2 = 1;

            while(j < n - blk)
            {
                nextpiv = j + blk;

                // solve for current diagonal block
                offA = idx2D(j, j, lda);
                offB = idx2D(0, j, ldb);
                FORWARD_SUBSTITUTIONS;

                // update left hand sides
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, m, n - nextpiv, blk,
                    &minone, B, shiftB + idx2D(0, j, ldb), ldb, strideB, A,
                    shiftA + idx2D(j, nextpiv, lda), lda, strideA, &one, B,
                    shiftB + idx2D(0, nextpiv, ldb), ldb, strideB, batch_count, nullptr);

                j = nextpiv;
            }

            // solve last diagonal block
            nx = n - j;
            dimx = nx;
            dimy = 1024 / dimx;
            blocks = (ny - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimy * sizeof(T);

            offA = idx2D(j, j, lda);
            offB = idx2D(0, j, ldb);
            FORWARD_SUBSTITUTIONS;
        }
    }
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_TRSM_MEM(BATCHED, STRIDED, T)                                                   \
    template void rocsolver_trsm_mem<BATCHED, STRIDED, T>(                                          \
        const rocblas_side side, const rocblas_operation trans, const rocblas_int m,                \
        const rocblas_int n, const rocblas_int batch_count, size_t* size_work1, size_t* size_work2, \
        size_t* size_work3, size_t* size_work4, bool* optim_mem, bool inblocked)
#define INSTANTIATE_TRSM_LOWER(BATCHED, STRIDED, T, U)                                      \
    template void rocsolver_trsm_lower<BATCHED, STRIDED, T, U>(                             \
        rocblas_handle handle, const rocblas_side side, const rocblas_operation trans,      \
        const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, U A,         \
        const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, U B, \
        const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB,      \
        const rocblas_int batch_count, const bool optim_mem, void* work1, void* work2,      \
        void* work3, void* work4)
#define INSTANTIATE_TRSM_UPPER(BATCHED, STRIDED, T, U)                                      \
    template void rocsolver_trsm_upper<BATCHED, STRIDED, T, U>(                             \
        rocblas_handle handle, const rocblas_side side, const rocblas_operation trans,      \
        const rocblas_diagonal diag, const rocblas_int m, const rocblas_int n, U A,         \
        const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA, U B, \
        const rocblas_int shiftB, const rocblas_int ldb, const rocblas_stride strideB,      \
        const rocblas_int batch_count, const bool optim_mem, void* work1, void* work2,      \
        void* work3, void* work4)
