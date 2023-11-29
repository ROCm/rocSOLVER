/************************************************************************
 * Derived from
 * Gotlub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
 * and
 * Hari & Kovac (2019). On the Convergence of Complex Jacobi Methods.
 *     Linear and Multilinear Algebra 69(3), p. 489-514.
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_syev_heev.hpp"
#include "rocsolver/rocsolver.h"

/************** Kernels and device functions for small size*******************/
/*****************************************************************************/

#define SYEVJ_BDIM 1024 // Max number of threads per thread-block used in syevj_small kernel

/** SYEVJ_SMALL_KERNEL/RUN_SYEVJ applies the Jacobi eigenvalue algorithm to matrices of size
    n <= SYEVJ_BLOCKED_SWITCH. For each off-diagonal element A[i,j], a Jacobi rotation J is
    calculated so that (J'AJ)[i,j] = 0. J only affects rows i and j, and J' only affects
    columns i and j. Therefore, ceil(n / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict between threads. We use top/bottom pairs
    to obtain i's and j's that do not conflict, and cycle them to cover all off-diagonal indices.

    (Call the syevj_small_kernel with batch_count groups in z, of dim = ddx * ddy threads in x.
	Then, the run_syevj device function will be run by all threads organized in a ddx-by-ddy array.
	Normally, ddx <= ceil(n / 2), and ddy <= ceil(n / 2). Any thread with index i >= ceil(n / 2) or
	j >= ceil(n / 2) will not execute any computations). **/
template <typename T, typename S>
__device__ void run_syevj(const rocblas_int dimx,
                          const rocblas_int dimy,
                          const rocblas_int tix,
                          const rocblas_int tiy,
                          const rocblas_esort esort,
                          const rocblas_evect evect,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          T* A,
                          const rocblas_int lda,
                          const S abstol,
                          const S eps,
                          S* residual,
                          const rocblas_int max_sweeps,
                          rocblas_int* n_sweeps,
                          S* W,
                          rocblas_int* info,
                          T* Acpy,
                          S* cosines_res,
                          T* sines_diag,
                          rocblas_int* top,
                          rocblas_int* bottom)
{
    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    S local_res = 0;
    S local_diag = 0;

    if(tiy == 0)
    {
        // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
        // squared Frobenius norm (first by column/row then sum)
        if(uplo == rocblas_fill_upper)
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = n - 1; j > i; j--)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
            cosines_res[tix] = local_res;
            sines_diag[tix] = local_diag;
        }
        else
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = 0; j < i; j++)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
            cosines_res[tix] = local_res;
            sines_diag[tix] = local_diag;
        }

        // initialize top/bottom pairs
        for(i = tix; i < half_n; i += dimx)
        {
            top[i] = i * 2;
            bottom[i] = i * 2 + 1;
        }
    }
    __syncthreads();

    // set tolerance
    local_res = 0;
    local_diag = 0;
    for(i = 0; i < dimx; i++)
    {
        local_res += cosines_res[i];
        local_diag += std::real(sines_diag[i]);
    }
    S tolerance = (local_res + local_diag) * abstol * abstol;

    // execute sweeps
    rocblas_int count = (half_n - 1) / dimx + 1;
    while(sweeps < max_sweeps && local_res > tolerance)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        for(rocblas_int k = 0; k < even_n - 1; ++k)
        {
            for(rocblas_int cc = 0; cc < count; ++cc)
            {
                rocblas_int kx = tix + cc * dimx;

                // get current top/bottom pair
                i = kx < half_n ? top[kx] : n;
                j = kx < half_n ? bottom[kx] : n;

                // calculate current rotation J
                if(tiy == 0 && i < n && j < n)
                {
                    aij = Acpy[i + j * n];
                    mag = std::abs(aij);

                    if(mag < eps)
                    {
                        c = 1;
                        s1 = 0;
                    }
                    else
                    {
                        g = 2 * mag;
                        f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
                        f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                        lartg(f, g, c, s, r);
                        s1 = s * aij / mag;
                    }
                    cosines_res[tix] = c;
                    sines_diag[tix] = s1;
                }
                __syncthreads();

                // apply J from the right and update vectors
                if(i < n && j < n)
                {
                    c = cosines_res[tix];
                    s1 = sines_diag[tix];
                    s2 = conj(s1);

                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[y1 + i * n];
                        temp2 = Acpy[y1 + j * n];
                        Acpy[y1 + i * n] = c * temp1 + s2 * temp2;
                        Acpy[y1 + j * n] = -s1 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[y2 + i * n];
                            temp2 = Acpy[y2 + j * n];
                            Acpy[y2 + i * n] = c * temp1 + s2 * temp2;
                            Acpy[y2 + j * n] = -s1 * temp1 + c * temp2;
                        }

                        if(evect != rocblas_evect_none)
                        {
                            temp1 = A[y1 + i * lda];
                            temp2 = A[y1 + j * lda];
                            A[y1 + i * lda] = c * temp1 + s2 * temp2;
                            A[y1 + j * lda] = -s1 * temp1 + c * temp2;
                            if(y2 < n)
                            {
                                temp1 = A[y2 + i * lda];
                                temp2 = A[y2 + j * lda];
                                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
                            }
                        }
                    }
                }
                __syncthreads();

                // apply J' from the left
                if(i < n && j < n)
                {
                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[i + y1 * n];
                        temp2 = Acpy[j + y1 * n];
                        Acpy[i + y1 * n] = c * temp1 + s1 * temp2;
                        Acpy[j + y1 * n] = -s2 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[i + y2 * n];
                            temp2 = Acpy[j + y2 * n];
                            Acpy[i + y2 * n] = c * temp1 + s1 * temp2;
                            Acpy[j + y2 * n] = -s2 * temp1 + c * temp2;
                        }
                    }
                }
                __syncthreads();

                // round aij and aji to zero
                if(tiy == 0 && i < n && j < n)
                {
                    Acpy[i + j * n] = 0;
                    Acpy[j + i * n] = 0;
                }
                __syncthreads();

                // rotate top/bottom pair
                if(tiy == 0 && kx < half_n)
                {
                    if(i > 0)
                    {
                        if(i == 2 || i == even_n - 1)
                            top[kx] = i - 1;
                        else
                            top[kx] = i + ((i % 2 == 0) ? -2 : 2);
                    }
                    if(j == 2 || j == even_n - 1)
                        bottom[kx] = j - 1;
                    else
                        bottom[kx] = j + ((j % 2 == 0) ? -2 : 2);
                }
                __syncthreads();
            }
        }

        // update norm
        if(tiy == 0)
        {
            local_res = 0;
            for(i = tix; i < n; i += dimx)
            {
                for(j = 0; j < i; j++)
                    local_res += 2 * std::norm(Acpy[i + j * n]);
            }
            cosines_res[tix] = local_res;
        }
        __syncthreads();

        local_res = 0;
        for(i = 0; i < dimx; i++)
            local_res += cosines_res[i];

        sweeps++;
    }

    if(tiy > 0)
        return;

    // finalize outputs
    if(tix == 0)
    {
        *residual = sqrt(local_res);
        if(sweeps <= max_sweeps)
        {
            *n_sweeps = sweeps;
            *info = 0;
        }
        else
        {
            *n_sweeps = max_sweeps;
            *info = 1;
        }
    }

    // update W and then sort eigenvalues and eigenvectors by selection sort
    for(i = tix; i < n; i += dimx)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if((evect == rocblas_evect_none && tix > 0) || esort == rocblas_esort_none)
        return;

    rocblas_int m;
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j)
        {
            if(tix == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tix; i < n; i += dimx)
                    swap(A[i + m * lda], A[i + j * lda]);
            }
        }
    }
}

__host__ __device__ inline void syevj_get_dims(rocblas_int n, rocblas_int* ddx, rocblas_int* ddy)
{
    // (TODO: Some tuning could be beneficial in the future.
    //	For now, we use a max of BDIM = ddx * ddy threads.
    //	ddy is set to ceil(n/2) and ddx to min(BDIM/ddy, ceil(n/2)).

    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    *ddy = half_n;
    *ddx = min(SYEVJ_BDIM / half_n, half_n);
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_BDIM) syevj_small_kernel(const rocblas_esort esort,
                                                                       const rocblas_evect evect,
                                                                       const rocblas_fill uplo,
                                                                       const rocblas_int n,
                                                                       U AA,
                                                                       const rocblas_int shiftA,
                                                                       const rocblas_int lda,
                                                                       const rocblas_stride strideA,
                                                                       const S abstol,
                                                                       const S eps,
                                                                       S* residualA,
                                                                       const rocblas_int max_sweeps,
                                                                       rocblas_int* n_sweepsA,
                                                                       S* WW,
                                                                       const rocblas_stride strideW,
                                                                       rocblas_int* infoA,
                                                                       T* AcpyA)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* W = WW + bid * strideW;
    S* residual = residualA + bid;
    rocblas_int* n_sweeps = n_sweepsA + bid;
    rocblas_int* info = infoA + bid;

    // get dimensions of 2D thread array
    rocblas_int ddx, ddy;
    syevj_get_dims(n, &ddx, &ddy);

    // shared memory
    extern __shared__ double lmem[];
    S* cosines_res = reinterpret_cast<S*>(lmem);
    T* sines_diag = reinterpret_cast<T*>(cosines_res + ddx);
    rocblas_int* top = reinterpret_cast<rocblas_int*>(sines_diag + ddx);
    rocblas_int* bottom = top + half_n;

    // re-arrange threads in 2D array
    rocblas_int tix = tid / ddy;
    rocblas_int tiy = tid % ddy;

    // execute
    run_syevj(ddx, ddy, tix, tiy, esort, evect, uplo, n, A, lda, abstol, eps, residual, max_sweeps,
              n_sweeps, W, info, Acpy, cosines_res, sines_diag, top, bottom);
}

/************** Kernels and device functions for large size*******************/
/*****************************************************************************/

/** SYEVJ_INIT copies A to Acpy, calculates the residual norm of the matrix, and
    initializes the top/bottom pairs.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_init(const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int half_blocks,
                                 const rocblas_int n,
                                 U AA,
                                 const rocblas_int shiftA,
                                 const rocblas_int lda,
                                 const rocblas_stride strideA,
                                 S abstol,
                                 S* residual,
                                 T* AcpyA,
                                 S* norms,
                                 rocblas_int* top,
                                 rocblas_int* bottom,
                                 rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    T temp;
    rocblas_int i, j;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);
    S* sh_diag = sh_res + hipBlockDim_x;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (by column/row)
    S local_res = 0;
    S local_diag = 0;
    if(uplo == rocblas_fill_upper)
    {
        for(i = tid; i < n; i += hipBlockDim_x)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = n - 1; j > i; j--)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    else
    {
        for(i = tid; i < n; i += hipBlockDim_x)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = 0; j < i; j++)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    sh_res[tid] = local_res;
    sh_diag[tid] = local_diag;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < min(n, hipBlockDim_x); i++)
        {
            local_res += sh_res[i];
            local_diag += sh_diag[i];
        }

        norms[bid] = (local_res + local_diag) * abstol * abstol;
        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = 1;
            atomicAdd(completed, 1);
        }
    }

    // initialize top/bottom pairs
    if(bid == 0 && top && bottom)
    {
        for(i = tid; i < half_blocks; i += hipBlockDim_x)
        {
            top[i] = 2 * i;
            bottom[i] = 2 * i + 1;
        }
    }
}

/** SYEVJ_DIAG_KERNEL decomposes diagonal blocks of size nb <= BS2. For each off-diagonal element
    A[i,j], a Jacobi rotation J is calculated so that (J'AJ)[i,j] = 0. J only affects rows i and j,
    and J' only affects columns i and j. Therefore, ceil(nb / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict between threads. We use top/bottom pairs
    to obtain i's and j's that do not conflict, and cycle them to cover all off-diagonal indices.

    Call this kernel with batch_count blocks in z, and BS2 / 2 threads in x and y. Each thread block
    will work on a separate diagonal block; for a matrix consisting of b * b blocks, use b thread
    blocks in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_diag_kernel(const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const S eps,
                                        T* JA,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int nb_max = 2 * hipBlockDim_x;
    rocblas_int offset = hipBlockIdx_x * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j, k;
    rocblas_int xx1 = 2 * tix, xx2 = xx1 + 1;
    rocblas_int yy1 = 2 * tiy, yy2 = yy1 + 1;
    rocblas_int x1 = xx1 + offset, x2 = x1 + 1;
    rocblas_int y1 = yy1 + offset, y2 = y1 + 1;

    rocblas_int half_n = (n - 1) / 2 + 1;
    rocblas_int nb = min(2 * half_n - offset, nb_max);
    rocblas_int half_nb = nb / 2;

    if(tix >= half_nb || tiy >= half_nb)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + half_nb);
    rocblas_int* sh_top = reinterpret_cast<rocblas_int*>(sh_sines + half_nb);
    rocblas_int* sh_bottom = sh_top + half_nb;

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * nb_max] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * nb_max] = 0;
        J[xx2 + yy1 * nb_max] = 0;
        J[xx2 + yy2 * nb_max] = (xx2 == yy2 ? 1 : 0);
    }

    // initialize top/bottom
    if(tiy == 0)
    {
        sh_top[tix] = x1;
        sh_bottom[tix] = x2;
    }

    // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to A
    i = x1;
    j = x2;
    for(k = 0; k < nb - 1; k++)
    {
        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag < eps)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offset;
                xx2 = j - offset;
                temp1 = J[xx1 + yy1 * nb_max];
                temp2 = J[xx2 + yy1 * nb_max];
                J[xx1 + yy1 * nb_max] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * nb_max] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * nb_max];
                    temp2 = J[xx2 + yy2 * nb_max];
                    J[xx1 + yy2 * nb_max] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * nb_max] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && i < n && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }

        // cycle top/bottom pairs
        if(tix == 1)
            i = sh_bottom[0];
        else if(tix > 1)
            i = sh_top[tix - 1];
        if(tix == half_nb - 1)
            j = sh_top[half_nb - 1];
        else
            j = sh_bottom[tix + 1];
        __syncthreads();

        if(tiy == 0)
        {
            sh_top[tix] = i;
            sh_bottom[tix] = j;
        }
    }
}

/** SYEVJ_DIAG_ROTATE rotates off-diagonal blocks of size nb <= BS2 using the rotations calculated
    by SYEVJ_DIAG_KERNEL.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on a separate off-diagonal block; for a matrix consisting of b * b blocks, use b groups
    in x and b - 1 groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_diag_rotate(const bool skip_block,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* JA,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + bix;

    if(completed[bid + 1])
        return;
    if(skip_block && bix == biy)
        return;

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offsetx = bix * nb_max;
    rocblas_int offsety = biy * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = min(n - offsetx, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += J[tix + k * nb_max] * A[y + (k + offsetx) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + k * nb_max]) * A[(k + offsetx) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}

/** SYEVJ_OFFD_KERNEL decomposes off-diagonal blocks of size nb <= BS2. For each element in the block
    (which is an off-diagonal element A[i,j] in the matrix A), a Jacobi rotation J is calculated so that
    (J'AJ)[i,j] = 0. J only affects rows i and j, and J' only affects columns i and j. Therefore,
    nb rotations can be computed and applied in parallel, so long as the rotations do not conflict between
    threads. We select the initial set of i's and j's to span the block's diagonal, and iteratively move
    to the right (wrapping around as necessary) to cover all indices.

    Since A[i,i], A[j,j], and A[j,i] are all in separate blocks, we also need to ensure that
    rotations do not conflict between thread groups. We use block-level top/bottom pairs
    to obtain off-diagonal block indices that do not conflict.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on four matrix blocks; for a matrix consisting of b * b blocks, use b / 2 groups in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_kernel(const rocblas_int blocks,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const S eps,
                                        T* JA,
                                        rocblas_int* top,
                                        rocblas_int* bottom,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[hipBlockIdx_x];
    rocblas_int j = bottom[hipBlockIdx_x];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int k;
    rocblas_int xx1 = tix, xx2 = tix + nb_max;
    rocblas_int yy1 = tiy, yy2 = tiy + nb_max;
    rocblas_int x1 = tix + offseti, x2 = tix + offsetj;
    rocblas_int y1 = tiy + offseti, y2 = tiy + offsetj;

    if(y1 >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * 4 * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + nb_max);

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * ldj] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * ldj] = 0;
        J[xx2 + yy1 * ldj] = 0;
        J[xx2 + yy2 * ldj] = (xx2 == yy2 ? 1 : 0);
    }

    // for each element, calculate the Jacobi rotation and apply it to A
    for(k = 0; k < nb_max; k++)
    {
        // get element indices
        i = x1;
        j = (tix + k) % nb_max + offsetj;

        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag < eps)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offseti;
                xx2 = j - offsetj + nb_max;
                temp1 = J[xx1 + yy1 * ldj];
                temp2 = J[xx2 + yy1 * ldj];
                J[xx1 + yy1 * ldj] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * ldj] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * ldj];
                    temp2 = J[xx2 + yy2 * ldj];
                    J[xx1 + yy2 * ldj] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * ldj] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }
    }
}

/** SYEVJ_OFFD_ROTATE rotates off-diagonal blocks using the rotations calculated by SYEVJ_OFFD_KERNEL.

    Call this kernel with batch_count groups in z, 2*BS2 threads in x and BS2/2 threads in y.
    For a matrix consisting of b * b blocks, use b / 2 groups in x and 2(b - 2) groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_rotate(const bool skip_block,
                                        const rocblas_int blocks,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* JA,
                                        rocblas_int* top,
                                        rocblas_int* bottom,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[bix];
    rocblas_int j = bottom[bix];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);
    if(skip_block && (biy / 2 == i || biy / 2 == j))
        return;

    rocblas_int nb_max = hipBlockDim_x / 2;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int offsetx = (tix < nb_max ? offseti : offsetj - nb_max);
    rocblas_int offsety = biy * hipBlockDim_y;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = min(n - offsetj, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * 4 * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += J[tix + k * ldj] * A[y + (k + offseti) * lda];
        for(k = 0; k < nb; k++)
            temp += J[tix + (k + nb_max) * ldj] * A[y + (k + offsetj) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += conj(J[tix + k * ldj]) * A[(k + offseti) + y * lda];
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + (k + nb_max) * ldj]) * A[(k + offsetj) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}

/** SYEVJ_CYCLE_PAIRS cycles the block-level top/bottom pairs to progress the sweep.

    Call this kernel with any number of threads in x. (Top/bottom pairs are shared across batch instances,
    so only one thread group is needed.) **/
template <typename T>
ROCSOLVER_KERNEL void
    syevj_cycle_pairs(const rocblas_int half_blocks, rocblas_int* top, rocblas_int* bottom)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int i, j, k;

    if(half_blocks <= hipBlockDim_x && tix < half_blocks)
    {
        if(tix == 0)
            i = 0;
        else if(tix == 1)
            i = bottom[0];
        else if(tix > 1)
            i = top[tix - 1];

        if(tix == half_blocks - 1)
            j = top[half_blocks - 1];
        else
            j = bottom[tix + 1];
        __syncthreads();

        top[tix] = i;
        bottom[tix] = j;
    }
    else
    {
        // shared memory
        extern __shared__ double lmem[];
        rocblas_int* sh_top = reinterpret_cast<rocblas_int*>(lmem);
        rocblas_int* sh_bottom = reinterpret_cast<rocblas_int*>(sh_top + half_blocks);

        for(k = tix; k < half_blocks; k += hipBlockDim_x)
        {
            sh_top[k] = top[k];
            sh_bottom[k] = bottom[k];
        }
        __syncthreads();

        for(k = tix; k < half_blocks; k += hipBlockDim_x)
        {
            if(k == 1)
                top[k] = sh_bottom[0];
            else if(k > 1)
                top[k] = sh_top[k - 1];

            if(k == half_blocks - 1)
                bottom[k] = sh_top[half_blocks - 1];
            else
                bottom[k] = sh_bottom[k + 1];
        }
    }
}

/** SYEVJ_CALC_NORM calculates the residual norm of the matrix.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void syevj_calc_norm(const rocblas_int n,
                                      const rocblas_int sweeps,
                                      S* residual,
                                      T* AcpyA,
                                      S* norms,
                                      rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    if(completed[bid + 1])
        return;

    // local variables
    rocblas_int i, j;

    // array pointers
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);

    S local_res = 0;
    for(i = tid; i < n; i += hipBlockDim_x)
    {
        for(j = 0; j < i; j++)
            local_res += 2 * std::norm(Acpy[i + j * n]);
    }
    sh_res[tid] = local_res;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < min(n, hipBlockDim_x); i++)
            local_res += sh_res[i];

        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = sweeps + 1;
            atomicAdd(completed, 1);
        }
    }
}

/** SYEVJ_FINALIZE sets the output values for SYEVJ, and sorts the eigenvalues and
    eigenvectors by selection sort if applicable.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_finalize(const rocblas_esort esort,
                                     const rocblas_evect evect,
                                     const rocblas_int n,
                                     U AA,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     S* residual,
                                     const rocblas_int max_sweeps,
                                     rocblas_int* n_sweeps,
                                     S* WW,
                                     const rocblas_stride strideW,
                                     rocblas_int* info,
                                     T* AcpyA,
                                     rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, m;
    rocblas_int sweeps = 0;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* W = WW + bid * strideW;
    T* Acpy = AcpyA + bid * n * n;

    // finalize outputs
    if(tid == 0)
    {
        rocblas_int sweeps = completed[bid + 1] - 1;
        residual[bid] = sqrt(residual[bid]);
        if(sweeps >= 0)
        {
            n_sweeps[bid] = sweeps;
            info[bid] = 0;
        }
        else
        {
            n_sweeps[bid] = max_sweeps;
            info[bid] = 1;
        }
    }

    // put eigenvalues into output array
    for(i = tid; i < n; i += hipBlockDim_x)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if((evect == rocblas_evect_none && tid > 0) || esort == rocblas_esort_none)
        return;

    // sort eigenvalues & vectors
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += hipBlockDim_x)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
            }
        }
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevj_heevj_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_Acpy,
                                         size_t* size_J,
                                         size_t* size_norms,
                                         size_t* size_top,
                                         size_t* size_bottom,
                                         size_t* size_completed)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_Acpy = 0;
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
        return;
    }

    // size of temporary workspace for copying A
    *size_Acpy = sizeof(T) * n * n * batch_count;

    if(n <= SYEVJ_BLOCKED_SWITCH)
    {
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
        return;
    }

    rocblas_int half_n = (n - 1) / 2 + 1;
    rocblas_int blocks = (n - 1) / BS2 + 1;
    rocblas_int half_blocks = (blocks - 1) / 2 + 1;

    // size of temporary workspace to store the block rotation matrices
    if(half_blocks == 1 && evect == rocblas_evect_none)
        *size_J = sizeof(T) * blocks * BS2 * BS2 * batch_count;
    else
        *size_J = sizeof(T) * half_blocks * 4 * BS2 * BS2 * batch_count;

    // size of temporary workspace to store the full matrix norm
    *size_norms = sizeof(S) * batch_count;

    // size of arrays for temporary top/bottom pairs
    *size_top = sizeof(rocblas_int) * half_blocks * batch_count;
    *size_bottom = sizeof(rocblas_int) * half_blocks * batch_count;

    // size of temporary workspace to indicate problem completion
    *size_completed = sizeof(rocblas_int) * (batch_count + 1);
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevj_heevj_argCheck(rocblas_handle handle,
                                              const rocblas_esort esort,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(esort != rocblas_esort_none && esort != rocblas_esort_ascending)
        return rocblas_status_invalid_value;
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || max_sweeps <= 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_template(rocblas_handle handle,
                                              const rocblas_esort esort,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              const S abstol,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              const rocblas_stride strideW,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* Acpy,
                                              T* J,
                                              S* norms,
                                              rocblas_int* top,
                                              rocblas_int* bottom,
                                              rocblas_int* completed)
{
    ROCSOLVER_ENTER("syevj_heevj", "esort:", esort, "evect:", evect, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n <= 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        // scalar case
        if(n == 1)
            ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream, evect, A,
                                    strideA, W, strideW, batch_count);

        return rocblas_status_success;
    }

    // absolute tolerance for evaluating when the algorithm has converged
    S eps = get_epsilon<S>();
    S atol = (abstol <= 0 ? eps : abstol);

    // local variables
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    if(n <= SYEVJ_BLOCKED_SWITCH)
    {
        // *** USE SINGLE SMALL-SIZE KERNEL ***
        // (TODO: SYEVJ_BLOCKED_SWITCH may need re-tuning as it could be larger than 64 now).

        rocblas_int ddx, ddy;
        syevj_get_dims(n, &ddx, &ddy);
        dim3 grid(1, 1, batch_count);
        dim3 threads(ddx * ddy, 1, 1);
        size_t lmemsize = (sizeof(S) + sizeof(T)) * ddx + 2 * sizeof(rocblas_int) * half_n;

        ROCSOLVER_LAUNCH_KERNEL(syevj_small_kernel<T>, grid, threads, lmemsize, stream, esort,
                                evect, uplo, n, A, shiftA, lda, strideA, atol, eps, residual,
                                max_sweeps, n_sweeps, W, strideW, info, Acpy);
    }
    else
    {
        // *** USE BLOCKED KERNELS ***

        // kernel dimensions
        rocblas_int blocksReset = batch_count / BS1 + 1;
        rocblas_int blocks = (n - 1) / BS2 + 1;
        rocblas_int even_blocks = blocks + blocks % 2;
        rocblas_int half_blocks = even_blocks / 2;

        dim3 gridReset(blocksReset, 1, 1);
        dim3 grid(1, batch_count, 1);
        dim3 gridDK(blocks, 1, batch_count);
        dim3 gridDR(blocks, blocks, batch_count);
        dim3 gridOK(half_blocks, 1, batch_count);
        dim3 gridOR(half_blocks, 2 * blocks, batch_count);
        dim3 gridPairs(1, 1, 1);
        dim3 threadsReset(BS1, 1, 1);
        dim3 threads(BS1, 1, 1);
        dim3 threadsDK(BS2 / 2, BS2 / 2, 1);
        dim3 threadsDR(BS2, BS2, 1);
        dim3 threadsOK(BS2, BS2, 1);
        dim3 threadsOR(2 * BS2, BS2 / 2, 1);

        // shared memory sizes
        size_t lmemsizeInit = 2 * sizeof(S) * BS1;
        size_t lmemsizeDK = (sizeof(S) + sizeof(T) + 2 * sizeof(rocblas_int)) * (BS2 / 2);
        size_t lmemsizeDR = (2 * sizeof(rocblas_int)) * (BS2 / 2);
        size_t lmemsizeOK = (sizeof(S) + sizeof(T)) * BS2;
        size_t lmemsizePairs = (half_blocks > BS1 ? 2 * sizeof(rocblas_int) * half_blocks : 0);

        bool ev = (evect != rocblas_evect_none);
        rocblas_int h_sweeps = 0;
        rocblas_int h_completed = 0;

        // set completed = 0
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, completed,
                                batch_count + 1, 0);

        // copy A to Acpy, set A to identity (if applicable), compute initial residual, and
        // initialize top/bottom pairs (if applicable)
        ROCSOLVER_LAUNCH_KERNEL(syevj_init<T>, grid, threads, lmemsizeInit, stream, evect, uplo,
                                half_blocks, n, A, shiftA, lda, strideA, atol, residual, Acpy,
                                norms, top, bottom, completed);

        while(h_sweeps < max_sweeps)
        {
            // if all instances in the batch have finished, exit the loop
            HIP_CHECK(hipMemcpyAsync(&h_completed, completed, sizeof(rocblas_int),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            if(h_completed == batch_count)
                break;

            // decompose diagonal blocks
            ROCSOLVER_LAUNCH_KERNEL(syevj_diag_kernel<T>, gridDK, threadsDK, lmemsizeDK, stream, n,
                                    Acpy, 0, n, n * n, eps, J, completed);

            // apply rotations calculated by diag_kernel
            ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR, lmemsizeDR,
                                    stream, true, n, Acpy, 0, n, n * n, J, completed);
            ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<true, T, S>), gridDR, threadsDR, lmemsizeDR,
                                    stream, true, n, Acpy, 0, n, n * n, J, completed);

            // update eigenvectors
            if(ev)
                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, false, n, A, shiftA, lda, strideA, J,
                                        completed);

            if(half_blocks == 1)
            {
                // decompose off-diagonal block
                ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK, lmemsizeOK,
                                        stream, blocks, n, Acpy, 0, n, n * n, eps,
                                        (ev ? J : nullptr), top, bottom, completed);

                // update eigenvectors
                if(ev)
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0,
                                            stream, false, blocks, n, A, shiftA, lda, strideA, J,
                                            top, bottom, completed);
            }
            else
            {
                for(rocblas_int b = 0; b < even_blocks - 1; b++)
                {
                    // decompose off-diagonal blocks, indexed by top/bottom pairs
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK,
                                            lmemsizeOK, stream, blocks, n, Acpy, 0, n, n * n, eps,
                                            J, top, bottom, completed);

                    // apply rotations calculated by offd_kernel
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0,
                                            stream, true, blocks, n, Acpy, 0, n, n * n, J, top,
                                            bottom, completed);
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<true, T, S>), gridOR, threadsOR, 0,
                                            stream, true, blocks, n, Acpy, 0, n, n * n, J, top,
                                            bottom, completed);

                    // update eigenvectors
                    if(ev)
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR,
                                                0, stream, false, blocks, n, A, shiftA, lda,
                                                strideA, J, top, bottom, completed);

                    // cycle top/bottom pairs
                    ROCSOLVER_LAUNCH_KERNEL(syevj_cycle_pairs<T>, gridPairs, threads, lmemsizePairs,
                                            stream, half_blocks, top, bottom);
                }
            }

            // compute new residual
            h_sweeps++;
            ROCSOLVER_LAUNCH_KERNEL(syevj_calc_norm<T>, grid, threads, lmemsizeInit, stream, n,
                                    h_sweeps, residual, Acpy, norms, completed);
        }

        // set outputs and sort eigenvalues & vectors
        ROCSOLVER_LAUNCH_KERNEL(syevj_finalize<T>, grid, threads, 0, stream, esort, evect, n, A,
                                shiftA, lda, strideA, residual, max_sweeps, n_sweeps, W, strideW,
                                info, Acpy, completed);
    }

    return rocblas_status_success;
}
