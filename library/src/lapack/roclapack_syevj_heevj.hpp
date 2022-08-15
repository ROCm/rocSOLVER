/************************************************************************
 * Derived from
 * Gotlub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
 * and
 * Hari & Kovac (2019). On the Convergence of Complex Jacobi Methods.
 *     Linear and Multilinear Algebra 69(3), p. 489-514.
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_syev_heev.hpp"
#include "rocsolver/rocsolver.h"

/** Thread-block size for calling the syevj kernel. **/
#define SYEVJ_MAX_THDS BS2

/** SYEVJ_SMALL_KERNEL applies the Jacobi eigenvalue algorithm to matrices of size n <= 2 * SYEVJ_MAX_THDS.
    Call this kernel with batch_count groups in z, and ceil(n / 2) threads in x and y. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_small_kernel(const rocblas_esort esort,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         const S abstol,
                                         const S eps,
                                         S* residual,
                                         const rocblas_int max_sweeps,
                                         rocblas_int* n_sweeps,
                                         S* WW,
                                         const rocblas_stride strideW,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         T* AcpyA)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;

    // local variables
    T aij, temp1, temp2;
    rocblas_int i, j, k;
    rocblas_int x1 = 2 * tix, x2 = 2 * tix + 1;
    rocblas_int y1 = 2 * tiy, y2 = 2 * tiy + 1;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* W = WW + bid * strideW;

    // shared memory
    extern __shared__ double lmem[];
    S* resarr = reinterpret_cast<S*>(lmem);
    S* cosines = resarr + half_n;
    T* sines = reinterpret_cast<T*>(cosines + half_n);
    rocblas_int* top = reinterpret_cast<rocblas_int*>(sines + half_n);
    rocblas_int* bottom = top + half_n;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (first by column/row then sum)
    S local_res = 0;
    if(tiy == 0 && uplo == rocblas_fill_upper)
    {
        for(i = tix; i < n; i += half_n)
        {
            Acpy[i + i * n] = A[i + i * lda];
            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = i + 1; j < n; j++)
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
        resarr[tix] = local_res;
    }
    if(tiy == 0 && uplo == rocblas_fill_lower)
    {
        for(j = tix; j < n; j += half_n)
        {
            Acpy[j + j * n] = A[j + j * lda];
            if(evect != rocblas_evect_none)
                A[j + j * lda] = 1;

            for(i = j + 1; i < n; i++)
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
        resarr[tix] = local_res;
    }
    __syncthreads();

    local_res = 0;
    for(i = 0; i < half_n; i++)
        local_res += resarr[i];

    // quick return if norm is already small
    if(local_res <= abstol * abstol)
    {
        if(tix == 0 && tiy == 0)
        {
            residual[bid] = sqrt(local_res);
            n_sweeps[bid] = 0;
            info[bid] = 0;
        }
        return;
    }

    // initialize top/bottom
    if(tiy == 0)
    {
        top[tix] = 2 * tix;
        bottom[tix] = 2 * tix + 1;
    }
    __syncthreads();

    // execute sweeps
    while(sweeps < max_sweeps && local_res > abstol * abstol)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        S c, mag, f, g, r, s;
        T s1, s2;
        for(k = 0; k < n - 1; k++)
        {
            // get current top/bottom pair
            i = top[tix];
            j = bottom[tix];

            if(tiy == 0 && i < n && j < n)
            {
                aij = Acpy[i + j * n];
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
                    f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
                    f += (f < 0) ? -sqrt(f * f + g * g) : sqrt(f * f + g * g);
                    lartg(f, g, c, s, r);
                    s1 = s * aij / mag;
                }

                cosines[tix] = c;
                sines[tix] = s1;
            }
            __syncthreads();

            if(i < n && j < n)
            {
                c = cosines[tix];
                s1 = sines[tix];
                s2 = conj(s1);

                // apply J from the right
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

                // update eigenvectors
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
            __syncthreads();

            if(i < n && j < n)
            {
                // apply J' from the left
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
            __syncthreads();

            if(tiy == 0)
            {
                // round aij and aji to zero
                if(i < n && j < n)
                    Acpy[j + i * n] = Acpy[i + j * n] = 0;

                // cycle top/bottom pairs
                if(tix == 1)
                    i = bottom[0];
                else if(tix > 1)
                    i = top[tix - 1];
                if(tix == half_n - 1)
                    j = top[half_n - 1];
                else
                    j = bottom[tix + 1];

                top[tix] = i;
                bottom[tix] = j;
            }
            __syncthreads();
        }

        // update norm
        if(tiy == 0)
        {
            local_res = 0;
            for(i = x1 + 1; i < n; i++)
                local_res += 2 * std::norm(Acpy[i + x1 * n]);
            if(x2 < n)
                for(i = x2 + 1; i < n; i++)
                    local_res += 2 * std::norm(Acpy[i + x2 * n]);
            resarr[tix] = local_res;
        }
        __syncthreads();

        // update norm
        local_res = 0;
        for(i = 0; i < half_n; i++)
            local_res += resarr[i];

        sweeps++;
    }

    if(tiy > 0)
        return;

    // update residual, n_sweeps, and info
    if(tix == 0)
    {
        residual[bid] = sqrt(local_res);
        n_sweeps[bid] = sweeps;
        if(sweeps > max_sweeps)
            info[bid] = 1;
        else
            info[bid] = 0;
    }

    // update W and then sort eigenvalues and eigenvectors by selection sort
    W[x1] = std::real(Acpy[x1 + x1 * n]);
    if(x2 < n)
        W[x2] = std::real(Acpy[x2 + x2 * n]);
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
                swap(A[x1 + m * lda], A[x1 + j * lda]);
                if(x2 < n)
                    swap(A[x2 + m * lda], A[x2 + j * lda]);
            }
        }
    }
}

/** SYEVJ_LARGE_INIT copies A to Acpy, calculates the initial residual, and initializes top/bottom pairs.
 *  Call this kernel with batch_count groups in y, and SYEVJ_MAX_THDS threads in x. **/
template <int MAX_THDS, typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_MAX_THDS)
    syevj_large_init(const rocblas_evect evect,
                     const rocblas_fill uplo,
                     const rocblas_int n,
                     U AA,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const S abstol,
                     S* residual,
                     rocblas_int* n_sweeps,
                     T* AcpyA,
                     S* resarrA,
                     rocblas_int* tbarrA)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* resarr = resarrA + bid * half_n;
    rocblas_int* top = tbarrA + bid * (2 * even_n);
    rocblas_int* bottom = top + half_n;
    rocblas_int* top_temp = bottom + half_n;
    rocblas_int* bottom_temp = top_temp + half_n;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (by column/row)
    S local_res = 0;
    if(uplo == rocblas_fill_upper)
    {
        T temp;
        for(i = tid; i < n; i += MAX_THDS)
        {
            Acpy[i + i * n] = A[i + i * lda];
            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = i + 1; j < n; j++)
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
        T temp;
        for(j = tid; j < n; j += MAX_THDS)
        {
            Acpy[j + j * n] = A[j + j * lda];
            if(evect != rocblas_evect_none)
                A[j + j * lda] = 1;

            for(i = j + 1; i < n; i++)
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
    resarr[tid] = local_res;

    // initialize top/bottom
    if(tid == 0)
        top_temp[0] = 0;
    for(i = tid; i < half_n; i += MAX_THDS)
    {
        top[i] = 2 * i;
        bottom[i] = 2 * i + 1;
    }
}

/** SYEVJ_LARGE_KERNEL applies the Jacobi eigenvalue algorithm to matrices of size > 2 * SYEVJ_MAX_THDS.
 *  Call this kernel with batch_count groups in z, and n threads in x and y. **/
template <int MAX_THDS, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_large_iterate(const rocblas_evect evect,
                                          const rocblas_int n,
                                          U AA,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          const S abstol,
                                          const S eps,
                                          S* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          S* WW,
                                          const rocblas_stride strideW,
                                          rocblas_int* info,
                                          T* AcpyA,
                                          S* resarrA,
                                          S* cosinesA,
                                          T* sinesA,
                                          rocblas_int* tbarrA,
                                          rocblas_int* countersA)
{
    rocblas_int dimx = hipBlockDim_x;
    rocblas_int dimy = hipBlockDim_y;
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int ty = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int tix, tiy;

    // task variables
    __shared__ rocblas_int phase_id;
    __shared__ rocblas_int task_id;
    __shared__ rocblas_int task_count1;
    __shared__ rocblas_int task_count2;
    if(tx == 0 && ty == 0)
    {
        phase_id = 0;
        task_id = 0;
        task_count1 = hipGridDim_x;
        task_count2 = hipGridDim_x * hipGridDim_y;
    }

    // local variables
    T temp1, temp2;
    rocblas_int i, j, k, j_new;
    rocblas_int x1, x2;
    rocblas_int y1, y2;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* W = WW + bid * strideW;
    S* resarr = resarrA + bid * half_n;
    S* cosines = cosinesA + bid * half_n;
    T* sines = sinesA + bid * half_n;
    rocblas_int* top = tbarrA + bid * (2 * even_n);
    rocblas_int* bottom = top + half_n;
    rocblas_int* top_temp = bottom + half_n;
    rocblas_int* bottom_temp = top_temp + half_n;
    rocblas_int* counters = countersA + bid * 4;

    // compute off-diagonal squared Frobenius norm
    S local_res = 0;
    for(i = 0; i < MAX_THDS; i++)
        local_res += resarr[i];

    // execute sweeps
    while(sweeps < max_sweeps && local_res > abstol * abstol)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        S c, mag, f, g, r, s;
        T s1, s2, aij;
        for(k = 0; k < n - 1; k++)
        {
            // STAGE 1: CALCULATE ROTATION
            do
            {
                if(tx == 0 && ty == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count1);
                __syncthreads();

                tix = (task_id - 1) * dimx + tx;
                if(task_id > 0 && tix < half_n && ty == 0)
                {
                    // get current top/bottom pair
                    i = top[tix];
                    j = bottom[tix];
                    top_temp[tix] = i;
                    bottom_temp[tix] = j;

                    if(i < n && j < n)
                    {
                        aij = Acpy[i + j * n];
                        mag = std::abs(aij);

                        // calculate rotation J
                        if(mag < eps)
                        {
                            c = 1;
                            s1 = s2 = 0;
                        }
                        else
                        {
                            g = 2 * mag;
                            f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
                            f += (f < 0) ? -sqrt(f * f + g * g) : sqrt(f * f + g * g);
                            lartg(f, g, c, s, r);
                            s1 = s * aij / mag;
                        }

                        cosines[tix] = c;
                        sines[tix] = s1;
                    }
                }
                __threadfence();
                __syncthreads();
            } while(task_id > 0);

            // STAGE 2: APPLY ROTATION FROM THE RIGHT
            do
            {
                if(tx == 0 && ty == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count2);
                __syncthreads();

                tix = ((task_id - 1) % task_count1) * dimx + tx;
                tiy = ((task_id - 1) / task_count1) * dimy + ty;
                y1 = 2 * tiy;
                y2 = 2 * tiy + 1;
                if(task_id > 0 && tix < half_n && tiy < half_n)
                {
                    // get current top/bottom pair
                    i = top_temp[tix];
                    j = bottom_temp[tix];

                    if(i < n && j < n)
                    {
                        c = cosines[tix];
                        s1 = sines[tix];
                        s2 = conj(s1);

                        // apply J from the right
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

                        // update eigenvectors
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
                __threadfence();
                __syncthreads();
            } while(task_id > 0);

            // STAGE 3: APPLY ROTATION FROM THE LEFT
            do
            {
                if(tx == 0 && ty == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count2);
                __syncthreads();

                tix = ((task_id - 1) % task_count1) * dimx + tx;
                tiy = ((task_id - 1) / task_count1) * dimy + ty;
                y1 = 2 * tiy;
                y2 = 2 * tiy + 1;
                if(task_id > 0 && tix < half_n && tiy < half_n)
                {
                    // get current top/bottom pair
                    i = top_temp[tix];
                    j = bottom_temp[tix];

                    if(i < n && j < n)
                    {
                        c = cosines[tix];
                        s1 = sines[tix];
                        s2 = conj(s1);

                        // apply J' from the left
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
                __threadfence();
                __syncthreads();
            } while(task_id > 0);

            // STAGE 4: CYCLE TOP/BOTTOM PAIRS
            do
            {
                if(tx == 0 && ty == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count1);
                __syncthreads();

                tix = (task_id - 1) * dimx + tx;
                if(task_id > 0 && tix < half_n && ty == 0)
                {
                    // get current top/bottom pair
                    i = top_temp[tix];
                    j = bottom_temp[tix];

                    // round aij and aji to zero
                    if(i < n && j < n)
                        Acpy[j + i * n] = Acpy[i + j * n] = 0;

                    // cycle top/bottom pairs
                    if(tix == 1)
                        top[tix] = bottom_temp[0];
                    else if(tix > 1)
                        top[tix] = top_temp[tix - 1];
                    if(tix == half_n - 1)
                        bottom[tix] = top_temp[half_n - 1];
                    else
                        bottom[tix] = bottom_temp[tix + 1];

                    // update row norms
                    if(k == n - 2)
                    {
                        local_res = 0;
                        if(i < n)
                            for(j_new = i + 1; j_new < n; j_new++)
                                local_res += 2 * std::norm(Acpy[i + j_new * n]);
                        if(j < n)
                            for(j_new = j + 1; j_new < n; j_new++)
                                local_res += 2 * std::norm(Acpy[j + j_new * n]);
                        resarr[tix] = local_res;
                    }
                }
                __threadfence();
                __syncthreads();
            } while(task_id > 0);
        }

        // update norm
        local_res = 0;
        for(i = 0; i < half_n; i++)
            local_res += resarr[i];

        sweeps++;
    }

    if(ty > 0)
        return;

    // STAGE 5: UPDATE OUTPUTS
    do
    {
        if(tx == 0 && ty == 0)
            get_task_id(&phase_id, &task_id, counters, task_count1);
        __syncthreads();

        tix = (task_id - 1) * dimx + tx;
        x1 = 2 * tix;
        x2 = 2 * tix + 1;
        if(task_id > 0)
        {
            // update residual, n_sweeps, and info()
            if(tix == 0)
            {
                residual[bid] = sqrt(local_res);
                n_sweeps[bid] = sweeps;
                if(sweeps > max_sweeps)
                    info[bid] = 1;
                else
                    info[bid] = 0;
            }

            // update W
            if(x1 < n)
                W[x1] = std::real(Acpy[x1 + x1 * n]);
            if(x2 < n)
                W[x2] = std::real(Acpy[x2 + x2 * n]);
        }
        __syncthreads();
    } while(task_id > 0);
}

/** SYEVJ_LARGE_SORT sorts the eigenvalues and eigenvectors by selection sort.
 *  Call this kernel with batch_count groups in y, and SYEVJ_MAX_THDS threads in x. **/
template <int MAX_THDS, typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_MAX_THDS)
    syevj_large_sort(const rocblas_evect evect,
                     const rocblas_int n,
                     U AA,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     S* WW,
                     const rocblas_stride strideW)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, m;
    rocblas_int sweeps = 0;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* W = WW + bid * strideW;

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

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += MAX_THDS)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
            }
        }
    }
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevj_heevj_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_Acpy,
                                         size_t* size_resarr,
                                         size_t* size_cosines,
                                         size_t* size_sines,
                                         size_t* size_tbarr,
                                         size_t* size_counters)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_Acpy = 0;
        *size_resarr = 0;
        *size_cosines = 0;
        *size_sines = 0;
        *size_tbarr = 0;
        *size_counters = 0;
        return;
    }

    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // size of temporary workspace for copying A
    *size_Acpy = sizeof(T) * n * n * batch_count;

    if(half_n <= SYEVJ_MAX_THDS)
    {
        *size_resarr = 0;
        *size_cosines = 0;
        *size_sines = 0;
        *size_tbarr = 0;
        *size_counters = 0;
        return;
    }

    // size of array for per-thread residuals
    *size_resarr = sizeof(S) * half_n * batch_count;

    // size of arrays for temporary cosines/sine pairs
    *size_cosines = sizeof(S) * half_n * batch_count;
    *size_sines = sizeof(T) * half_n * batch_count;

    // size of array for temporary top/bottom pairs
    *size_tbarr = sizeof(rocblas_int) * (2 * even_n) * batch_count;

    // size of array for task counters
    *size_counters = sizeof(rocblas_int) * 4 * batch_count;
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
                                              S* resarr,
                                              S* cosines,
                                              T* sines,
                                              rocblas_int* tbarr,
                                              rocblas_int* counters)
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
    S atol = (abstol <= 0) ? eps : abstol;

    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    if(half_n <= SYEVJ_MAX_THDS)
    {
        dim3 grid(1, 1, batch_count);
        dim3 threads(half_n, half_n, 1);

        size_t lmemsize = (2 * sizeof(S) + sizeof(T) + 2 * sizeof(rocblas_int)) * half_n;

        ROCSOLVER_LAUNCH_KERNEL(syevj_small_kernel<T>, grid, threads, lmemsize, stream, esort,
                                evect, uplo, n, A, shiftA, lda, strideA, atol, eps, residual,
                                max_sweeps, n_sweeps, W, strideW, info, batch_count, Acpy);
    }
    else
    {
        rocblas_int blocksReset = ((4 * n) * batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        rocblas_int blocks = (half_n - 1) / SYEVJ_MAX_THDS + 1;
        dim3 grid1(1, batch_count, 1);
        dim3 grid2(blocks, blocks, batch_count);
        dim3 threads1(SYEVJ_MAX_THDS, 1, 1);
        dim3 threads2(SYEVJ_MAX_THDS, SYEVJ_MAX_THDS, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, counters,
                                4 * batch_count, 0);

        ROCSOLVER_LAUNCH_KERNEL((syevj_large_init<SYEVJ_MAX_THDS, T>), grid1, threads1, 0, stream,
                                evect, uplo, n, A, shiftA, lda, strideA, atol, residual, n_sweeps,
                                Acpy, resarr, tbarr);

        ROCSOLVER_LAUNCH_KERNEL((syevj_large_iterate<SYEVJ_MAX_THDS, T>), grid2, threads2, 0,
                                stream, evect, n, A, shiftA, lda, strideA, atol, eps, residual,
                                max_sweeps, n_sweeps, W, strideW, info, Acpy, resarr, cosines,
                                sines, tbarr, counters);

        if(esort == rocblas_esort_ascending)
            ROCSOLVER_LAUNCH_KERNEL((syevj_large_sort<SYEVJ_MAX_THDS, T>), grid1, threads1, 0,
                                    stream, evect, n, A, shiftA, lda, strideA, W, strideW);
    }

    return rocblas_status_success;
}
