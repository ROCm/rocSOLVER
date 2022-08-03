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
#define SYEVJ_MAX_THDS warpSize

/** Apply rotation from the left (based on LASR). **/
template <typename T, typename S>
__device__ void syevj_lasr_left(const rocblas_int n,
                                const rocblas_int i,
                                const rocblas_int ii,
                                S c,
                                T s1,
                                T s2,
                                T* A,
                                const rocblas_int lda)
{
    T temp1, temp2;

    for(rocblas_int j = 0; j < n; ++j)
    {
        temp1 = A[i + j * lda];
        temp2 = A[ii + j * lda];
        A[i + j * lda] = c * temp1 + s1 * temp2;
        A[ii + j * lda] = -s2 * temp1 + c * temp2;
    }
}
/** Apply rotation from the right (based on LASR). **/
template <typename T, typename S>
__device__ void syevj_lasr_right(const rocblas_int n,
                                 const rocblas_int j,
                                 const rocblas_int jj,
                                 S c,
                                 T s1,
                                 T s2,
                                 T* A,
                                 const rocblas_int lda)
{
    T temp1, temp2;

    for(rocblas_int i = 0; i < n; ++i)
    {
        temp1 = A[i + j * lda];
        temp2 = A[i + jj * lda];
        A[i + j * lda] = c * temp1 + s2 * temp2;
        A[i + jj * lda] = -s1 * temp1 + c * temp2;
    }
}

/** SYEVJ_SMALL_KERNEL applies the Jacobi eigenvalue algorithm to matrices of size <= 2 * SYEVJ_MAX_THDS.
 *  Call this kernel with batch_count groups in y, and ceil(n / 2) threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_MAX_THDS)
    syevj_small_kernel(const rocblas_esort esort,
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
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, k, i_new = 0, j_new = 0;
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
    rocblas_int* top = reinterpret_cast<rocblas_int*>(resarr + half_n);
    rocblas_int* bottom = top + half_n;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (first by column/row then sum)
    S local_res = 0;
    if(uplo == rocblas_fill_upper)
    {
        T temp;
        for(i = tid; i < n; i += half_n)
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
        for(j = tid; j < n; j += half_n)
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
    __syncthreads();

    local_res = 0;
    for(i = 0; i < half_n; i++)
        local_res += resarr[i];

    // quick return if norm is already small
    if(local_res <= abstol * abstol)
    {
        if(tid == 0)
        {
            residual[bid] = sqrt(local_res);
            n_sweeps[bid] = 0;
            info[bid] = 0;
        }
        return;
    }

    // initialize top/bottom
    top[tid] = 2 * tid;
    bottom[tid] = 2 * tid + 1;
    __syncthreads();

    // execute sweeps
    while(sweeps < max_sweeps && local_res > abstol * abstol)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        S c, mag, f, g, r, s;
        T s1, s2, aij;
        for(k = 0; k < n - 1; k++)
        {
            // get current top/bottom pair
            i = top[tid];
            j = bottom[tid];

            // cycle top/bottom pairs
            if(tid == 1)
                i_new = bottom[0];
            else if(tid > 1)
                i_new = top[tid - 1];
            if(tid == half_n - 1)
                j_new = top[half_n - 1];
            else
                j_new = bottom[tid + 1];

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
                    s2 = conj(s1);
                }

                // apply J from the right
                syevj_lasr_right(n, i, j, c, s1, s2, Acpy, n);

                // update eigenvectors
                if(evect != rocblas_evect_none)
                    syevj_lasr_right(n, i, j, c, s1, s2, A, lda);
            }
            __syncthreads();

            if(i < n && j < n)
            {
                // apply J' from the left
                syevj_lasr_left(n, i, j, c, s1, s2, Acpy, n);

                // round aij and aji to zero
                Acpy[j + i * n] = Acpy[i + j * n] = 0;
            }

            top[tid] = i_new;
            bottom[tid] = j_new;

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
                resarr[tid] = local_res;
            }
            __syncthreads();
        }

        // update norm
        local_res = 0;
        for(i = 0; i < half_n; i++)
            local_res += resarr[i];

        sweeps++;
    }

    // update residual, n_sweeps, and info
    if(tid == 0)
    {
        residual[bid] = sqrt(local_res);
        n_sweeps[bid] = sweeps;
        if(sweeps > max_sweeps)
            info[bid] = 1;
        else
            info[bid] = 0;
    }

    // update W and then sort eigenvalues and eigenvectors by selection sort
    for(i = tid; i < n; i += half_n)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if((evect == rocblas_evect_none && tid > 0) || esort == rocblas_esort_none)
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

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += half_n)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
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
 *  Call this kernel with batch_count groups in y, and SYEVJ_MAX_THDS threads in x. **/
template <int MAX_THDS, typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_MAX_THDS)
    syevj_large_iterate(const rocblas_evect evect,
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
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid;

    // task variables
    __shared__ rocblas_int phase_id;
    __shared__ rocblas_int task_id;
    __shared__ rocblas_int task_count;
    if(tx == 0)
    {
        phase_id = 0;
        task_id = 0;
        task_count = hipGridDim_x;
    }

    // local variables
    rocblas_int i, j, k, j_new;
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
    rocblas_int* counters = countersA + bid * 5;

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
            // STAGE 1: APPLY ROTATION FROM THE RIGHT
            do
            {
                if(tx == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count);
                __syncthreads();

                tid = (task_id - 1) * dimx + tx;
                if(task_id > 0 && tid < half_n)
                {
                    // get current top/bottom pair
                    i = top[tid];
                    j = bottom[tid];
                    top_temp[tid] = i;
                    bottom_temp[tid] = j;

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
                            s2 = conj(s1);
                        }

                        cosines[tid] = c;
                        sines[tid] = s1;

                        // apply J from the right
                        syevj_lasr_right(n, i, j, c, s1, s2, Acpy, n);

                        // update eigenvectors
                        if(evect != rocblas_evect_none)
                            syevj_lasr_right(n, i, j, c, s1, s2, A, lda);
                    }
                }
                __threadfence();
                __syncthreads();
            } while(task_id > 0);

            // STAGE 2: APPLY ROTATION FROM THE LEFT
            do
            {
                if(tx == 0)
                    get_task_id(&phase_id, &task_id, counters, task_count);
                __syncthreads();

                tid = (task_id - 1) * dimx + tx;
                if(task_id > 0 && tid < half_n)
                {
                    // get current top/bottom pair
                    i = top_temp[tid];
                    j = bottom_temp[tid];

                    // cycle top/bottom pairs
                    if(tid == 1)
                        top[tid] = bottom_temp[0];
                    else if(tid > 1)
                        top[tid] = top_temp[tid - 1];
                    if(tid == half_n - 1)
                        bottom[tid] = top_temp[half_n - 1];
                    else
                        bottom[tid] = bottom_temp[tid + 1];

                    if(i < n && j < n)
                    {
                        c = cosines[tid];
                        s1 = sines[tid];
                        s2 = conj(s1);

                        // apply J' from the left
                        syevj_lasr_left(n, i, j, c, s1, s2, Acpy, n);

                        // round aij and aji to zero
                        Acpy[j + i * n] = Acpy[i + j * n] = 0;
                    }

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
                        resarr[tid] = local_res;
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

    // STAGE 3: UPDATE OUTPUTS
    do
    {
        if(tx == 0)
            get_task_id(&phase_id, &task_id, counters, task_count);
        __syncthreads();

        tid = (task_id - 1) * dimx + tx;
        if(task_id > 0)
        {
            // update residual, n_sweeps, and info()
            if(tid == 0)
            {
                residual[bid] = sqrt(local_res);
                n_sweeps[bid] = sweeps;
                if(sweeps > max_sweeps)
                    info[bid] = 1;
                else
                    info[bid] = 0;
            }

            // update W
            i = 2 * tid;
            if(i < n)
                W[i] = std::real(Acpy[i + i * n]);

            i++;
            if(i < n)
                W[i] = std::real(Acpy[i + i * n]);
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
    *size_counters = sizeof(rocblas_int) * 5 * batch_count;
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
        dim3 grid(1, batch_count, 1);
        dim3 threads(half_n, 1, 1);

        size_t lmemsize = (sizeof(S) + 2 * sizeof(rocblas_int)) * half_n;

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
        dim3 grid2(blocks, batch_count, 1);
        dim3 threads(SYEVJ_MAX_THDS, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, counters,
                                5 * batch_count, 0);

        ROCSOLVER_LAUNCH_KERNEL((syevj_large_init<SYEVJ_MAX_THDS, T>), grid1, threads, 0, stream,
                                evect, uplo, n, A, shiftA, lda, strideA, atol, residual, n_sweeps,
                                Acpy, resarr, tbarr);

        ROCSOLVER_LAUNCH_KERNEL((syevj_large_iterate<SYEVJ_MAX_THDS, T>), grid2, threads, 0, stream,
                                evect, n, A, shiftA, lda, strideA, atol, eps, residual, max_sweeps,
                                n_sweeps, W, strideW, info, Acpy, resarr, cosines, sines, tbarr,
                                counters);

        if(esort == rocblas_esort_ascending)
            ROCSOLVER_LAUNCH_KERNEL((syevj_large_sort<SYEVJ_MAX_THDS, T>), grid1, threads, 0,
                                    stream, evect, n, A, shiftA, lda, strideA, W, strideW);
    }

    return rocblas_status_success;
}
