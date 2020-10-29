/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_STERF_HPP
#define ROCLAPACK_STERF_HPP

#include "rocblas.hpp"
#include "rocblas_device_functions.hpp"
#include "rocsolver.h"

/****************************************************************************
(TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH).
***************************************************************************/

/** STERF_FIND_MAX finds the element with the largest magnitude in the
    tridiagonal matrix **/
template <typename T>
__device__ void sterf_find_max(const rocblas_int start, const rocblas_int end, T* D, T* E, T& anorm)
{
    anorm = abs(D[end - 1]);
    for(int i = start; i < end; i++)
        anorm = max(anorm, max(abs(D[i - 1]), abs(E[i - 1])));
}

/** STERF_SCALE scales the elements of the tridiagonal matrix by a given
    scale factor **/
template <typename T>
__device__ void sterf_scale(const rocblas_int start, const rocblas_int end, T* D, T* E, T scale)
{
    D[end - 1] *= scale;
    for(int i = start; i < end; i++)
    {
        D[i - 1] *= scale;
        E[i - 1] *= scale;
    }
}

/** STERF_SQ_E squares the elements of E **/
template <typename T>
__device__ void sterf_sq_e(const rocblas_int start, const rocblas_int end, T* E)
{
    for(int i = start; i < end; i++)
        E[i - 1] = E[i - 1] * E[i - 1];
}

/** STERF_KERNEL implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T>
__global__ void sterf_kernel(const rocblas_int n,
                             T* DD,
                             const rocblas_stride strideD,
                             T* EE,
                             const rocblas_stride strideE,
                             rocblas_int* info,
                             rocblas_int* stack,
                             const rocblas_int max_iters,
                             const T eps,
                             const T ssfmin,
                             const T ssfmax)
{
    rocblas_int bid = hipBlockIdx_x;

    T* D = DD + (bid * strideD);
    T* E = EE + (bid * strideE);

    rocblas_int m, l, lsv, lend, lendsv;
    rocblas_int l1 = 1;
    rocblas_int iters = 0;
    T sigma, gamma, anorm, bb, p, r_oldgam, rte_oldc, rt1_c, rt2_s;

    while(l1 <= n && iters < max_iters)
    {
        // Determine submatrix indices
        if(l1 > 1)
            E[l1 - 1 - 1] = 0;
        for(m = l1; m <= n - 1; m++)
        {
            if(abs(E[m - 1]) <= sqrt(abs(D[m - 1])) * sqrt(abs(D[m + 1 - 1])) * eps)
            {
                E[m - 1] = 0;
                break;
            }
        }

        lsv = l = l1;
        lendsv = lend = m;
        l1 = m + 1;
        if(lend == l)
            continue;

        // Scale submatrix
        sterf_find_max(l, lend, D, E, anorm);
        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            sterf_scale(l, lend, D, E, anorm / ssfmax);
        else if(anorm < ssfmin)
            sterf_scale(l, lend, D, E, anorm / ssfmin);
        sterf_sq_e(l, lend, E);

        // Choose iteration type (QL or QR)
        if(abs(D[lend - 1]) < abs(D[l - 1]))
        {
            lend = lsv;
            l = lendsv;
        }

        if(lend >= l)
        {
            // QL iteration
            while(l <= lend && iters < max_iters)
            {
                // Find small subdiagonal element
                for(m = l; m <= lend - 1; m++)
                    if(abs(E[m - 1]) <= eps * eps * abs(D[m - 1] * D[m + 1 - 1]))
                        break;

                if(m < lend)
                    E[m - 1] = 0;
                p = D[l - 1];
                if(m == l)
                {
                    D[l - 1] = p;
                    l++;
                }
                else if(m == l + 1)
                {
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    rte_oldc = sqrt(E[l - 1]);
                    lae2(D[l - 1], rte_oldc, D[l + 1 - 1], rt1_c, rt2_s);
                    D[l - 1] = rt1_c;
                    D[l + 1 - 1] = rt2_s;
                    E[l - 1] = 0;
                    l = l + 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    // Form shift. Using rte, r, c, s.
                    rte_oldc = sqrt(E[l - 1]);
                    sigma = (D[l + 1 - 1] - p) / (2 * rte_oldc);
                    if(sigma >= 0)
                        r_oldgam = abs(sqrt(1 + sigma * sigma));
                    else
                        r_oldgam = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte_oldc / (sigma + r_oldgam));

                    rt1_c = 1;
                    rt2_s = 0;
                    gamma = D[m - 1] - sigma;
                    p = gamma * gamma;

                    for(int i = m - 1; i >= l; i--)
                    {
                        bb = E[i - 1];
                        r_oldgam = p + bb;
                        if(i != m - 1)
                            E[i + 1 - 1] = rt2_s * r_oldgam;

                        // Using oldc, r, c, s.
                        rte_oldc = rt1_c;
                        rt1_c = p / r_oldgam;
                        rt2_s = bb / r_oldgam;

                        // Using oldc, oldgam, c, s.
                        r_oldgam = gamma;
                        gamma = rt1_c * (D[i - 1] - sigma) - rt2_s * r_oldgam;
                        D[i + 1 - 1] = r_oldgam + (D[i - 1] - gamma);
                        if(rt1_c != 0)
                            p = (gamma * gamma) / rt1_c;
                        else
                            p = rte_oldc * bb;
                    }

                    E[l - 1] = rt2_s * p;
                    D[l - 1] = sigma + gamma;
                }
            }
        }

        else
        {
            // QR iteration
            while(l >= lend && iters < max_iters)
            {
                // Find small subdiagonal element
                for(m = l; m >= lend + 1; m--)
                    if(abs(E[m - 1 - 1]) <= eps * eps * abs(D[m - 1] * D[m - 1 - 1]))
                        break;

                if(m > lend)
                    E[m - 1 - 1] = 0;
                p = D[l - 1];
                if(m == l)
                {
                    D[l - 1] = p;
                    l--;
                }
                else if(m == l - 1)
                {
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    rte_oldc = sqrt(E[l - 1 - 1]);
                    lae2(D[l - 1], rte_oldc, D[l - 1 - 1], rt1_c, rt2_s);
                    D[l - 1] = rt1_c;
                    D[l - 1 - 1] = rt2_s;
                    E[l - 1 - 1] = 0;
                    l = l - 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    // Form shift. Using rte, r, c, s.
                    rte_oldc = sqrt(E[l - 1 - 1]);
                    sigma = (D[l - 1 - 1] - p) / (2 * rte_oldc);
                    if(sigma >= 0)
                        r_oldgam = abs(sqrt(1 + sigma * sigma));
                    else
                        r_oldgam = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte_oldc / (sigma + r_oldgam));

                    rt1_c = 1;
                    rt2_s = 0;
                    gamma = D[m - 1] - sigma;
                    p = gamma * gamma;

                    for(int i = m; i <= l - 1; i++)
                    {
                        bb = E[i - 1];
                        r_oldgam = p + bb;
                        if(i != m)
                            E[i - 1 - 1] = rt2_s * r_oldgam;

                        // Using oldc, r, c, s.
                        rte_oldc = rt1_c;
                        rt1_c = p / r_oldgam;
                        rt2_s = bb / r_oldgam;

                        // Using oldc, oldgam, c, s.
                        r_oldgam = gamma;
                        gamma = rt1_c * (D[i + 1 - 1] - sigma) - rt2_s * r_oldgam;
                        D[i - 1] = r_oldgam + (D[i + 1 - 1] - gamma);
                        if(rt1_c != 0)
                            p = (gamma * gamma) / rt1_c;
                        else
                            p = rte_oldc * bb;
                    }

                    E[l - 1 - 1] = rt2_s * p;
                    D[l - 1] = sigma + gamma;
                }
            }
        }

        // Undo scaling
        if(anorm > ssfmax)
            sterf_scale(lsv, lendsv, D, E, ssfmax / anorm);
        if(anorm < ssfmin)
            sterf_scale(lsv, lendsv, D, E, ssfmin / anorm);
    }

    // Check for convergence
    for(int i = 1; i <= n - 1; i++)
        if(E[i - 1] != 0)
            info[bid]++;

    // Sort eigenvalues
    lasrt_increasing(n, D, stack + bid * (2 * 32));
}

template <typename T>
void rocsolver_sterf_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_stack = 0;
        return;
    }

    // size of stack (for lasrt)
    *size_stack = sizeof(rocblas_int) * (2 * 32) * batch_count;
}

template <typename T>
rocblas_status rocsolver_sterf_argCheck(const rocblas_int n, T D, T E, rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_sterf_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* stack)
{
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
    if(n <= 1)
        return rocblas_status_success;

    T eps = get_epsilon<T>();
    T ssfmin = get_safemin<T>();
    T ssfmax = T(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / T(3.0);

    hipLaunchKernelGGL(sterf_kernel<T>, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
                       strideD, E + shiftE, strideE, info, stack, 30 * n, eps, ssfmin, ssfmax);

    return rocblas_status_success;
}

#endif
