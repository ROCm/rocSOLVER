/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/****************************************************************************
(TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH).)
***************************************************************************/

/** STEQR_KERNEL/RUN_STEQR implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T, typename S>
__device__ void run_steqr(const rocblas_int n,
                          S* D,
                          S* E,
                          T* C,
                          const rocblas_int ldc,
                          rocblas_int* info,
                          S* work,
                          const rocblas_int max_iters,
                          const S eps,
                          const S ssfmin,
                          const S ssfmax,
                          const bool ordered = true)
{
    rocblas_int m, l, lsv, lend, lendsv;
    rocblas_int l1 = 0;
    rocblas_int iters = 0;
    S anorm, p;

    while(l1 < n && iters < max_iters)
    {
        // Determine submatrix indices
        if(l1 > 0)
            E[l1 - 1] = 0;
        for(m = l1; m < n - 1; m++)
        {
            if(abs(E[m]) <= sqrt(abs(D[m])) * sqrt(abs(D[m + 1])) * eps)
            {
                E[m] = 0;
                break;
            }
        }

        lsv = l = l1;
        lendsv = lend = m;
        l1 = m + 1;
        if(lend == l)
            continue;

        // Scale submatrix
        anorm = find_max_tridiag(l, lend, D, E);
        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            scale_tridiag(l, lend, D, E, anorm / ssfmax);
        else if(anorm < ssfmin)
            scale_tridiag(l, lend, D, E, anorm / ssfmin);

        // Choose iteration type (QL or QR)
        if(abs(D[lend]) < abs(D[l]))
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
                    if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                        break;

                if(m < lend)
                    E[m] = 0;
                p = D[l];
                if(m == l)
                {
                    D[l] = p;
                    l++;
                }
                else if(m == l + 1)
                {
                    // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                    S rt1, rt2, c, s;
                    laev2(D[l], E[l], D[l + 1], rt1, rt2, c, s);
                    work[l] = c;
                    work[n - 1 + l] = s;
                    lasr(rocblas_side_right, rocblas_backward_direction, n, 2, work + l,
                         work + n - 1 + l, C + 0 + l * ldc, ldc);

                    D[l] = rt1;
                    D[l + 1] = rt2;
                    E[l] = 0;
                    l = l + 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    S f, g, c, s, b, r;

                    // Form shift
                    g = (D[l + 1] - p) / (2 * E[l]);
                    if(g >= 0)
                        r = abs(sqrt(1 + g * g));
                    else
                        r = -abs(sqrt(1 + g * g));
                    g = D[m] - p + (E[l] / (g + r));

                    c = 1;
                    s = 1;
                    p = 0;

                    for(int i = m - 1; i >= l; i--)
                    {
                        f = s * E[i];
                        b = c * E[i];
                        lartg(g, f, c, s, r);
                        s = -s; //get the transpose of the rotation
                        if(i != m - 1)
                            E[i + 1] = r;

                        g = D[i + 1] - p;
                        r = (D[i] - g) * s + 2 * c * b;
                        p = s * r;
                        D[i + 1] = g + p;
                        g = c * r - b;

                        // Save rotations
                        work[i] = c;
                        work[n - 1 + i] = -s;
                    }

                    // Apply saved rotations
                    lasr(rocblas_side_right, rocblas_backward_direction, n, m - l + 1, work + l,
                         work + n - 1 + l, C + 0 + l * ldc, ldc);

                    D[l] -= p;
                    E[l] = g;
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
                    if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                        break;

                if(m > lend)
                    E[m - 1] = 0;
                p = D[l];
                if(m == l)
                {
                    D[l] = p;
                    l--;
                }
                else if(m == l - 1)
                {
                    // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                    S rt1, rt2, c, s;
                    laev2(D[l - 1], E[l - 1], D[l], rt1, rt2, c, s);
                    work[m] = c;
                    work[n - 1 + m] = s;
                    lasr(rocblas_side_right, rocblas_forward_direction, n, 2, work + m,
                         work + n - 1 + m, C + 0 + (l - 1) * ldc, ldc);

                    D[l - 1] = rt1;
                    D[l] = rt2;
                    E[l - 1] = 0;
                    l = l - 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    S f, g, c, s, b, r;

                    // Form shift
                    g = (D[l - 1] - p) / (2 * E[l - 1]);
                    if(g >= 0)
                        r = abs(sqrt(1 + g * g));
                    else
                        r = -abs(sqrt(1 + g * g));
                    g = D[m] - p + (E[l - 1] / (g + r));

                    c = 1;
                    s = 1;
                    p = 0;

                    for(int i = m; i <= l - 1; i++)
                    {
                        f = s * E[i];
                        b = c * E[i];
                        lartg(g, f, c, s, r);
                        s = -s; //get the transpose of the rotation
                        if(i != m)
                            E[i - 1] = r;

                        g = D[i] - p;
                        r = (D[i + 1] - g) * s + 2 * c * b;
                        p = s * r;
                        D[i] = g + p;
                        g = c * r - b;

                        // Save rotations
                        work[i] = c;
                        work[n - 1 + i] = s;
                    }

                    // Apply saved rotations
                    lasr(rocblas_side_right, rocblas_forward_direction, n, l - m + 1, work + m,
                         work + n - 1 + m, C + 0 + m * ldc, ldc);

                    D[l] -= p;
                    E[l - 1] = g;
                }
            }
        }

        // Undo scaling
        if(anorm > ssfmax)
            scale_tridiag(lsv, lendsv, D, E, ssfmax / anorm);
        if(anorm < ssfmin)
            scale_tridiag(lsv, lendsv, D, E, ssfmin / anorm);
    }

    // Check for convergence
    for(int i = 0; i < n - 1; i++)
        if(E[i] != 0)
            info[0]++;

    // Sort eigenvalues and eigenvectors by selection sort
    if(ordered)
    {
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
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void steqr_kernel(const rocblas_int n,
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
                                   const rocblas_int max_iters,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax)
{
    // select bacth instance
    rocblas_int bid = hipBlockIdx_x;
    rocblas_stride strideW = 2 * n - 2;

    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    T* C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* work = WW + (bid * strideW);
    rocblas_int* info = iinfo + bid;

    // execute
    run_steqr(n, D, E, C, ldc, info, work, max_iters, eps, ssfmin, ssfmax);
}

template <typename T, typename S>
void rocsolver_steqr_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work_stack = 0;
        return;
    }

    // size of stack (for lasrt)
    if(evect == rocblas_evect_none)
        *size_work_stack = sizeof(rocblas_int) * (2 * 32) * batch_count;
    else
        *size_work_stack = sizeof(S) * (2 * n - 2) * batch_count;
}

template <typename T, typename S>
rocblas_status rocsolver_steqr_argCheck(rocblas_handle handle,
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

template <typename T, typename S, typename U>
rocblas_status rocsolver_steqr_template(rocblas_handle handle,
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
                                        void* work_stack)
{
    ROCSOLVER_ENTER("steqr", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

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

    // Initialize identity matrix
    if(evect == rocblas_evect_tridiagonal)
    {
        rocblas_int blocks = (n - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                                stream, n, n, C, shiftC, ldc, strideC);
    }

    S eps = get_epsilon<S>();
    S ssfmin = get_safemin<S>();
    S ssfmax = S(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / S(3.0);

    if(evect == rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(sterf_kernel<S>, dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, E + shiftE, strideE, info,
                                (rocblas_int*)work_stack, 30 * n, eps, ssfmin, ssfmax);
    else
        ROCSOLVER_LAUNCH_KERNEL((steqr_kernel<T>), dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, E + shiftE, strideE, C, shiftC, ldc, strideC,
                                info, (S*)work_stack, 30 * n, eps, ssfmin, ssfmax);

    return rocblas_status_success;
}
