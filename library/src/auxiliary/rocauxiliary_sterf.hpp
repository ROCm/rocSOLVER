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
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <fstream>
#include <stdio.h>

/****************************************************************************
(TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH).)
***************************************************************************/

#ifdef LAPACK_FUNCTIONS
/** direct call sterf from LAPACK **/
template <typename T>
void lapack_sterf(rocblas_int n, T* D, T* E, int &info);
#endif

/** STERF_SQ_E squares the elements of E **/
template <typename T>
__device__ void sterf_sq_e(const rocblas_int start, const rocblas_int end, T* E)
{
    for(int i = start; i < end; i++)
        E[i] = E[i] * E[i];
}

template <typename T>
void host_sterf_sq_e(const rocblas_int start, const rocblas_int end, T* E)
{
    for(int i = start; i < end; i++)
        E[i] = E[i] * E[i];
}

/** STERF_CPU implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E on CPUi, non batched version**/
template <typename T>
void sterf_cpu(const rocblas_int n,
               T* D,
               T* E,
               rocblas_int& info,
               const rocblas_int max_iters,
               const T eps,
               const T ssfmin,
               const T ssfmax)
{
    rocblas_int m, l, lsv, lend, lendsv;
    rocblas_int l1 = 0;
    rocblas_int iters = 0;
    T anorm, p;

    while(l1 < n && iters < max_iters)
    {
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
        anorm = host_find_max_tridiag(l, lend, D, E);

        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            host_scale_tridiag(l, lend, D, E, anorm / ssfmax);
        else if(anorm < ssfmin)
            host_scale_tridiag(l, lend, D, E, anorm / ssfmin);
        host_sterf_sq_e(l, lend, E);

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
                {
                    if(abs(E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                    {
                        break;
                    }
                }

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
                    T rte, rt1, rt2;
                    rte = sqrt(E[l]);
                    host_lae2(D[l], rte, D[l + 1], rt1, rt2);
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

                    T sigma, gamma, r, rte, c, s;

                    // Form shift
                    rte = sqrt(E[l]);
                    sigma = (D[l + 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = abs(sqrt(1 + sigma * sigma));
                    else
                        r = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m - 1; i >= l; i--)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m - 1)
                            E[i + 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i] - sigma) - s * oldgam;
                        D[i + 1] = oldgam + (D[i] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l] = s * p;
                    D[l] = sigma + gamma;
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
                {
                    if(abs(E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                    {
                        break;
                    }
                }

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
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    T rte, rt1, rt2;
                    rte = sqrt(E[l - 1]);
                    host_lae2(D[l], rte, D[l - 1], rt1, rt2);
                    D[l] = rt1;
                    D[l - 1] = rt2;
                    E[l - 1] = 0;
                    l = l - 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    T sigma, gamma, r, rte, c, s;

                    // Form shift. Using rte, r, c, s.
                    rte = sqrt(E[l - 1]);
                    sigma = (D[l - 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = abs(sqrt(1 + sigma * sigma));
                    else
                        r = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m; i <= l - 1; i++)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m)
                            E[i - 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i + 1] - sigma) - s * oldgam;
                        D[i] = oldgam + (D[i + 1] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l - 1] = s * p;
                    D[l] = sigma + gamma;
                }
            }
        }

        // Undo scaling
        if(anorm > ssfmax)
            host_scale_tridiag(lsv, lendsv, D, E, ssfmax / anorm);
        if(anorm < ssfmin)
            host_scale_tridiag(lsv, lendsv, D, E, ssfmin / anorm);
    }

    // Check for convergence
    for(int i = 0; i < n - 1; i++)
        if(E[i] != 0)
            info++;

    // Sort eigenvalues
    /** (TODO: the quick-sort method implemented in lasrt_increasing fails for some cases.
        Substituting it here with a simple sorting algorithm. If more performance is required in
        the future, lasrt_increasing should be debugged or another quick-sort method
        could be implemented) **/
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
        }
    }
}

/** STERF_KERNEL implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T>
ROCSOLVER_KERNEL void sterf_kernel(const rocblas_int n,
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
    rocblas_int l1 = 0;
    rocblas_int iters = 0;
    T anorm, p;

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
        sterf_sq_e(l, lend, E);

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
                {
                    if(abs(E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                    {
                        break;
                    }
                }

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
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    T rte, rt1, rt2;
                    rte = sqrt(E[l]);
                    lae2(D[l], rte, D[l + 1], rt1, rt2);
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

                    T sigma, gamma, r, rte, c, s;

                    // Form shift
                    rte = sqrt(E[l]);
                    sigma = (D[l + 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = abs(sqrt(1 + sigma * sigma));
                    else
                        r = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m - 1; i >= l; i--)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m - 1)
                            E[i + 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i] - sigma) - s * oldgam;
                        D[i + 1] = oldgam + (D[i] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l] = s * p;
                    D[l] = sigma + gamma;
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
                {
                    if(abs(E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                    {
                        break;
                    }
                }

                //printf("Finished check subranges\n");
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
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    T rte, rt1, rt2;
                    rte = sqrt(E[l - 1]);
                    lae2(D[l], rte, D[l - 1], rt1, rt2);
                    D[l] = rt1;
                    D[l - 1] = rt2;
                    E[l - 1] = 0;
                    l = l - 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    T sigma, gamma, r, rte, c, s;

                    // Form shift. Using rte, r, c, s.
                    rte = sqrt(E[l - 1]);
                    sigma = (D[l - 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = abs(sqrt(1 + sigma * sigma));
                    else
                        r = -abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m; i <= l - 1; i++)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m)
                            E[i - 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i + 1] - sigma) - s * oldgam;
                        D[i] = oldgam + (D[i + 1] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l - 1] = s * p;
                    D[l] = sigma + gamma;
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
            info[bid]++;

    /// into another kernel, check time
    // Sort eigenvalues
    /** (TODO: the quick-sort method implemented in lasrt_increasing fails for some cases.
        Substituting it here with a simple sorting algorithm. If more performance is required in
        the future, lasrt_increasing should be debugged or another quick-sort method
        could be implemented) **/
    //lasrt_increasing(n, D, stack + bid * (2 * 32));

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
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void sterf_find_subranges_gre(const rocblas_int n,
                                               T* DD,
                                               const rocblas_int offsetD,
                                               T* EE,
                                               const rocblas_int offsetE,
                                               const T eps,
                                               rocblas_int* split_ranges)
{
    T* D = DD;
    T* E = EE;

    int m = 0, l = 0;

    rocblas_int range_count = 0;

    T Eold = 0;
    T Emax = 0;
    T GL, GU;
    T tnrm = 0;

    for(int i = 0; i < n; ++i)
    {
        T Eabs = abs(E[i]);
        if(Eabs >= Emax)
            Emax = Eabs;

        T tmp = Eabs + Eold;
        GL = min(GL, D[i] - tmp);
        GU = max(GU, D[i] + tmp);
        Eold = Eabs;
    }

    /// spectral diametr
    tnrm = GU - GL;

    while(l < n)
    {
        if(l > 0)
            E[l - 1] = 0;

        for(m = l; m < n - 1; m++)
        {
            if(abs(E[m]) <= tnrm * eps)
            {
                E[m] = 0;
                break;
            }
        }

        if(l != m)
        {
            split_ranges[range_count] = l;
            ++range_count;
            split_ranges[range_count] = m;
            ++range_count;
        }

        l = m + 1;
    }

    for(int i = range_count; i < n; ++i)
        split_ranges[i] = -1;
}

template <typename T>
ROCSOLVER_KERNEL void sterf_find_subranges_default(const rocblas_int n,
                                                   T* DD,
                                                   const rocblas_int offsetD,
                                                   T* EE,
                                                   const rocblas_int offsetE,
                                                   const T eps,
                                                   rocblas_int* split_ranges)
{
    T* D = DD;
    T* E = EE;

    int m = 0, l = 0;

    rocblas_int range_count = 0;

    while(l < n)
    {
        if(l > 0)
            E[l - 1] = 0;

        for(m = l; m < n - 1; m++)
        {
            if(abs(E[m]) <= sqrt(abs(D[m])) * sqrt(abs(D[m + 1])) * eps)
            {
                E[m] = 0;
                break;
            }
        }

        if(l != m)
        {
            split_ranges[range_count] = l;
            ++range_count;
            split_ranges[range_count] = m;
            ++range_count;
        }

        l = m + 1;
    }

    for(int i = range_count; i < n; ++i)
        split_ranges[i] = -1;
}

/// default parallel kernel
template <typename T>
ROCSOLVER_KERNEL void sterf_parallelize(T* D,
                                        T* E,
                                        const rocblas_int max_iter,
                                        const T eps,
                                        const T ssfmax,
                                        const T ssfmin,
                                        rocblas_int* split_ranges,
                                        rocblas_int* info)
{
    rocblas_int m = 0;
    rocblas_int count = 0, l = -1, lend = -1;
    rocblas_int l_orig, lend_orig;
    T p, anorm;

    const rocblas_int tid = hipThreadIdx_x;

    l_orig = l = split_ranges[2 * tid];
    lend_orig = lend = split_ranges[2 * tid + 1];

    if(l == -1 || lend == -1)
        return;

    anorm = find_max_tridiag(l, lend, D, E);

    if(anorm == 0)
        return;
    else if(anorm > ssfmax)
        scale_tridiag(l, lend, D, E, anorm / ssfmax);
    else if(anorm < ssfmin)
        scale_tridiag(l, lend, D, E, anorm / ssfmin);
    sterf_sq_e(l, lend, E);

    // Choose iteration type (QL or QR)
    if(abs(D[lend]) < abs(D[l]))
    {
        lend = l;
        l = lend_orig;
    }

    rocblas_int iters = 0;
    if(lend >= l)
    {
        // for QL
        while(l <= lend && iters < max_iter)
        {
            // Find small subdiagonal element (QL)
            for(m = l; m <= lend - 1; m++)
            {
                if(abs(E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                {
                    break;
                }
            }

            if(m < lend)
                E[m] = 0;

            p = D[l];
            if(m == l)
            {
                ++l;
                continue;
            }
            else if(m == l + 1)
            {
                T rte, rt1, rt2;
                rte = sqrt(E[l]);
                lae2(D[l], rte, D[l + 1], rt1, rt2);
                D[l] = rt1;
                D[l + 1] = rt2;
                E[l] = 0;
                l = l + 2;
                continue;
            }
            else
            {
                if(iters == max_iter)
                    break;
                ++iters;
                T sigma, gamma, r, rte, c, s;

                // Form shift
                rte = sqrt(E[l]);
                sigma = (D[l + 1] - p) / (2 * rte);

                if(sigma >= 0)
                    r = abs(sqrt(1 + sigma * sigma));
                else
                    r = -abs(sqrt(1 + sigma * sigma));
                sigma = p - (rte / (sigma + r));

                c = 1;
                s = 0;
                gamma = D[m] - sigma;
                p = gamma * gamma;

                for(int i = m - 1; i >= l; i--)
                {
                    T bb = E[i];
                    r = p + bb;
                    if(i != m - 1)
                        E[i + 1] = s * r;

                    T oldc = c;
                    c = p / r;
                    s = bb / r;
                    T oldgam = gamma;
                    gamma = c * (D[i] - sigma) - s * oldgam;
                    D[i + 1] = oldgam + (D[i] - gamma);

                    if(c != 0)
                        p = (gamma * gamma) / c;
                    else
                        p = oldc * bb;
                }

                E[l] = s * p;
                D[l] = sigma + gamma;
            }
        }
    }
    else
    {
        //for QR
        while(l >= lend && iters < max_iter)
        {
            // Find small subdiagonal element
            for(m = l; m >= lend + 1; m--)
            {
                if(abs(E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                {
                    break;
                }
            }

            if(m > lend)
                E[m - 1] = 0;

            p = D[l];
            if(m == l)
            {
                --l;
                continue;
            }
            else if(m == l - 1)
            {
                T rte, rt1, rt2;
                rte = sqrt(E[l - 1]);
                lae2(D[l], rte, D[l - 1], rt1, rt2);
                D[l] = rt1;
                D[l - 1] = rt2;
                E[l - 1] = 0;
                l = l - 2;
                continue;
            }
            else
            {
                if(iters == max_iter)
                    break;
                ++iters;
                T sigma, gamma, r, rte, c, s;

                // Form shift. Using rte, r, c, s.
                rte = sqrt(E[l - 1]);
                sigma = (D[l - 1] - p) / (2 * rte);
                if(sigma >= 0)
                    r = abs(sqrt(1 + sigma * sigma));
                else
                    r = -abs(sqrt(1 + sigma * sigma));
                sigma = p - (rte / (sigma + r));

                c = 1;
                s = 0;
                gamma = D[m] - sigma;
                p = gamma * gamma;

                for(int i = m; i <= l - 1; i++)
                {
                    T bb = E[i];
                    r = p + bb;
                    if(i != m)
                        E[i - 1] = s * r;

                    T oldc = c;
                    c = p / r;
                    s = bb / r;
                    T oldgam = gamma;
                    gamma = c * (D[i + 1] - sigma) - s * oldgam;
                    D[i] = oldgam + (D[i + 1] - gamma);
                    if(c != 0)
                        p = (gamma * gamma) / c;
                    else
                        p = oldc * bb;
                }

                E[l - 1] = s * p;
                D[l] = sigma + gamma;
            }
        }
    }

    if(anorm > ssfmax)
        scale_tridiag(l_orig, lend_orig, D, E, ssfmax / anorm);
    if(anorm < ssfmin)
        scale_tridiag(l_orig, lend_orig, D, E, ssfmin / anorm);

    for(int i = l_orig; i <= lend_orig; i++)
        if(E[i] != 0)
            info[0]++;
}

template <typename T>
ROCSOLVER_KERNEL void sterf_sorting(const rocblas_int n, T* D)
{
    rocblas_int l, m;
    T p;

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
        }
    }
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
void rocsolver_sterf_parallel_getMemorySize(const rocblas_int n, size_t* size_ranges)
{
    // if quick return no workspace needed
    if(n == 0)
    {
        *size_ranges = 0;
        return;
    }

    *size_ranges = sizeof(rocblas_int) * (n / 2);
}

template <typename T>
rocblas_status
    rocsolver_sterf_argCheck(rocblas_handle handle, const rocblas_int n, T D, T E, rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

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
    ROCSOLVER_ENTER("sterf", "n:", n, "shiftD:", shiftD, "shiftE:", shiftE, "bc:", batch_count);

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
    if(n <= 1)
        return rocblas_status_success;

    T eps = get_epsilon<T>();
    T ssfmin = get_safemin<T>();
    T ssfmax = T(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / T(3.0);

#ifdef HYBRID_CPU

    for(int i = 0; i < batch_count; ++i)
    {
        T* h_D = new T[n];
        T* h_E = new T[n];
        rocblas_int h_info = 0;

        hipStreamSynchronize(stream);

        T* shD = D + i * strideD + shiftD;
        T* shE = E + i * strideE + shiftE;

        // copy to CPU
        hipMemcpy(h_D, shD, sizeof(T) * n, hipMemcpyDeviceToHost);
        hipMemcpy(h_E, shE, sizeof(T) * n, hipMemcpyDeviceToHost);

#ifdef LAPACK_FUNCTIONS
        lapack_sterf(n, h_D, h_E, h_info);
#else
        sterf_cpu<T>(n, h_D, h_E, h_info, 30 * n, eps, ssfmin, ssfmax);
#endif

        hipMemcpy(shD, h_D, sizeof(T) * n, hipMemcpyHostToDevice);
        hipMemcpy(shE, h_E, sizeof(T) * n, hipMemcpyHostToDevice);
        hipMemcpy(info + i, &h_info, sizeof(rocblas_int), hipMemcpyHostToDevice);

        delete[] h_D;
        delete[] h_E;
    }

#elif EXPERIMENTAL

    int max_threads = 1024;
    int CU_count = (n - 1) / max_threads + 1;
    int thread_count = max_threads;

    dim3 grid(CU_count, 1, 1);
    dim3 block(thread_count, 1, 1);

    int offsetD = thread_count;
    int offsetE = thread_count;
    rocblas_int* split_ranges = stack;

    size_t lmemsize = n * sizeof(int) + sizeof(T);

    /// find ranges for sterf
    ROCSOLVER_LAUNCH_KERNEL(sterf_find_subranges_default<T>, dim3(1), dim3(1), 0, stream, n,
                            D + shiftD, offsetD, E + shiftE, offsetE, eps, ranges_m);

    /// execute parallel sterf
    CU_count = (n - 2) / (2 * max_threads) + 1;
    thread_count = n > max_threads ? max_threads : n / 2;

    dim3 sterf_grid(CU_count, 1, 1);
    dim3 sterf_block(thread_count, 1, 1);

    ROCSOLVER_LAUNCH_KERNEL(sterf_parallelize<T>, dim3(CU_count), dim3(thread_count), 0, stream, D,
                            E, n, eps, ssfmax, ssfmin, ranges_m, info);

    ROCSOLVER_LAUNCH_KERNEL(sterf_sorting<T>, dim3(1), dim3(1), 0, stream, n, D);

#else

    ROCSOLVER_LAUNCH_KERNEL(sterf_kernel<T>, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
                            strideD, E + shiftE, strideE, info, stack, 30 * n, eps, ssfmin, ssfmax);
#endif

    return rocblas_status_success;
}
