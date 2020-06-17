/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_BDSQR_H
#define ROCLAPACK_BDSQR_H

#include "rocblas.hpp"
#include "rocsolver.h"
#include "common_device.hpp"


/** LARTG device function computes the sine (s) and cosine (c) values 
    to create a givens rotation such that:
    [  c s ]' * [ f ] = [ r ]
    [ -s c ]    [ g ]   [ 0 ] **/
template <typename T>
__device__ void lartg(T &f, T &g, T &c, T &s, T &r)
{
    T t;
    if (std::abs(g) > std::abs(f)) {
        t = -f/g;
        s = 1 / std::sqrt(1 + t*t);
        c = s * t;
    } else {
        t = -g/f;
        c = 1 / std::sqrt(1 + t*t);
        s = c * t;
    }
    r = std::sqrt(f*f + g*g);
} 


/** LOWER2UPPER kernel transforms a lower bidiagonal matrix given by D and E
    into an upper bidiagonal matrix via givens rotations **/
template <typename T, typename W1, typename W2>
__global__ void lower2upper(const rocblas_int n,
                            const rocblas_int nu,
                            const rocblas_int nc,
                            W1* DD, const rocblas_stride strideD,
                            W1* EE, const rocblas_stride strideE,
                            W2 UU, const rocblas_int shiftU,
                            const rocblas_int ldu, const rocblas_stride strideU,
                            W2 CC, const rocblas_int shiftC,
                            const rocblas_int ldc, const rocblas_stride strideC)
{
    rocblas_int bid = hipBlockIdx_x;
    W1 f, g, c, s, r;

    // select batch instance to work with
    W1* D = DD + bid*strideD;
    W1* E = EE + bid*strideE;

    f = D[0];
    g = E[0];
    for (rocblas_int i = 0; i < n-1; ++i) {
        // apply rotations by rows
        lartg(f,g,c,s,r);
        D[i] = r;
        E[i] = -s*D[i+1];
        f = c*D[i+1];
        g = E[i+1];
    }       
    D[n-1] = f; 
}


/** ESTIMATE device function computes an estimate of the smallest
    singular value of a n-by-n upper bidiagonal matrix given by D and E 
    It also applies convergence test if conver = 1 **/
template <typename T>
__device__ T estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver)
{
    T smin = t2b ? D[0] : D[n-1];
    T t = smin;

    rocblas_int je, jd;

    for (rocblas_int i = 1; i < n; ++i) {
        jd = t2b ? i : n-1-i;
        je = jd - t2b;
        if ((std::abs(D[jd]) <= tol*t) && conver) {
            smin = -1;
            break;
        }
        t = std::abs(D[jd])*t / (t + std::abs(E[je]));
        smin = (t < smin) ? t : smin;
    } 

    return smin;
}      


/** MAXVAL device function extracts the maximum absolute value  
    element of all the n elements of vector V **/
template <typename T>
__device__ T maxval(const rocblas_int n, T* V)
{
    T maxv = std::abs(V[0]);
    for (rocblas_int i = 1; i < n; ++i)
        maxv = (std::abs(V[i]) > maxv) ? std::abs(V[i]) : maxv;

    return maxv; 
}


/** BDSQRKERNEL implements the main loop of the bdsqr algorithm 
    to compute the SVD of an upper bidiagonal matrix given by D and E **/
template <typename T, typename W1, typename W2>
__global__ void Kernel(const rocblas_int n,
                            const rocblas_int nu,
                            const rocblas_int nc,
                            W1* DD, const rocblas_stride strideD,
                            W1* EE, const rocblas_stride strideE,
                            W2 VV, const rocblas_int shiftV,
                            const rocblas_int ldv, const rocblas_stride strideV,
                            W2 UU, const rocblas_int shiftU,
                            const rocblas_int ldu, const rocblas_stride strideU,
                            W2 CC, const rocblas_int shiftC,
                            const rocblas_int ldc, const rocblas_stride strideC,
                            rocblas_int *info, const rocblas_int maxiter) 
//                            const W1 eps, const W1 sfm, const W1 tol, W1 minshift)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance to work with
    W1* D = DD + bid*strideD;
    W1* E = EE + bid*strideE;

/*    // calculate threshold for zeroing elements (convergence threshold)
    int t2b = (D[0] >= D[n-1]) ? 1 : 0;                 //direction
    W1 smin = estimate<W1>(n,D,E,t2b,tol,0);            //estimate of the smallest singular value 
    W1 thresh = std::max(tol*smin/W1(std::sqrt(n)),
                W1(maxiter)*sfm);                       //threshold
    W1 smax = std::max(maxval<W1>(n,D), 
              maxval<W1>(n-1,E));                       //estimate of the largest singular value
    minshift *= smax;                                   //minimum accepted value for the shift

    rocblas_int k = n-1;    //k is the last element of last unconverged diagonal block 
    rocblas_int iter = 0;   //iter is the number of iterations (QR steps) applied
    rocblas_int i;
    W1 sh;

    // main loop
    while (k > 0 && iter < maxiter) {
        
        // split the diagonal blocks
        for (rocblas_int j = 0; j < k+1; ++j) {
            i = k-j-1;
            if (i >= 0 && E[i] < thresh) {
                E[i] = 0;
                break;
            }                
        }

        // check is last singular value converged, 
        // if not, continue with the QR step
        if (i == k-1) k--;
        else {

            // last block goes from i+1 until k
            // determine shift for the QR step
            // (apply convergence test to find gaps)
            i++;
            if (D[i] >= D[k]) {
                t2b = 1;
                sh = D[i];
            } else {
                t2b = 0;
                sh = D[k];
            } 
            smin = estimate<W1>(k-i+1,D+i,E+k,t2b,tol,1);   //shift

            // check for gaps, if none then continue
            if (smin >=0) {                             
                if (smin <= minshift) smin = 0;             //shift set to zero if less than accepted value 
                else if (sh > 0) {
                    if (smin*smin/sh/sh < eps) smin = 0;    //shift set to zero if negligible
                }

                // apply QR step
                iter += k-i;    
//                if (t2b) t2bQRstep();
//                else b2tQRstep();
            }
        }
    }
    D[n-1]=minshift;
    D[n-2]=tol;
    D[n-3]=eps;
    D[n-4]=thresh;
    D[n-5]=iter;*/
}


template <typename W1, typename W2>
rocblas_status rocsolver_bdsqr_argCheck(const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nv,
                                       const rocblas_int nu,
                                       const rocblas_int nc,
                                       const rocblas_int ldv,
                                       const rocblas_int ldu,
                                       const rocblas_int ldc,
                                       W1   D,
                                       W1   E,
                                       W2   V,
                                       W2   U,
                                       W2   C,
                                       rocblas_int *info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if (n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if ((nv > 0 && ldv < n) || (nc > 0 && ldc < n))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !D) || (n > 1 && !E) || (n*nv && !V) || (n*nu && !U) || (n*nc && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename W1, typename W2>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nv,
                                           const rocblas_int nu,
                                           const rocblas_int nc,
                                           W1* D, const rocblas_stride strideD,
                                           W1* E, const rocblas_stride strideE,
                                           W2 V, const rocblas_int shiftV,
                                           const rocblas_int ldv, const rocblas_stride strideV,
                                           W2 U, const rocblas_int shiftU,
                                           const rocblas_int ldu, const rocblas_stride strideU,
                                           W2 C, const rocblas_int shiftC,
                                           const rocblas_int ldc, const rocblas_stride strideC,
                                           rocblas_int *info,
                                           const rocblas_int batch_count)
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set tolerance and max number of iterations
//    W1 eps = get_epsilon<W1>() / 2;         //machine precision (considering rounding strategy)
//    W1 sfm = get_safemin<W1>();             //safest minimum value such that 1/sfm does not overflow 
    rocblas_int maxiter = 6*n*n;            //max number of iterations (QR steps) before declaring not convergence
//    W1 tol = std::max(W1(10.0),
//             std::min(W1(100.0),
//             W1(pow(eps,-0.125)))) * eps;   //relative accuracy tolerance  
//    W1 minshift = std::max(eps,
//                  tol/W1(100)) / (n*tol);   //(minimum accepted shift to not ruin relative accuracy) / (max singular value)

//    printf("\n %2.25f %2.25f %2.25f\n",eps,tol,minshift);

    // rotate to upper bidiagonal if necessary 
    if (uplo == rocblas_fill_lower) {
       hipLaunchKernelGGL(lower2upper<T>,dim3(batch_count),dim3(1),0,stream,
                          n, nu, nc, D, strideD, E, strideE, 
                          U, shiftU, ldu, strideU,
                          C, shiftC, ldc, strideC);
    }                             

    // main computation of SVD
    hipLaunchKernelGGL(Kernel<T>,dim3(batch_count),dim3(1),0,stream,
                       n, nu, nc, D, strideD, E, strideE, 
                       V, shiftV, ldv, strideV,
                       U, shiftU, ldu, strideU,
                       C, shiftC, ldc, strideC, 
                       info, maxiter); 
//                       W1(eps), W1(sfm), W1(tol), W1(minshift));
    
    return rocblas_status_success;
}

#endif /* ROCLAPACK_BDSQR_H */
