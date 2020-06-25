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
    if (g == 0) {
        c = 1;
        s = 0;
    } else {
        T t;
        if (std::abs(g) > std::abs(f)) {
            t = -f/g;
            s = 1 / T(std::sqrt(1 + t*t));
            c = s * t;
        } else {
            t = -g/f;
            c = 1 / T(std::sqrt(1 + t*t));
            s = c * t;
        }
    }
    r = c*f - s*g;
} 


/** LASR device function applies a sequence of rotations P(i) i=1,2,...z 
    to a m-by-n matrix A from either the left (P*A with z=m) or the right (A*P' with z=n).
    P = P(z-1)*...*P(1) if forward direction, 
    P = P(1)*...*P(z-1) if backward direction. **/
template <typename T, typename W>
__device__ void lasr(const rocblas_side side, 
                     const rocblas_direct direc,
                     const rocblas_int m,
                     const rocblas_int n,
                     W* c, W* s,
                     T* A, const rocblas_int lda)
{
    T temp;
    W cs, sn;

    if (side == rocblas_side_left) {
        if (direc == rocblas_forward_direction) {
            for (rocblas_int i = 0; i < m-1; ++i) {
                for (rocblas_int j = 0; j < n; ++j) {
                    temp = A[i + j*lda];
                    cs = c[i];
                    sn = s[i];
                    A[i + j*lda] = cs*temp + sn*A[i+1 + j*lda];
                    A[i+1 + j*lda] = cs*A[i+1 + j*lda] - sn*temp;
                }
            }

        } else {
            for (rocblas_int i = m-1; i > 0; --i) {
                for (rocblas_int j = 0; j < n; ++j) {
                    temp = A[i + j*lda];
                    cs = c[i];
                    sn = s[i];
                    A[i + j*lda] = cs*temp - sn*A[i-1 + j*lda];
                    A[i-1 + j*lda] = cs*A[i-1 + j*lda] + sn*temp;
                }
            }
        }
    }

    else {
        if (direc == rocblas_forward_direction) {
            for (rocblas_int j = 0; j < n-1; ++j) {
                for (rocblas_int i = 0; i < m; ++i) {
                    temp = A[i + j*lda];
                    cs = c[j];
                    sn = s[j];
                    A[i + j*lda] = cs*temp + sn*A[i + (j+1)*lda];
                    A[i + (j+1)*lda] = cs*A[i + (j+1)*lda] - sn*temp;
                }
            }

        } else {
            for (rocblas_int j = n-1; j > 0; --j) {
                for (rocblas_int i = 0; i < m; ++i) {
                    temp = A[i + j*lda];
                    cs = c[j];
                    sn = s[j];
                    A[i + j*lda] = cs*temp - sn*A[i + (j-1)*lda];
                    A[i + (j-1)*lda] = cs*A[i + (j-1)*lda] + sn*temp;
                }
            }
        }
    }
}


/** ESTIMATE device function computes an estimate of the smallest
    singular value of a n-by-n upper bidiagonal matrix given by D and E 
    It also applies convergence test if conver = 1 **/
template <typename T>
__device__ T estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver)
{
    T smin = t2b ? std::abs(D[0]) : std::abs(D[n-1]);
    T t = smin;

    rocblas_int je, jd;

    for (rocblas_int i = 1; i < n; ++i) {
        jd = t2b ? i : n-1-i;
        je = jd - t2b;
        if ((std::abs(E[je]) <= tol*t) && conver) {
            E[je] = 0;
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


/** NEGVECT device function multiply a vector x of dimension n by -1**/
template <typename T>
__device__ void negvect(const rocblas_int n, T *x, const rocblas_int incx)
{
    for (rocblas_int i = 0; i < n; ++i)
    {
        if (x[incx * i] != 0) x[incx * i] = -x[incx * i];
    }
}



/** SWAPVECT device function swap vectors a and b of dimension n **/
template <typename T>
__device__ void swapvect(const rocblas_int n, T *a, const rocblas_int inca,
                        T *b, const rocblas_int incb)
{
    T orig;
    for (rocblas_int i = 0; i < n; ++i)
    {
        orig = a[inca * i];
        a[inca * i] = b[incb * i];
        b[incb * i] = orig;
    }
}


/** T2BQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from top to bottom **/
template <typename W1, typename W2>
__device__ void t2bQRstep(const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          const rocblas_int nc,
                          W1* D, W1* E, 
                          W2* V, const rocblas_int ldv, 
                          W2* U, const rocblas_int ldu, 
                          W2* C, const rocblas_int ldc, 
                          const W1 sh, W1* rots)
{
    W1 f, g, c, s, r;
    rocblas_int nr = nv ? 2*(n-1) : 0;

    int sgn = (W1(0) < D[0]) - (D[0] < W1(0)); 
    if (D[0] == 0) f = 0;
    else f = (std::abs(D[0]) - sh) * (W1(sgn) + sh/D[0]); 
    g = E[0];

    for (rocblas_int k = 0; k < n-1; ++k) { 
        // first apply rotation by columns
        lartg(f,g,c,s,r);
        if (k > 0) E[k-1] = r;
        f = c*D[k] - s*E[k];
        E[k] = c*E[k] + s*D[k];
        g = -s*D[k+1];
        D[k+1] = c*D[k+1];
        // save rotations to update singular vectors
        if (nv) {
            rots[k] = c;
            rots[k+n-1] = -s;
        }

        // then apply rotation by rows
        lartg(f,g,c,s,r);
        D[k] = r;
        f = c*E[k] - s*D[k+1];
        D[k+1] = c*D[k+1] + s*E[k];
        if (k < n-2) {
            g = -s*E[k+1];
            E[k+1] = c*E[k+1];
        }    
        // save rotations to update singular vectors
        if (nu || nc) {
            rots[k+nr] = c;
            rots[k+nr+n-1] = -s;
        }
    } 
    E[n-2] = f;

    // update singular vectors
    if (nv)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nv, rots, rots+n-1, V, ldv);
    if (nu)
        lasr(rocblas_side_right, rocblas_forward_direction, nu, n, rots+nr, rots+nr+n-1, U, ldu);
    if (nc)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nc, rots+nr, rots+nr+n-1, C, ldc);
}


/** B2TQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from bottom to top **/
template <typename W1, typename W2>
__device__ void b2tQRstep(const rocblas_int n,
                          const rocblas_int nv,
                          const rocblas_int nu,
                          const rocblas_int nc,
                          W1* D, W1* E, 
                          W2* V, const rocblas_int ldv, 
                          W2* U, const rocblas_int ldu, 
                          W2* C, const rocblas_int ldc, 
                          const W1 sh, W1* rots)
{
    W1 f, g, c, s, r;
    rocblas_int nr = nv ? 2*(n-1) : 0;

    int sgn = (W1(0) < D[n-1]) - (D[n-1] < W1(0));
    if (D[n-1] == 0) f = 0;
    else f = (std::abs(D[n-1]) - sh) * (W1(sgn) + sh/D[n-1]);
    g = E[n-2];

    for (rocblas_int k = n-1; k > 0; --k) {
        // first apply rotation by rows
        lartg(f,g,c,s,r);
        if (k < n-1) E[k] = r;
        f = c*D[k] - s*E[k-1];
        E[k-1] = c*E[k-1] + s*D[k];
        g = -s*D[k-1];
        D[k-1] = c*D[k-1];
        // save rotations to update singular vectors
        if (nu || nc) {
            rots[k+nr] = c;
            rots[k+nr+n-1] = s;
        }

        // then apply rotation by columns
        lartg(f,g,c,s,r);
        D[k] = r;
        f = c*E[k-1] - s*D[k-1];
        D[k-1] = c*D[k-1] + s*E[k-1];
        if (k > 1) {
            g = -s*E[k-2];
            E[k-2] = c*E[k-2];
        }
        // save rotations to update singular vectors
        if (nv) {
            rots[k] = c;
            rots[k+n-1] = s;
        }
    }
    E[0] = f;
    
    // update singular vectors
    if (nv)
        lasr(rocblas_side_left, rocblas_backward_direction, n, nv, rots, rots+n-1, V, ldv);
    if (nu)
        lasr(rocblas_side_right, rocblas_backward_direction, nu, n, rots+nr, rots+nr+n-1, U, ldu);
    if (nc)
        lasr(rocblas_side_left, rocblas_backward_direction, n, nc, rots+nr, rots+nr+n-1, C, ldc);
}


/** BDSQRKERNEL implements the main loop of the bdsqr algorithm 
    to compute the SVD of an upper bidiagonal matrix given by D and E **/
template <typename T, typename W1, typename W2>
__global__ void bdsqrKernel(const rocblas_int n,
                            const rocblas_int nv,
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
                            rocblas_int *info, const rocblas_int maxiter, 
                            const W1 eps, const W1 sfm, const W1 tol, const W1 minshift,
                            W1* workA, const rocblas_stride strideW)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance to work with
    W1* rots = workA + bid*strideW;
    W1* D = DD + bid*strideD;
    W1* E = EE + bid*strideE;
    T* V = load_ptr_batch<T>(VV,bid,shiftV,strideV);
    T* U = load_ptr_batch<T>(UU,bid,shiftU,strideU);
    T* C = load_ptr_batch<T>(CC,bid,shiftC,strideC);

    // calculate threshold for zeroing elements (convergence threshold)
    int t2b = (D[0] >= D[n-1]) ? 1 : 0;                 //direction
    W1 smin = estimate<W1>(n,D,E,t2b,tol,0);            //estimate of the smallest singular value 
    W1 thresh = std::max(tol*smin/W1(std::sqrt(n)),
                W1(maxiter)*sfm);                       //threshold

    rocblas_int k = n-1;    //k is the last element of last unconverged diagonal block 
    rocblas_int iter = 0;   //iter is the number of iterations (QR steps) applied
    rocblas_int i;
    W1 sh, smax;

    // main loop
    while (k > 0 && iter < maxiter) {
        // split the diagonal blocks
        for (rocblas_int j = 0; j < k+1; ++j) {
            i = k-j-1;
            if (i >= 0 && std::abs(E[i]) < thresh) {
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
            if (std::abs(D[i]) >= std::abs(D[k])) {
                t2b = 1;
                sh = std::abs(D[i]);
            } else {
                t2b = 0;
                sh = std::abs(D[k]);
            } 
            smin = estimate<W1>(k-i+1,D+i,E+i,t2b,tol,1);   //shift
            smax = std::max(maxval<W1>(k-i+1,D+i), 
                            maxval<W1>(k-i,E+i));           //estimate of the largest singular value in the block

            // check for gaps, if none then continue
            if (smin >=0) {                             
                if (smin/smax <= minshift) smin = 0;        //shift set to zero if less than accepted value 
                else if (sh > 0) {
                    if (smin*smin/sh/sh < eps) smin = 0;    //shift set to zero if negligible
                }

                // apply QR step
                iter += k-i;    
                if (t2b) t2bQRstep(k-i+1, nv, nu, nc, D+i, E+i, V+i, ldv,
                                   U+i*ldu, ldu, C+i, ldc, smin, rots);
                else b2tQRstep(k-i+1, nv, nu, nc, D+i, E+i, V+i, ldv,
                               U+i*ldu, ldu, C+i, ldc, smin, rots);
            }
        }
    }
    
    // re-arange singular values/vectors if algorithm converged
    if (k == 0) {
        // all positive
        for (rocblas_int ii = 0; ii < n; ++ii) {
            if (D[ii] < 0) {
                D[ii] = -D[ii];
                if (nv) negvect(nv, V+ii, ldv);
            }
        }    

        // in drecreasing order
        rocblas_int idx;
        for (rocblas_int ii = 0; ii < n-1; ++ii) {
            idx = ii;
            smax = D[ii];
            // detect maximum
            for (rocblas_int jj = ii+1; jj < n; ++jj) {
                if (D[jj] > smax) {
                    idx = jj;
                    smax = D[jj];
                }
            }
            // swap
            if (idx != ii) {
                D[idx] = D[ii];
                D[ii] = smax;
                if (nv) swapvect(nv, V+idx, ldv, V+ii, ldv);
                if (nu) swapvect(nu, U+idx*ldu, 1, U+ii*ldu, 1);
                if (nc) swapvect(nc, C+idx, ldc, C+ii, ldc);
            }
        }
    }

    // if not, set value of info
    else {
        info[bid] = 0;
        for (rocblas_int i = 0; i < n-1; ++i)
            if (E[i] != 0) info[bid] += 1;
    }
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
                            const rocblas_int ldc, const rocblas_stride strideC,
                            W1* workA, const rocblas_stride strideW)
{
    rocblas_int bid = hipBlockIdx_x;
    W1 f, g, c, s, r;

    // select batch instance to work with
    W1* rots = workA + bid*strideW;
    W1* D = DD + bid*strideD;
    W1* E = EE + bid*strideE;
    T* U = load_ptr_batch<T>(UU,bid,shiftU,strideU);
    T* C = load_ptr_batch<T>(CC,bid,shiftC,strideC);

    f = D[0];
    g = E[0];
    for (rocblas_int i = 0; i < n-1; ++i) {
        // apply rotations by rows
        lartg(f,g,c,s,r);
        D[i] = r;
        E[i] = -s*D[i+1];
        f = c*D[i+1];
        g = E[i+1];

        // save rotation to update singular vectors
        if (nu || nc) {
            rots[i] = c;
            rots[i+n-1] = -s;
        }
    }       
    D[n-1] = f; 

    // update singular vectors
    if (nu) 
        lasr(rocblas_side_right, rocblas_forward_direction, nu, n, rots, rots+n-1, U, ldu);
    if (nc)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nc, rots, rots+n-1, C, ldc);
}


template <typename T>
void rocsolver_bdsqr_getMemorySize(const rocblas_int n, const rocblas_int nv, const rocblas_int nu, const rocblas_int nc, 
                                   const rocblas_int batch_count, size_t *size)
{
    // size of workspace
    *size = 0;
    if (nv) *size += 2;
    if (nu || nc) *size += 2;
    
    *size *= sizeof(T)*(n-1)*batch_count;
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
                                           const rocblas_int batch_count, W1* work)
{
    // quick return
    if (n == 0 || batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set tolerance and max number of iterations
    W1 eps = get_epsilon<W1>() / 2;         //machine precision (considering rounding strategy)
    W1 sfm = get_safemin<W1>();             //safest minimum value such that 1/sfm does not overflow 
    rocblas_int maxiter = 6*n*n;            //max number of iterations (QR steps) before declaring not convergence
    W1 tol = std::max(W1(10.0),
             std::min(W1(100.0),
             W1(pow(eps,-0.125)))) * eps;   //relative accuracy tolerance  
    W1 minshift = std::max(eps,
                  tol/W1(100)) / (n*tol);   //(minimum accepted shift to not ruin relative accuracy) / (max singular value)

    rocblas_stride strideW = 4*n;

    // rotate to upper bidiagonal if necessary 
    if (uplo == rocblas_fill_lower) {
       hipLaunchKernelGGL((lower2upper<T>),dim3(batch_count),dim3(1),0,stream,
                          n, nu, nc, D, strideD, E, strideE, 
                          U, shiftU, ldu, strideU,
                          C, shiftC, ldc, strideC, work, strideW);
    }                             

    // main computation of SVD
    hipLaunchKernelGGL((bdsqrKernel<T>),dim3(batch_count),dim3(1),0,stream,
                       n, nv, nu, nc, D, strideD, E, strideE, 
                       V, shiftV, ldv, strideV,
                       U, shiftU, ldu, strideU,
                       C, shiftC, ldc, strideC, 
                       info, maxiter,
                       eps, sfm, tol, minshift, work, strideW);
    
    return rocblas_status_success;
}

#endif /* ROCLAPACK_BDSQR_H */
