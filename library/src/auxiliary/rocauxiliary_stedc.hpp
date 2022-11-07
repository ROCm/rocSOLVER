/************************************************************************
  Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#define BDIM 512  // Number of threads per block used in main stedc kernel
#define MAXITERS 100 // Max number of iterations for Newton's method


/** This function evaluates the secular equation (rational function f) at x. 
    It updates fx with f(x) and fdx with f'(x).
    Returns -1 if f(x) < 0, and 1 otherwise **/
/*template <typename S>
__device__ void seq_eval(const rocblas_int din,
                         const rocblas_int dout,
                         const rocblas_int dd,
                         const S* D,
                         const S* z,
                         const S p,
                         const S x,
                         S* fx, 
                         S* fdx) 
{
    S vald = 0;
    S zz, den;
    S val = (dout - din == dd) ? p : 0;
    int inc = (din < dout) ? 1 : -1;
    int i = din;

    while(i != dout)
    {
        zz = z[i] * z[i];
        den = D[i] - x;
        val += zz / den;
        vald += zz / (den * den);
        i += inc;
    }

    *fx = val;
    *fdx = vald;
}*/


template <typename S>
__device__ void seq_eval(const rocblas_int type,
                         const rocblas_int k,
                         const rocblas_int dd,
                         const S* D,
                         const S* z,
                         const S p,
                         const S x,
                         S* pt_fx, 
                         S* pt_fdx,
                         S* pt_gx,
                         S* pt_gdx,
                         S* pt_hx,
                         S* pt_hdx,
                         S* pt_er,
                         bool print) 
{
    S er, fx, gx, hx, fdx, gdx, hdx, zz, tmp;
    rocblas_int gout, hout;

    // type = 0: evaluates secular equation
    if(type == 0)
    {
        gout = k + 1;
        hout = k;
    }
    // type = 1: evaluates secular equation without the k-th pole
    else if(type == 1) 
    {
        gout = k;
        hout = k;
    }
    // type = 2: evaluates secular equation without the k-th and (k+1)-th poles
    else
    {
        gout = k;
        hout = k + 1;
    }
    
    gx = 0;
    gdx = 0;
    er = 0;
    for(int i = 0; i < gout; ++i)
    {
        zz = z[i];
        tmp = zz / (D[i] - x);
        gx += zz * tmp;
        gdx += tmp * tmp;
        er += gx;
    }
    er = abs(er);

    hx = 0;
    hdx = 0;
    for(int i = dd - 1; i > hout; --i)
    {
        zz = z[i];
        tmp = zz / (D[i] - x);
        hx += zz * tmp;
        hdx += tmp * tmp;
        er += hx;
    }

    fx = p + gx + hx;
    fdx = gdx + hdx;

    *pt_fx = fx;
    *pt_fdx = fdx;
    *pt_gx = gx;
    *pt_gdx = gdx;
    *pt_hx = hx;
    *pt_hdx = hdx;
}

/*template <typename S>
__device__ rocblas_int seq_solve(const rocblas_int dd, 
                                    const S* D,
                                    const S* z,
                                    const S p, 
                                    rocblas_int k, 
                                    S* ev, 
                                    const S tol, 
                                    const S ssfmin,
                                    const S ssfmax,
                                    bool print)
{
    bool converged = false;
    bool up;
    S lowb, uppb, aa, bb, cc, x;
    S nx, er, fx, fdx, gx, gdx, hx, hdx;
    S tau, eta;
    S dk, dk1, pinv;
    roblas_int kk;
    rocblas_int k1 = k + 1;
    
    // initialize
    dk = D[k];
    dk1 = D[k1];
    x = (dk + dk1) / 2; // midpoint of interval
    tau = (dk1 - dk);
    pinv = 1 / p;    
    

    // find bounds and initial guess; translate origin
    seq_eval(2, k, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er);
    gdx = z[k] * z[k];
    hdx = z[k1] * z[k1];
    fx = cc + gdx / (dk - x) + hdx / (dk1 - x);
    if(fx > 0)
    {   
        // if the secular eq at the midpoint is positive, the root is in between D[k] and the midpoint
        // take D[k] as the origin, i.e. x = D[k] + tau with tau in (0, uppb)
        lowb = 0;
        uppb = tau / 2;        
        up = true;
        kk = k; // origin remains the same
        
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(abs(aa * aa - 4 * bb * cc));
        if(aa > 0)
           tau = 2 * bb / (aa + eta);
        else
           tau = (aa - eta) / (2 * cc);
        x = dk + tau; // initial guess
    }
    else
    { 
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0) 
        lowb = -tau / 2;
        uppb = 0;
        up = false;
        kk = k + 1; // translate the origin

        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(abs(aa * aa + 4 * bb * cc));
        if(aa < 0)
           tau = 2 * bb / (aa - eta);
        else
           tau = -(aa + eta) / (2 * cc);
        x = dk1 + tau; // initial guess
    }
    
    // calculate tolerance er for convergence test
    seq_eval(1, kk, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er);
    bb = z[kk];
    aa = bb / (D[kk] - x); 
    fdx += aa * aa;
    aa *= bb;
    fx = cc + aa;
    er += 8 * (hx - gx) + 2 * pinv + 3 * abs(aa) + abs(tau) * fdx;

    // if the value of secular eq is small enough, no point to continue; converged!!!
    if(abs(fx) <= tol * er)
        converged = true;
    
    // otherwise...
    else
    {
            
    }

    *ev = x;
    return converged ? 0 : 1;    
}*/





/** Basic implementation of hybrid Newton-Raphson + bisection (Newt-safe).
    SEQ_NEWTSAFE computes a root 'r' of the secular equation f (i.e. f(r) = 0) within the
    initial interval [a,b]. It updates ev with r. Returns 1 if failed to converge, 0 otherwise **/
template <typename S>
__device__ rocblas_int seq_newtsafe(const rocblas_int dd, 
                                    const S* D,
                                    const S* z,
                                    const S p, 
                                    rocblas_int k, 
                                    S* ev, 
                                    const S tol, 
                                    const S ssfmin,
                                    const S ssfmax,
                                    bool print)
{
    bool converged = false;
    bool ext = (k == dd - 1);
    S a, b, aa, bb, cc, x;
    S nx, fx, fdx, er, gx, gdx, hx, hdx;    
    S cor, cor0;
    S dk, dk1;

    // find bounds and initial approximation
    a = D[k];
    if(ext)
    {
        b = a + abs(a);
        k = dd - 2;
    }
    else
    {
        b = D[k + 1];
    }
    x = (a + b) / 2;

    for(int i = 0; i < MAXITERS; ++i)
    {
        // find the full correction 'cor' to 'x' (new step would be x + cor)
        dk = D[k] - x;
        dk1 = D[k + 1] - x;
        seq_eval(0, k, dd, D, z, p, x, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, print);                
        aa = (dk + dk1) * fx - dk * dk1 * fdx;
        bb = dk * dk1 * fx;
        if(ext)
        {
            cc = fx - dk * gdx - z[dd-1] * z[dd-1] / dk1;
            cor = sqrt(aa * aa - 4 * bb * cc);
            cor = (aa >= 0) ? (aa + cor) / (2 * cc) : (2 * bb) / (aa - cor);
        }
        else
        {
            cc = fx - dk * gdx - dk1 * hdx;
            cor = sqrt(aa * aa - 4 * bb * cc);
            cor = (aa <= 0) ? (aa - cor) / (2 * cc) : (2 * bb) / (aa + cor);
        }
        nx = x + cor;
        er = abs(x - nx);

        // if the new x would be indistinguishable, the algorithm converged!!!
        if(er / max(abs(x), abs(nx)) <= tol)
        {
            converged = true;
            break;
        }

        // otherwise...
        else
        {
            // verify that the correction is not too large 
            // (take a damped correction if necessary)
            if(nx <= a)
                x = (a + x) / 2;
            else if(!ext && nx >= b)
                x = (x + b) / 2;
            else
                x = nx;
        }
    }
    
    *ev = x;
    return converged ? 0 : 1;    
}

/** STEDC_NUM_LEV returns the ideal number of times or levels a matrix (or split block)
    will be divided during the divide phase of divide & conquer algorithm.
    i.e. number of sub-blocks = 2^levels **/
__host__ __device__ inline rocblas_int stedc_num_levs(const rocblas_int n)
{
    return 1;
}

/** STEDC_SPLIT finds independent blocks in the tridiagonal matrix
    given by D and E. (The independent blocks can then be solved in
    parallel by the DC algorithm) **/
template <typename S>
ROCSOLVER_KERNEL void stedc_split(const rocblas_int n,
                                  S* DD,
                                  const rocblas_stride strideD,
                                  S* EE,
                                  const rocblas_stride strideE,
                                  rocblas_int* splitsA,
                                  const S eps)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* splits = splitsA + bid * (2 * n + 2);

    rocblas_int k = 0; //position where the last block starts
    S tol; //tolerance. If an element of E is <= tol we have an independent block
    rocblas_int bs; //size of an independent block
    rocblas_int nb = 1; //number of blocks
    splits[0] = 0; //positions where each block begings

    // main loop
    while(k < n)
    {
        bs = 1;
        for(rocblas_int j = k; j < n - 1; ++j)
        {
            tol = eps * sqrt(abs(D[j])) * sqrt(abs(D[j + 1]));
            if(abs(E[j]) < tol)
            {
                // Split next independent block
                // save its location in matrix
                splits[nb] = j+1;
                nb++;
                break;
            }
            bs++;
        }
        k += bs;
    }
    splits[nb] = n;
    splits[n + 1] = nb; //also save the number of split blocks
}

/** STEDC_KERNEL implements the main loop of the DC algorithm
    to compute the eigenvalues/eigenvectors of the symmetric tridiagonal
    submatrices **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(BDIM) 
                        stedc_kernel(const rocblas_int n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   S* CC,
                                   const rocblas_int shiftC,
                                   const rocblas_int ldc,
                                   const rocblas_stride strideC,
                                   rocblas_int* iinfo,
                                   S* WA,
                                   S* tmpzA,
                                   S* vecsA,
                                   rocblas_int* splitsA,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax,
                                   const rocblas_int maxblks)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_y; 
    // split block id
    rocblas_int sid = hipBlockIdx_x; 
    rocblas_int id = hipThreadIdx_x; 
    rocblas_int tid, tidb;
    /* --------------------------------------------------- */


    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;
    /* --------------------------------------------------- */
    
   
    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (2 * n + 2); 
    // worksapce 
    S* W = WA + bid * (2 * n);  
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = splits + n + 2;  
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);  
    // roots of secular equations          
    S* evs = z + n; 
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges 
    S* temps = vecs + (n * n);       
    /* --------------------------------------------------- */


    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lmem[];
    // shares the sub-blocks sizes
    rocblas_int* ns = lmem; 
    // shares the sub-blocks initial positions
    rocblas_int* ps = ns + maxblks; 
    // used to temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(ps + maxblks); 
    /* --------------------------------------------------- */


    // local variables
    /* --------------------------------------------------- */
    // total number of blocks
    rocblas_int nb = splits[n + 1]; 
    // size of split block
    rocblas_int bs; 
    // begining of split block
    rocblas_int p1; 
    // begining of sub-block
    rocblas_int p2; 
    // number of sub-blocks
    rocblas_int blks; 
    // number of level of division
    rocblas_int levs; 
    S p;
    /* --------------------------------------------------- */


    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levs(bs);
        blks = 1 << levs;

        // if split block is too small, solve it with steqr
        if(blks == 1)
        {
            if(id == 0)
            {
                run_steqr(bs, D + p1, E + p1, C + p1 + p1 * ldc, ldc, info, W + p1 * 2, 30 * bs,
                          eps, ssfmin, ssfmax, false);
            }
        }

        // otherwise, divide & conquer 
        else
        {
            // arrange threads so that a group of bdim/blks threads works with each sub-block
            rocblas_int tn = BDIM / blks;
            tid = id / tn;
            tidb = id % tn;
             
            // 1. DIVIDE PHASE
            /* ----------------------------------------------------------------- */
            // (artificially divide split block into blks sub-blocks
            // find initial positions of each sub-blocks)
            if(tidb == 0)
                ns[tid] = 0;

            // find sub-block sizes
            if(id == 0)
            {
                ns[0] = bs;
                rocblas_int t, t2;
                for(int i = 0; i < levs; ++i)
                {
                    for(int j = (1 << i); j > 0; --j)
                    {
                        t = ns[j - 1];
                        t2 = t / 2;
                        ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                        ns[j * 2 - 2] = t2;
                    }
                }
            }
            __syncthreads();

            // find begining of sub-block and update D elements
            p2 = 0;
            for(int i = 0; i < tid; ++i)
                p2 += ns[i];
            p2 += p1;
            if(tidb == 0)
            {
                ps[tid] = p2;
                if(tid > 0 && tid < blks)
                {
                    // perform sub-block division
                    p = E[p2 - 1];
                    D[p2] -= p;
                    D[p2 - 1] -= p;
                }
            }
            __syncthreads();
            /* ----------------------------------------------------------------- */

            
            // 2. SOLVE PHASE
            /* ----------------------------------------------------------------- */
            // (solve the blks sub-blocks in parallel)
            if(tidb == 0)
            {
                run_steqr(ns[tid], D + p2, E + p2, C + p2 + p2 * ldc, ldc, info,
                          W + p2 * 2, 30 * bs, eps, ssfmin, ssfmax, false);
            }
            __syncthreads();
            /* ----------------------------------------------------------------- */


            // 3. MERGE PHASE
            /* ----------------------------------------------------------------- */
            rocblas_int iam, sz, bdm;
            S* ptz;

            // main loop to work with merges on each level
            // (once two leaves in the merge tree are identified, all related threads
            // work together to solve the secular equation and update eiegenvectors)
            for(int k = 0; k < levs; ++k)
            {
                // 3a. find rank-1 modification components (z and p) for this merge
                // (threads with iam < bd work with components above the merge point;
                //  threads with iam >= bd work below the merge point)
                /* ----------------------------------------------------------------- */
                rocblas_int bd = 1 << k;
                bdm = bd << 1;
                iam = tid % bdm;
                if(iam < bd && tid < blks)
                {
                    sz = ns[tid];
                    for(int j = 1; j < bd - iam; ++j)
                        sz += ns[tid + j];
                    // with this, all threads involved in a merge (above merge point)
                    // will point to the same row of C and the same off-diag element
                    ptz = C + p2 - 1 + sz;
                    p = E[p2 - 1 + sz];
                }
                else if(iam >= bd && tid < blks)
                {
                    sz = 0;
                    for(int j = 0; j < iam - bd; ++j)
                        sz += ns[tid - j - 1];
                    // with this, all threads involved in a merge (below merge point)
                    // will point to the same row of C and the same off-diag element
                    ptz = C + p2 - sz;
                    p = E[p2 - sz - 1];
                }
                // copy elements of z
                if(tidb == 0)
                {
                    for(int j = 0; j < ns[tid]; ++j)
                        z[p2 + j] = ptz[(p2 + j) * ldc];
                }
                __syncthreads();
                /* ----------------------------------------------------------------- */


                // 3b. calculate deflation tolerance
                // tol = 8 * eps * (max diagonal or z element participating in merge)
                /* ----------------------------------------------------------------- */
                S valf, valg, f, g, c, s, r;
                S tol;// = 0.00001;
                tol = k == 0 ? 0.000000001 : 0.000000001;
                /* ----------------------------------------------------------------- */


                // 3c. deflate enigenvalues
                /* ----------------------------------------------------------------- */
                // first deflate each thread sub-block
                if(tidb == 0)
                {
                    for(int i = 0; i < ns[tid]; ++i)
                    {
                        g = z[p2 + i];
                        if(abs(p * g) <= tol)
                        {
                            // deflated ev because component in z is zero
                            idd[p2 + i] = 0;
                        }
                        else
                        {
                            rocblas_int jj = 1;
                            valg = D[p2 + i];
                            for(int j = 0; j < i; ++j)
                            {
                                if(idd[p2 + j] == 1 && abs(D[p2 + j] - valg) <= tol)
                                {
                                    // deflated ev because it is repeated
                                    idd[p2 + i] = 0;
                                    // rotation to eliminate component in z
                                    f = z[p2 + j];
                                    lartg(f, g, c, s, r);
                                    z[p2 + j] = r;
                                    z[p2 + i] = 0;
                                    // update C with the rotation
                                    for(int ii = 0; ii < n; ++ii)
                                    {
                                        valf = C[ii + (p2 + j) * ldc];
                                        valg = C[ii + (p2 + i) * ldc];
                                        C[ii + (p2 + j) * ldc] = valf * c - valg * s;
                                        C[ii + (p2 + i) * ldc] = valf * s + valg * c;
                                    }
                                    break;
                                }
                                jj++;
                            }
                            if(jj > i)
                            {
                                // non-deflated ev
                                idd[p2 + i] = 1;
                            }
                        }
                    }
                }
                __syncthreads();

                // then compare with other sub-blocks participating in this merge
                // (follows a simple, reduction-like process)
                for(int ii = 0; ii <= k; ++ii)
                {
                    if(tidb == 0)
                    {
                        rocblas_int div = 1 << (ii + 1);
                        //actual number of threads is halved each time
                        if(iam % div == div - 1) 
                        {
                            // find limits
                            rocblas_int inb = (1 << ii) - 1;
                            rocblas_int inc = div - 1;
                            rocblas_int countb = ns[tid];
                            rocblas_int countc = 0;
                            for(int i = inc; i > inb; --i)
                                countc += ns[tid - i];
                            for(int i = inb; i > 0; --i)
                                countb += ns[tid - i];
                            inb = ps[tid - inb];
                            inc = ps[tid - inc];

                            // perform comparisons
                            for(int i = 0; i < countb; ++i)
                            {
                                if(idd[inb + i] == 1)
                                {
                                    valg = D[inb + i];
                                    for(int j = 0; j < countc; ++j)
                                    {
                                        if(idd[inc + j] == 1 && abs(D[inc + j] - valg) <= tol)
                                        {
                                            // deflated ev because it is repeated
                                            idd[inb + i] = 0;
                                            // rotation to eliminate component in z
                                            g = z[inb + i];
                                            f = z[inc + j];
                                            lartg(f, g, c, s, r);
                                            z[inc + j] = r;
                                            z[inb + i] = 0;
                                            // update C with the rotation
                                            for(int ii = 0; ii < n; ++ii)
                                            {
                                                valf = C[ii + (inc + j) * ldc];
                                                valg = C[ii + (inb + i) * ldc];
                                                C[ii + (inc + j) * ldc] = valf * c - valg * s;
                                                C[ii + (inb + i) * ldc] = valf * s + valg * c;
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                /* ----------------------------------------------------------------- */
                
                
                // 3d. Organize data with non-deflated values to prepare secular equation
                /* ----------------------------------------------------------------- */
                // determine boundaries in D
                rocblas_int in = ps[tid - iam];
                sz = ns[tid];
                for(int i = iam; i > 0; --i)
                    sz += ns[tid - i];
                for(int i = bdm - 1 - iam; i > 0; --i)
                    sz += ns[tid + i];

                // define shifted arrays 
                S* tmpd = temps + in;
                S* tmpz = tmpd + n;
                S* ev = evs + in;
                S* diag = D + in;
                rocblas_int* mask = idd + in;
                S* zz = z + in;
                rocblas_int dd = 0; // degree of secular equation
                
                // 
                for(int i = 0; i < sz; ++i)
                {
                    if(mask[i] == 1)
                    {
                        if(tidb == 0 && iam == 0)
                        {
                            tmpd[dd] = diag[i];
                            tmpz[dd] = zz[i];
                        }
                        dd++;
                    }
                }
                __syncthreads();
                
                // put final polynomial in poly and diagonal elements in ev
                iam = iam * tn + tidb;
                bdm *= tn;
                for(int i = iam; i < sz; i += bdm)
                    ev[i] = diag[i];
                __syncthreads();
                /* ----------------------------------------------------------------- */
                
                
                // 3e. Solve secular eqns, i.e. find the dd roots 
                // corresponding to non-deflated eigenvalues
                /* ----------------------------------------------------------------- */
                // first, order elements in temps to find initial intervals for the roots
                // (using a simple parallel selection/bubble sort)
                for(int i = 0; i < dd; ++i)
                {
                    if(i % 2 == 0)
                    {
                        for(int j = iam; j < dd / 2; j += bdm) 
                        {
                            if(tmpd[2 * j] > tmpd[2 * j + 1])
                            {
                                valf = tmpd[2 * j];
                                tmpd[2 * j] = tmpd[2 * j + 1];
                                tmpd[2 * j + 1] = valf; 
                                valf = tmpz[2 * j];
                                tmpz[2 * j] = tmpz[2 * j + 1];
                                tmpz[2 * j + 1] = valf; 
                            }
                        }
                    }
                    else
                    {
                        for(int j = iam; j < (dd - 1) / 2; j += bdm)
                        {
                            if(tmpd[2 * j + 1] > tmpd[2 * j + 2])
                            {
                                valf = tmpd[2 * j + 1];
                                tmpd[2 * j + 1] = tmpd[2 * j + 2];
                                tmpd[2 * j + 2] = valf;
                                valf = tmpz[2 * j + 1];
                                tmpz[2 * j + 1] = tmpz[2 * j + 2];
                                tmpz[2 * j + 2] = valf;
                            }    
                        }
                    }
                    __syncthreads();
                }

                // now, each thread will find a different root in parallel
                S a, b; 
                for(int i = iam; i < sz; i += bdm)
                {
                    if(mask[i] == 1)
                    {
                        // determine initial interval; root is in (a, b)
                        int cc = 0;
                        valf = ev[i];
                        for(int j = 0; j < dd; ++j)
                        {
                            if(tmpd[j] == valf)
                                break;
                            else
                                cc++;
                        }

                        // find root within the given initial interval
                        // (Using a basic implementation of a Newt-safe algorithm.
                        // TODO: We may want to investigate other root-finding methods, or
                        // optimize the Newt-safe in the future if necessary)
bool print = false;
                        rocblas_int linfo = seq_newtsafe(dd, tmpd, tmpz, 1/p, cc, ev + i, eps, ssfmin, ssfmax, print);
                    }
                }
                __syncthreads();
                /* ----------------------------------------------------------------- */


                // 3f. Compute vectors corresponding to non-deflated values
                /* ----------------------------------------------------------------- */
                // Re-arrange vector Z to avoid bad numerics when ev[i] is close to D[i]
                for(int i = iam; i < sz; i += bdm)
                {
                    if(mask[i] == 1)
                    { 
                        valf = (diag[i] - ev[i]);            
                        for(int j = 0; j < sz; ++j)
                        {
                            if(mask[j] == 1 && j != i)
                                valf *= (diag[i] - ev[j]) / (diag[i] - diag[j]);            
                        }
                        valf = sqrt(-valf);
                        zz[i] = zz[i] < 0 ? -valf : valf;
                    }
                }
                __syncthreads();    
                                                        
                // g. Compute vectors corresponding to non-deflated values
                S temp, nrm, evj;
                rocblas_int nn = (n - 1) / blks + 1;
                bool go;
                for(int j = 0; j < nn; ++j)
                {
                    go = (j < ns[tid] && idd[p2 + j] == 1);

                    // compute vectors and norms
                    nrm = 0;
                    if(go)
                    {
                        evj = evs[p2 + j];
                        for(int i = in + tidb; i < in + sz; i += tn)
                        {
                            valf = z[i] / (D[i] - evj);
                            nrm += valf * valf;
                            temps[i + (p2 + j) * n] = valf;
                        }
                    }
                    inrms[tid * tn + tidb] = nrm;
                    __syncthreads();

                    // reduction (for the norm)
                    for(int r = tn / 2; r > 0; r /= 2)
                    {
                        if(go && tidb < r)
                        {
                            nrm += inrms[tid * tn + tidb + r];
                            inrms[tid * tn + tidb] = nrm;
                        }
                        __syncthreads();
                    }
                    nrm=sqrt(nrm);
                    
                    // multiply by C (row by row) 
                    for(int i = in; i < in + sz; ++i)
                    {
                        // inner products
                        temp = 0;
                        if(go)
                        {
                            for(int kk = in + tidb; kk < in + sz; kk += tn)
                                temp += C[i + kk * ldc] * temps[kk + (p2 + j) * n]; 
                        }
                        inrms[tid * tn + tidb] = temp;
                        __syncthreads();
                        
                        // reduction
                        for(int r = tn / 2; r > 0; r /= 2)
                        {
                            if(go && tidb < r)
                            {
                                temp += inrms[tid * tn + tidb + r];
                                inrms[tid * tn + tidb] = temp;
                            }
                            __syncthreads();
                        }

                        // result
                        if(go && tidb == 0)
                            vecs[i + (p2 + j) * n] = temp / nrm;
                        __syncthreads();
                    }
                }
                __syncthreads();
                /* ----------------------------------------------------------------- */


                // 3g. update D and C with computed values
                /* ----------------------------------------------------------------- */
                for(int j = 0; j < ns[tid]; ++j)
                {
                    if(idd[p2 + j] == 1)
                    {
                        D[p2 + j] = evs[p2 + j];
                        for(int i = in + tidb; i < in + sz; i += tn)
                            C[i + (p2 + j) * ldc] = vecs[i + (p2 + j) * n];
                    }
                }
                /* ----------------------------------------------------------------- */

            } // end of main loop in merge phase
            /* ----------------------------------------------------------------- */

        }
        __syncthreads();
    } // end of for-loop for the split blocks
}

/** STEDC_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void stedc_sort(const rocblas_int n,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 U CC,
                                 const rocblas_int shiftC,
                                 const rocblas_int ldc,
                                 const rocblas_stride strideC)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T* C;
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* D = DD + (bid * strideD);

    rocblas_int l, m;
    S p;

    // Sort eigenvalues and eigenvectors by selection sort
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

/** This local gemm adapts rocblas_gemm to multiply complex*real, and
    overwrite result: A = A*B **/
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // temp = A*B
    rocblasCall_gemm<BATCHED, STRIDED, T>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, A, shiftA, lda,
        strideA, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // A = temp
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                            stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, temp);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A -> work; work*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // work = real(A)
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // real(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    // work = imag(A)
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm<BATCHED, STRIDED, S>(
        handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work, shiftT, ldt,
        strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt, strideT, batch_count, workArr);

    // imag(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack,
                                   size_t* size_tempvect,
                                   size_t* size_tempgemm,
                                   size_t* size_tmpz,
                                   size_t* size_splits,
                                   size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
        return;
    }

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
    }

    // if size is too small, use steqr
    else if(n <= STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        size_t s1, s2;

        // requirements for steqr of small independent blocks
        // (TODO: Size should be STEDC_MIN_DC_SIZE when DC method is implemented)
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &s1);

        // extra requirements for original eigenvectors of small independent blocks
        //        if(evect != rocblas_evect_tridiagonal)
        //        {
        *size_tempvect = (n * n) * batch_count * sizeof(S);
        *size_tempgemm = 2 * (n * n) * batch_count * sizeof(S);
        if(COMPLEX)
            s2 = n * n * batch_count * sizeof(S);
        else
            s2 = 0;
        if(BATCHED && !COMPLEX)
            *size_workArr = sizeof(S*) * batch_count;
        else
            *size_workArr = 0;
        //        }
        //        else
        //        {
        //            *size_tempvect = 0;
        //            *size_tempgemm = 0;
        //            *size_workArr = 0;
        //            s2 = 0;
        //        }
        *size_work_stack = max(s1, s2);

        // size for split blocks and sub-blocks positions
        *size_splits = sizeof(rocblas_int) * (2 * n + 2) * batch_count;

        // size for temporary diagonal and rank-1 modif vector
        *size_tmpz = sizeof(S) * (2 * n) * batch_count;
    }
}

template <typename T, typename S>
rocblas_status rocsolver_stedc_argCheck(rocblas_handle handle,
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

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedc_template(rocblas_handle handle,
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
                                        void* work_stack,
                                        S* tempvect,
                                        S* tempgemm,
                                        S* tmpz,
                                        rocblas_int* splits,
                                        S** workArr)
{
    ROCSOLVER_ENTER("stedc", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

print_device_matrix(std::cout, "D", 1, n, D, 1);
print_device_matrix(std::cout, "E", 1, n, E, 1);

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

    // if no eigenvectors required, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_template<S>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                    batch_count, (rocblas_int*)work_stack);
    }

    // if size is too small, use steqr
    else if(n <= STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);
        rocblas_int blocksn = (n - 1) / BS2 + 1;

        // find independent split blocks in matrix
        ROCSOLVER_LAUNCH_KERNEL(stedc_split, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
                                strideD, E + shiftE, strideE, splits, eps);

        // initialize identity matrix in C if required
        if(evect == rocblas_evect_tridiagonal)
            ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocksn, blocksn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

        // initialize identity matrix in tempvect
        rocblas_int ldt = n;
        rocblas_stride strideT = n * n;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, tempvect, 0, ldt, strideT);

        // find max number of sub-blocks to consider during the divide phase
        rocblas_int maxblks = 1 << stedc_num_levs(n);
        size_t lmemsize = sizeof(rocblas_int) * 2 * maxblks + sizeof(S) * BDIM;

        // execute divide and conquer kernel with tempvect
        ROCSOLVER_LAUNCH_KERNEL((stedc_kernel<S>), dim3(STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(BDIM), lmemsize, stream, n, D + shiftD, strideD, E + shiftE,
                                strideE, tempvect, 0, ldt, strideT, info, (S*)work_stack, tmpz, tempgemm, splits,
                                eps, ssfmin, ssfmax, maxblks);

        // update eigenvectors C <- C*tempvect
        local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                        (S*)work_stack, 0, ldt, strideT, batch_count, workArr);

//print_device_matrix(std::cout, "Dfin", 1, n, D, 1);
//print_device_matrix(std::cout, "idd", 1, n, splits + n + 2, 1);

        // finally sort eigenvalues and eigenvectors
        ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, C, shiftC, ldc, strideC);
    }

    return rocblas_status_success;
}
