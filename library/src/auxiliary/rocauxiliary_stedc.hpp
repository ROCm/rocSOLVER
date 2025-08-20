/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

ROCSOLVER_BEGIN_NAMESPACE

#define STEDC_BDIM 512 // Number of threads per thread-block used in main stedc kernels

// bit indicating base deflation candidate
#define L_F_BCAND_BIT 0
// bit indicating top deflation candidate
#define L_F_TCAND_BIT 1 

// TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
// STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
// external gemm-based updates.
#define STEDC_EXTERNAL_GEMM true


__host__ __device__ inline rocblas_int get_splits_size(const rocblas_int n)
{
    // splits_map layout:
    // n - number of eigenvalues (matrix size)
    // m - number of merges on a current level
    // struct {
    // 0     rocblas_int sort_tmp[n];         // temp buffer used for mapping in stedc_sort()
    // 1     rocblas_int ns[n];               // the sub-blocks sizes
    // 2     rocblas_int ps[n];               // the sub-blocks initial positions
    // 3     rocblas_int idd[n];              // if idd[i] = 0, the value in position i has been deflated  (aka mask)
    // 4     rocblas_int pers[n];             // container of permutations when solving the secular eqns
    // 5     rocblas_int ;      //
    // 6     rocblas_int ;      //
    // 7     rocblas_int dds[n];              // degrees of secular equation
    // 8     rocblas_int map[n];              // 
    // 9     rocblas_int dcount[n];           // number of deflations
    // 10    rocblas_int esz[n];              // size of a corresponding merge (per each eigenvalue)
    // 11    rocblas_int eps[n];              // starting position of a corresponding merge (per each eigenvalue)
    // 12    rocblas_int cand[n];             // deflation candidate flags
    // 13    rocblas_int sidd[n];             // sorted idd
    // 14    rocblas_int em[n];               // id of a corresponding merge (per each eigenvalue)
    // 15    rocblas_int msz[m], _[n-m];      // size of each merge
    // 16    rocblas_int mps[m], _[n-m];      // starting position of each merge
    // 17    rocblas_int bsz[2*m], _[n-2*m];  // size of each block
    // 18    rocblas_int bps[2*m], _[n-2*m];  // starting position of each block
    // 19    rocblas_int dbg4[n];             //
    // };
    return 20 * n;
}

template <typename S> __host__ __device__ inline S* ptr_ns    (rocblas_int n, S* splits) { return splits +  1 * n; }
template <typename S> __host__ __device__ inline S* ptr_ps    (rocblas_int n, S* splits) { return splits +  2 * n; }
template <typename S> __host__ __device__ inline S* ptr_idd   (rocblas_int n, S* splits) { return splits +  3 * n; }
template <typename S> __host__ __device__ inline S* ptr_pers  (rocblas_int n, S* splits) { return splits +  4 * n; }
template <typename S> __host__ __device__ inline S* ptr_      (rocblas_int n, S* splits) { return splits +  5 * n; }
template <typename S> __host__ __device__ inline S* ptr__     (rocblas_int n, S* splits) { return splits +  6 * n; }
template <typename S> __host__ __device__ inline S* ptr_dds   (rocblas_int n, S* splits) { return splits +  7 * n; }
template <typename S> __host__ __device__ inline S* ptr_map   (rocblas_int n, S* splits) { return splits +  8 * n; }
template <typename S> __host__ __device__ inline S* ptr_dcount(rocblas_int n, S* splits) { return splits +  9 * n; }
template <typename S> __host__ __device__ inline S* ptr_esz   (rocblas_int n, S* splits) { return splits + 10 * n; }
template <typename S> __host__ __device__ inline S* ptr_eps   (rocblas_int n, S* splits) { return splits + 11 * n; }
template <typename S> __host__ __device__ inline S* ptr_cand  (rocblas_int n, S* splits) { return splits + 12 * n; }
template <typename S> __host__ __device__ inline S* ptr_sidd  (rocblas_int n, S* splits) { return splits + 13 * n; }
template <typename S> __host__ __device__ inline S* ptr_em    (rocblas_int n, S* splits) { return splits + 14 * n; }
template <typename S> __host__ __device__ inline S* ptr_msz   (rocblas_int n, S* splits) { return splits + 15 * n; }
template <typename S> __host__ __device__ inline S* ptr_mps   (rocblas_int n, S* splits) { return splits + 16 * n; }
template <typename S> __host__ __device__ inline S* ptr_bsz   (rocblas_int n, S* splits) { return splits + 17 * n; }
template <typename S> __host__ __device__ inline S* ptr_bps   (rocblas_int n, S* splits) { return splits + 18 * n; }
template <typename S> __host__ __device__ inline S* ptr_dbg   (rocblas_int n, S* splits) { return splits + 19 * n; }

__host__ __device__ inline rocblas_int get_tmpz_size(const rocblas_int n)
{
    // tmpz layout:
    // struct {
    // 0    S z[n];       // the rank-1 modification vectors in the merges
    // 1    S evs[n];     // roots of secular equations
    // 2    S cc[n];      // c value of rotation of corresponding deflation
    // 3    S ss[n];      // s value of rotation of corresponding deflation
    // 4    S tolsD[n];   // tollerance for deflation of repeaded values in D
    // 5    S tolsZ[n];   // tollerance for deflation of zero values in z
    // 6    S md[n];      // sorted d values
    return 7 * n;
}

template <typename S> __host__ __device__ inline S* ptr_z    (rocblas_int n, S* tmpz) { return tmpz + 0 * n; }
template <typename S> __host__ __device__ inline S* ptr_evs  (rocblas_int n, S* tmpz) { return tmpz + 1 * n; }
template <typename S> __host__ __device__ inline S* ptr_cc   (rocblas_int n, S* tmpz) { return tmpz + 2 * n; }
template <typename S> __host__ __device__ inline S* ptr_ss   (rocblas_int n, S* tmpz) { return tmpz + 3 * n; }
template <typename S> __host__ __device__ inline S* ptr_tolsD(rocblas_int n, S* tmpz) { return tmpz + 4 * n; }
template <typename S> __host__ __device__ inline S* ptr_tolsZ(rocblas_int n, S* tmpz) { return tmpz + 5 * n; }
template <typename S> __host__ __device__ inline S* ptr_md   (rocblas_int n, S* tmpz) { return tmpz + 6 * n; }


/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
    divides the input matrix into a 'blks' sub-blocks.
        - This kernel is to be called with as many groups in x as needed to cover all
        the batch_count problems. Each thread will work with a matrix in the batch. 
        - Size of groups is set to STEDC_BDIM. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_divide_kernel(const rocblas_int levs,
                                                                        const rocblas_int blks,
                                                                        const rocblas_int n,
                                                                        S* DD,
                                                                        const rocblas_stride strideD,
                                                                        S* EE,
                                                                        const rocblas_stride strideE,
                                                                        const rocblas_int batch_count,
                                                                        rocblas_int* splitsA)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // for each matrix in the batch
    if(bid < batch_count)
    {
        // select batch instance to work with
        S* D = DD + bid * strideD;
        S* E = EE + bid * strideE;

        // temporary arrays in global memory
        rocblas_int* splits = splitsA + bid * get_splits_size(n);
        rocblas_int* ns = ptr_ns(n, splits);
        rocblas_int* ps = ptr_ps(n, splits);

        // find sizes of sub-blocks
        ns[0] = n;
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

        // find beginning of sub-blocks and update elements in D
        rocblas_int p2 = 0;
        ps[0] = p2;
        for(int i = 1; i < blks; ++i)
        {
            p2 += ns[i - 1];
            ps[i] = p2;

            // perform sub-block division
            S p = E[p2 - 1];
            D[p2] -= p;
            D[p2 - 1] -= p;
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_SOLVE_KERNEL implements the solver phase of the DC algorithm to
   compute the eigenvalues/eigenvectors of the 'blks' different sub-blocks of a matrix.
        - Call this kernel with batch_count groups in y, and 'blks' groups in x. 
          Groups are single-thread **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_solve_kernel(const rocblas_int levs,
                                                                       const rocblas_int blks,
                                                                       const rocblas_int n,
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
                                                                       rocblas_int* splitsA,
                                                                       const S eps,
                                                                       const S ssfmin,
                                                                       const S ssfmax)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread index
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tidb_inc = hipBlockDim_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* ns = ptr_ns(n, splits);
    rocblas_int* ps = ptr_ps(n, splits);
    // workspace for solvers
    S* W = WA + bid * 2 * n;

    // Solve the blks sub-blocks in parallel (using classic QR iteration).
    if(sid < blks)
    {
        rocblas_int sbs = ns[sid];  // size of sub-block
        rocblas_int p2 = ps[sid];   // start position of sub-block

        run_steqr(tidb, tidb_inc, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc, info, W + p2 * 2,
                  30 * sbs, eps, ssfmin, ssfmax, false);
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_INIT_SPLITS initialize merge block related parts of a splits struct.
        - This kernel is to be called with 1 group in x and batch_count groups in y. 
        - Size of groups is set to STEDC_BDIM. **/
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_init_splits(const rocblas_int levs,
                                                                      const rocblas_int n,
                                                                      rocblas_int* splitsA)
{
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* ns  = ptr_ns(n, splits);
    rocblas_int* ps  = ptr_ps(n, splits);
    rocblas_int* em  = ptr_em(n, splits);
    rocblas_int* msz = ptr_msz(n, splits);
    rocblas_int* mps = ptr_mps(n, splits);

    rocblas_int n_blocks = 1 << levs;

    for(int i = hipThreadIdx_x; i < n_blocks; i+=hipBlockDim_x)
    {
        rocblas_int sz = ns[i];
        rocblas_int p = ps[i];
        msz[i] = sz;
        mps[i] = p;
        for (int j = 0; j < sz; j++) {
            em[p + j] = i;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_UPDATE_SPLITS updates merge block related parts of a splits struct.
        - This kernel is to be called with 1 group in x and batch_count groups in y. 
        - Size of groups is set to STEDC_BDIM. **/
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_update_splits(const rocblas_int levs,
                                                                        const rocblas_int k,
                                                                        const rocblas_int n,
                                                                        rocblas_int* splitsA)
{
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* em = ptr_em(n, splits);
    rocblas_int* msz = ptr_msz(n, splits);
    rocblas_int* mps = ptr_mps(n, splits);
    rocblas_int* esz = ptr_esz(n, splits);
    rocblas_int* eps = ptr_eps(n, splits);
    rocblas_int* bsz = ptr_bsz(n, splits);
    rocblas_int* bps = ptr_bps(n, splits);
    rocblas_int* map = ptr_map(n, splits);
    rocblas_int* dcount = ptr_dcount(n, splits);
    

    rocblas_int n_merges = 1 << (levs - k - 1);

    // previous merges becomes blocks on a current level
    for (int i = hipThreadIdx_x; i < n_merges * 2; i += hipBlockDim_x)
    {
        bsz[i] = msz[i];
        bps[i] = mps[i];
    }
    __syncthreads();

    for(int i = hipThreadIdx_x; i < n_merges; i += hipBlockDim_x)
    {
        rocblas_int sz1 = bsz[i * 2 + 0];
        rocblas_int sz2 = bsz[i * 2 + 1];
        rocblas_int p   = bps[i * 2];
        msz[i] = sz1 + sz2;
        mps[i] = p;
    }
    __syncthreads();

    for (int i = hipThreadIdx_x; i < n; i += hipBlockDim_x)
    {
        int m = em[i] / 2;
        esz[i] = msz[m];
        eps[i] = mps[m];
        map[i] = 0;
        dcount[i] = 0;
    }
    __syncthreads();

    for(int i = hipThreadIdx_x; i < n; i += hipBlockDim_x)
    {
        em[i] /= 2;
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_DEFLATEZERO_KERNEL performs deflation of zero values
        - Call this kernel with batch_count groups in y, and as many groups as half of the 
          unmerged sub-blocks in current level in x. Each group works with a merge of a pair
          of sub-blocks. Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_DeflateZero_kernel(const rocblas_int levs,
                                          const rocblas_int blks,
                                          const rocblas_int k,
                                          const rocblas_int n,
                                          S* DD,
                                          const rocblas_stride strideD,
                                          S* EE,
                                          const rocblas_stride strideE,
                                          S* CC,
                                          const rocblas_int shiftC,
                                          const rocblas_int ldc,
                                          const rocblas_stride strideC,
                                          S* tmpzA,
                                          rocblas_int* splitsA,
                                          const S eps)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* ns   = ptr_ns(n, splits);
    rocblas_int* ps   = ptr_ps(n, splits);
    rocblas_int* idd = ptr_idd(n, splits);
    rocblas_int* mps = ptr_mps(n, splits);
    rocblas_int* msz = ptr_msz(n, splits);
    rocblas_int* bsz = ptr_bsz(n, splits);
    rocblas_int* bps = ptr_bps(n, splits);


    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z     = ptr_z(n, tmpz);
    S* tolsD = ptr_tolsD(n, tmpz);
    S* tolsZ = ptr_tolsD(n, tmpz);


    // Work with merges on level k. A thread-group works with two leaves in the merge tree.
    {
        // 1. find rank-1 modification components (z and p) for this merge
        // ----------------------------------------------------------------
        rocblas_int sz1 = bsz[sid * 2 + 0];
        rocblas_int sz2 = bsz[sid * 2 + 1];
        rocblas_int p1  = bps[sid * 2 + 0];
        rocblas_int p2  = bps[sid * 2 + 1];

        S p = 2 * E[p2 - 1];

        S maxd = 0;
        S maxz = 0;
        // copy z values from first sub-block
        // copy last line of the first sub-block
        for (int i = hipThreadIdx_x; i < sz1; i += hipBlockDim_x) {
            S val = C[p2 - 1 + (p1 + i) * ldc] / sqrt(2);
            z[p1 + i] = val;
            maxz = std::max(maxz, std::abs(val));
        }
        // copy first line of the second sub-block
        for (int i = hipThreadIdx_x; i < sz2; i += hipBlockDim_x) {
            S val = C[p2 - 0 + (p2 + i) * ldc] / sqrt(2);
            z[p2 + i] = val;
            maxz = std::max(maxz, std::abs(val));
        }

        // 2. calculate deflation tolerance
        // ----------------------------------------------------------------
        // compute maximum of diagonal and z in each merge block
        rocblas_int sz = sz1 + sz2;
        for(int i = hipThreadIdx_x; i < sz; i += hipBlockDim_x)
        {
            maxd = std::max(maxd, abs(D[p1 + i]));
        }

        // temporary arrays in shared memory
        // used to store temp values during reduction
        __shared__ S lmaxz[STEDC_BDIM];
        __shared__ S lmaxd[STEDC_BDIM];
        lmaxd[hipThreadIdx_x] = maxd;
        lmaxz[hipThreadIdx_x] = maxz;
        __syncthreads();

        rocblas_int dim2 = hipBlockDim_x / 2;
        while(dim2 > 0)
        {
            if(hipThreadIdx_x < dim2)
            {
                S vald = lmaxd[hipThreadIdx_x + dim2];
                S valz = lmaxz[hipThreadIdx_x + dim2];
                maxd = std::max(maxd, vald);
                maxz = std::max(maxz, valz);
                lmaxd[hipThreadIdx_x] = maxd;
                lmaxz[hipThreadIdx_x] = maxz;
            }
            dim2 /= 2;
            __syncthreads();
        }
        
        // tol should be  8 * eps * (max diagonal or z element participating in merge)
        maxd = lmaxd[0];
        maxz = lmaxz[0];
        
        S tolD = 8 * eps * std::max(maxd, maxz);
        S tolZ = 8 * eps * std::max(maxd, maxz);
        // store tolerances in global memory
        for(int i = hipThreadIdx_x; i < sz; i += hipBlockDim_x)
        {
            tolsD[p1 + i] = tolD;
            tolsZ[p1 + i] = tolD;
        }


        // 3. deflate eigenvalues
        // ----------------------------------------------------------------
        // deflate zero components
        for(int i = hipThreadIdx_x; i < sz; i += hipBlockDim_x)
        {
            S g = z[p1 + i];
            // deflated ev if component in z is zero
            idd[p1 + i] = (abs(p * g) <= tolZ) ? 0 : 1;
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_SORTD_KERNEL sorts D array and construct map of original positions
        - Call this kernel with n groups in x and batch_count groups in y. Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_SortD_kernel(const rocblas_int levs,
                                    const rocblas_int blks,
                                    const rocblas_int k,
                                    const rocblas_int n,
                                    S* DD,
                                    const rocblas_stride strideD,
                                    S* tmpzA,
                                    rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;

    // select batch instance to work with
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* idd    = ptr_idd(n, splits);
    rocblas_int* map    = ptr_map(n, splits);
    rocblas_int* esz    = ptr_esz (n, splits);
    rocblas_int* eps    = ptr_eps (n, splits); 
    rocblas_int* cand   = ptr_cand(n, splits); 
    rocblas_int* sidd   = ptr_sidd(n, splits);
    
    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* md    = ptr_md(n, tmpz);
    S* tolsD = ptr_tolsD(n, tmpz);


    S d = D[sid];
    rocblas_int sz = esz[sid];
    rocblas_int p  = eps[sid];
    rocblas_int dz = idd[sid];

    constexpr int regs = 8;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks = (sz - 1) / chunk_width + 1;
    S bval[regs];
    int dzs[regs];


    int nan = 0;
    int lt = 0;
    int eq = 0;

    for(int chunk = 0; chunk < n_chunks; chunk++)
    {
        for(int i = 0; i < regs; i++)
        {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < sz)
            {
                bval[i] = D[p + x];
                dzs[i]  = idd[p + x];
            }
        }
        for(int i = 0; i < regs; i++)
        {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < sz)
            {
                nan += (bval[i] != bval[i]);
                // lt - how many values are less then the current
                // eq - how many values are equal to the current
                // all zero deflated values have to be grouped at the end
                // so we order any deflated value > any non-deflated value
                // dz == 0 - current is deflated, dz[i] == 1 - other value is not deflated
                lt += (dz < dzs[i]) || (dz == dzs[i] && bval[i] < d);
                eq += (dz == dzs[i]) && (bval[i] == d && (p + x) < sid);
            }
        }
    }

    int pos = lt + eq;
    __shared__ int lds[STEDC_BDIM];
    // Reduction (sum of pos across all lanes in a workgroup)
    // on each iteration reduction is done within a subgroup of size of 2*bit
    // by xoring corresponding bit of an address.
    // The faster implementation should use dpp + a single trip through lds
    // but keeping code simple for now.
    int bit = 1;
    while (bit < STEDC_BDIM) {
        lds[hipThreadIdx_x ^ bit] = pos;
        __syncthreads();
        pos += lds[hipThreadIdx_x];
        __syncthreads();
        bit *= 2;
    }

    if (hipThreadIdx_x == 0) {
        map [pos+p] = sid;
        md  [pos+p] = d;
        sidd[pos+p] = dz;
    }

    __syncthreads();
    // The NAN fp value is unordered, so it is possible that with computed
    // new positions it would be silently overwriten with non NAN value.
    // Make sure we propagate NAN. It is likely to have more NANs in the output
    // than in the input, but the following computations are doomed anyway.
    if (nan) {
        md[sid] = NAN;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_SETCANDFLAGS_KERNEL fills cand[] array with deflation candidate flags
        - Call this kernel with ((n - 1)/STEDC_BDIM+1) groups in x and batch_count groups in y.
        Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_SetCandFlags_kernel(const rocblas_int levs,
                                           const rocblas_int blks,
                                           const rocblas_int k,
                                           const rocblas_int n,
                                           S* DD,
                                           const rocblas_stride strideD,
                                           S* tmpzA,
                                           rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* eps    = ptr_eps (n, splits); 
    rocblas_int* cand   = ptr_cand(n, splits); 
    rocblas_int* sidd   = ptr_sidd(n, splits);

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* md    = ptr_md(n, tmpz);
    S* tolsD = ptr_tolsD(n, tmpz);

    

    constexpr int F_BCAND = 1 << L_F_BCAND_BIT;
    constexpr int F_TCAND = 1 << L_F_TCAND_BIT;

    // find deflate candidates
    int i = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    if (i < n)
    {
        int next = (i + 1) < n ? (i + 1) : i;
        int prev = (i > 0) ? (i - 1) : 0;
        S tol = tolsD[i];
        S d   = md[i];
        S dn  = md[next];
        S dp  = md[prev];
        rocblas_int p  = eps[i];
        rocblas_int pn = eps[next];
        rocblas_int pp = eps[prev];

        int bcandidate =
            sidd[i] && sidd[next]       // not yet deflated
            && p == pn                  // in the same merge block
            && std::abs(d - dn) <= tol  // within tolerance
            && i != (n-1)               // isn't last 
            ;
        int tcandidate =
            sidd[i] && sidd[prev]       // not yet deflated
            && p == pp                  // in the same merge block
            && std::abs(d - dp) <= tol  // within tolerance
            && i > 0                    // isn't first
            ;
        cand[i] = (bcandidate << L_F_BCAND_BIT) + (tcandidate << L_F_TCAND_BIT);
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_DEFLATECOUNT_KERNEL fills dcount[] array with number of deflations for each base point
        - Call this kernel with ((n - 1)/STEDC_BDIM+1) groups in x and batch_count groups in y.
        Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_DeflateCount_kernel(const rocblas_int levs,
                                           const rocblas_int blks,
                                           const rocblas_int k,
                                           const rocblas_int n,
                                           S* DD,
                                           const rocblas_stride strideD,
                                           S* tmpzA,
                                           rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* dcount = ptr_dcount(n, splits);
    rocblas_int* cand   = ptr_cand(n, splits);

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* md    = ptr_md(n, tmpz);
    S* tolsD = ptr_tolsD(n, tmpz);


    constexpr rocblas_int max_len = 4096;
    __shared__ S   ldsD[max_len];
    __shared__ int lcand[max_len];
    constexpr int F_BCAND = 1 << L_F_BCAND_BIT;
    constexpr int F_TCAND = 1 << L_F_TCAND_BIT;

    int start = hipBlockDim_x * hipBlockIdx_x;
    int base  = start + hipThreadIdx_x;
    int prev  = (base > 0) ? (base - 1) : 0;
    int candp = (prev < n) ? cand[prev] : 0;
    int candb = (base < n) ? cand[base] : 0;
    S bval = (base < n) ? md[base] : 0;
    S tol  = (base < n) ? tolsD[base] : 0;

    // cache max_len values of D[] and cand[]
    for(int i = hipThreadIdx_x; i < max_len; i += hipBlockDim_x)
    {
        int x = start + i;
        ldsD[i]  = (x < n) ? md[x]   : 0;
        lcand[i] = (x < n) ? cand[x] : 0;
    }
    __syncthreads();
    
    if ((candb & F_BCAND) && (base == 0 || !(candp & F_BCAND))) {
        int top = base + 2;
        int candt = lcand[top - start];
        while (top < n && (candt & F_TCAND))
        {
            // first max_len values are prefetched into lds,
            // access global memory only if need to go beyond that
            // which is very unlikely
            S tval = (top - start) < max_len
                   ? ldsD[top-start]
                   : md[top];

            if ((tval - bval) > tol)
            {
                dcount[base] = top - base - 1;
                base = top;
                bval = tval;
            }
            top++;

            // first max_len values are prefetched into lds,
            // access global memory only if need to go beyond that
            // which is very unlikely
            candt = (top - start) < max_len
                ? lcand[top-start]
                : cand[top];
        }
        dcount[base] = top - base - 1;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_DEFLATEAPPLY_KERNEL applies deflations and saves c/s values used to rotate C vectors
        - Call this kernel with ((n - 1)/STEDC_BDIM+1) groups in x and batch_count groups in y.
        Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_DeflateApply_kernel(const rocblas_int levs,
                                           const rocblas_int blks,
                                           const rocblas_int k,
                                           const rocblas_int n,
                                           S* DD,
                                           const rocblas_stride strideD,
                                           S* tmpzA,
                                           rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* idd    = ptr_idd(n, splits);
    rocblas_int* map    = ptr_map(n, splits);
    rocblas_int* dcount = ptr_dcount(n, splits);

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z  = ptr_z(n, tmpz);
    S* cc = ptr_cc(n, tmpz);
    S* ss = ptr_ss(n, tmpz);

    constexpr rocblas_int max_len = 4096;
    __shared__ S   lz[max_len];
    __shared__ int lmap[max_len];

    int start   = hipBlockDim_x * hipBlockIdx_x;
    int base    = start + hipThreadIdx_x;
    int cnt     = (base < n) ? dcount[base] : 0;

    // cache max_len values of map[] and appropriate z[]
    for (int i = hipThreadIdx_x; i < max_len; i += hipBlockDim_x) {
        int x   = start + i;
        lmap[i] = (x < n) ? map[x]    : 0;
        lz[i]   = (x < n) ? z[map[x]] : 0;
    }
    __syncthreads();

    if (cnt) {
        S baseval = lz[hipThreadIdx_x];
        for (int j = 0; j < cnt; j++) {
            int top = base + j + 1;

            // first max_len values are prefetched into lds,
            // access global memory only if need to go beyond that
            // which is very unlikely
            int idx = (top - start) < max_len
                    ? lmap[top - start]
                    : map[top];
            S g     = (top - start) < max_len
                    ? lz[top - start]
                    : z[idx];

            S f = baseval;
            S c, s, rr;
            lartg(f, g, c, s, rr);
            baseval = rr;

            idd[idx] = 0;
            z[idx]   = 0;
            cc[idx]  = c;
            ss[idx]  = s;
        }
        z[lmap[hipThreadIdx_x]] = baseval;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEROTATE_KERNEL performs rotation of vectors corresponding to deflations
        - Call this kernel with batch_count groups in y, and n (matrix size) groups in x.
        - Each group will deal with one deflation group, groups that don't correspond to
          a deflation group will do nothing **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeRotate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* tmpzA,
                             rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;

    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* map     = ptr_map(n, splits);         
    rocblas_int* dcounts = ptr_dcount(n, splits); 

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* cc = ptr_cc(n, tmpz);
    S* ss = ptr_ss(n, tmpz);

    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks    = (n - 1) / chunk_width + 1;
    S bval[regs];
    S tval[regs];


    rocblas_int dgs = hipBlockIdx_x;
    rocblas_int dcnt = dcounts[dgs];
    if (dcnt)
    {
        rocblas_int base = map[dgs];
        S* Cbase = C + base * ldc;

        for (int chunk = 0; chunk < n_chunks; chunk++) {

            for(int i = 0; i < regs; i++) {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if (x < n) {
                    bval[i] = Cbase[x];
                }
            }

            for (int dn = 0; dn < dcnt; dn++) {
                rocblas_int top = map[dgs + dn + 1];
                S c = cc[top];
                S s = ss[top];
                S* Ctop = C + top * ldc;

                for(int i = 0; i < regs; i++) {
                    int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                    if(x < n){
                        tval[i] = Ctop[x];
                    }
                }

                for (int i = 0; i < regs; i++) {
                    S valf = bval[i];
                    S valg = tval[i];
                    bval[i] = valf * c - valg * s;
                    tval[i] = valf * s + valg * c; 
                }

                for(int i = 0; i < regs; i++) {
                    int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                    if(x < n){
                        Ctop[x] = tval[i];
                    }
                }
                __syncthreads();
            }

            for(int i = 0; i < regs; i++) {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if (x < n) {
                    Cbase[x] = bval[i];
                }
            }
        }
    }
}

template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_Organize_kernel(const rocblas_int levs,
                                       const rocblas_int blks,
                                       const rocblas_int k,
                                       const rocblas_int n,
                                       S* DD,
                                       const rocblas_stride strideD,
                                       S* EE,
                                       const rocblas_stride strideE,
                                       S* CC,
                                       const rocblas_int shiftC,
                                       const rocblas_int ldc,
                                       const rocblas_stride strideC,
                                       S* tmpzA,
                                       S* vecsA,
                                       rocblas_int* splitsA,
                                       const S eps)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* idd  = ptr_idd(n, splits);
    rocblas_int* pers = ptr_pers(n, splits);
    rocblas_int* msz  = ptr_msz(n, splits);
    rocblas_int* dds  = ptr_dds(n, splits);
    rocblas_int* mps = ptr_mps(n, splits);
    rocblas_int* bsz = ptr_bsz(n, splits);
    rocblas_int* bps = ptr_bps(n, splits);

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z     = ptr_z(n, tmpz);
    S* tolsD = ptr_tolsD(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    rocblas_int bdm = 1 << (k + 1);

    // Work with merges on level k. A thread-group works with two leaves in the merge tree.
    {
        // 1. find rank-1 modification component (p) for this merge
        // ----------------------------------------------------------------
        // rank-1 modification component p correspond to the last element in the first sub-block
        rocblas_int p2 = bps[sid * 2 + 1];
        S p = 2 * E[p2 - 1];

        // 3. deflate eigenvalues
        // ----------------------------------------------------------------
        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int sz = msz[sid];
        rocblas_int in = mps[sid];

        // 4. Organize data with non-deflated values to prepare secular equation
        // ------------------------------------------------------------------------ 
        // define shifted arrays
        S* tmpd = temps + in * n;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree and components of secular equation
        // tmpd contains the non-deflated diagonal elements (ie. poles of the
        // secular eqn) zz contains the corresponding non-zero elements of the
        // rank-1 modif vector
        rocblas_int dd = 0;
        for(int i = 0; i < sz; ++i)
        {
            if(mask[i] == 1)
            {
                if(tidb == 0)
                {
                    per[dd] = i;
                    tmpd[dd] = p < 0 ? -diag[i] : diag[i];
                    if(dd != i)
                        zz[dd] = zz[i];
                }
                dd++;
            }
        }
        dds[in] = dd;
    }
}


template <typename S>
void __device__ inline solve_seq_eqns(const rocblas_int dd,
                                      const rocblas_int iam,
                                      const rocblas_int bdm,
                                      const rocblas_int sz,
                                      const rocblas_int n,
                                      const S p,
                                      const S eps,
                                      const S ssfmin,
                                      const S ssfmax,
                                      const rocblas_int* mask,
                                            S* tmpd,
                                            S* ev,
                                            S* zz)
{
    /* ----------------------------------------------------------------- */

    // 2. Solve secular eqns, i.e. find the dd zeros
    // corresponding to non-deflated new eigenvalues of the merged block
    /* ----------------------------------------------------------------- */
    // each thread will find a different zero in parallel
    S a, b;
    for(int j = iam; j < sz; j += bdm)
    {
        if(mask[j] == 1)
        {
            // find position in the ordered array
            S valf = p < 0 ? -ev[j] : ev[j];
            int count = dd, cc = 0;
            while(count > 0)
            {
                auto step = count / 2;
                auto it = cc + step;
                if(tmpd[it + j * n] < valf)
                {
                    cc = ++it;
                    count -= step + 1;
                }
                else
                    count = step;
            }

            // computed zero will overwrite 'ev' at the corresponding position.
            // 'tmpd' will be updated with the distances D - lambda_i.
            // deflated values are not changed.
            rocblas_int linfo;

#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
            linfo = slaed4(dd, cc, tmpd + j * n, zz, std::abs(p), ev[j]);
#else
            if(cc == dd - 1)
                linfo = seq_solve_ext(dd, tmpd + j * n, zz, (p < 0 ? -p : p), ev + j, eps, ssfmin,
                                      ssfmax);
            else
                linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps, ssfmin,
                                  ssfmax);
#endif

            if(p < 0)
                ev[j] *= -1;
        }
    }
}


template <typename S>
void __device__ inline rescale_z(const rocblas_int dd,
                                 const rocblas_int iam,
                                 const rocblas_int bdm,
                                 const rocblas_int sz,
                                 const rocblas_int n,
                                 const rocblas_int* per,
                                 const rocblas_int* mask,
                                 const S* tmpd,
                                 const S* diag,
                                       S* zz)
{
    // Re-scale vector Z to avoid bad numerics when an eigenvalue
    // is too close to a pole
    for(int i = iam; i < dd; i += bdm)
    {
        S valf = 1;
        for(int j = 0; j < sz; ++j)
        {
            if(mask[j] == 1)
            {
                S valg = tmpd[i + j * n];
                valf *= (per[i] == j) ? valg : valg / (diag[per[i]] - diag[j]);
            }
        }
        valf = sqrt(std::abs(valf));
        zz[i] = zz[i] < 0 ? -valf : valf;
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVALUES_KERNEL solves the secular equation for every pair of sub-blocks 
    that need to be merged. 
        - Call this kernel with batch_count groups in y, and as many groups as half of the 
          unmerged sub-blocks in current level in x. Each group works with a merge of a pair
          of sub-blocks. Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeValues_Sort_kernel(const rocblas_int levs,
                                  const rocblas_int blks,
                                  const rocblas_int k,
                                  const rocblas_int n,
                                  S* DD,
                                  const rocblas_stride strideD,
                                  S* EE,
                                  const rocblas_stride strideE,
                                  S* tmpzA,
                                  S* vecsA,
                                  rocblas_int* splitsA,
                                  const S eps,
                                  const S ssfmin,
                                  const S ssfmax)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* mps  = ptr_mps(n, splits);
    rocblas_int* idd  = ptr_idd(n, splits);
    rocblas_int* pers = ptr_pers(n, splits);
    rocblas_int* msz  = ptr_msz(n, splits);
    rocblas_int* dds  = ptr_dds(n, splits);
    
    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z   = ptr_z(n, tmpz);
    S* evs = ptr_evs(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    rocblas_int bdm = 1 << (k + 1);

    // Work with merges on level k. A thread-group works with two leaves in the merge tree;
    // all threads work together to solve the secular equation.
    {
        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int sz = msz[sid];
        rocblas_int in = mps[sid];


        // 1. Organize data with non-deflated values to prepare secular equation
        // ----------------------------------------------------------------- 
        // All threads of the group participating in the merge will work together
        // to solve the correspondinbg secular eqn. Now 'iam' indexes those threads
        rocblas_int iam = tidb;
        bdm = hipBlockDim_x;

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* ev = evs + in;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = dds[in];

        // Order the elements in tmpd and zz using a simple parallel selection/bubble sort.
        // This will allow us to find initial intervals for eigenvalue guesses
        for(int i = 0; i < dd; i++)
        {
            for(int j = 2 * iam + i % 2; j < dd - 1; j += 2 * bdm)
            {
                if(tmpd[j] > tmpd[j + 1])
                {
                    swap(tmpd[j], tmpd[j + 1]);
                    swap(zz[j], zz[j + 1]);
                    swap(per[j], per[j + 1]);
                }
            }
            __syncthreads();
        }
        
        // make dd copies of the non-deflated ordered diagonal elements
        // (i.e. the poles of the secular eqn) so that the distances to the
        // eigenvalues (D - lambda_i) are updated while computing each eigenvalue.
        // This will prevent collapses and division by zero when an eigenvalue
        // is too close to a pole.
        for(int i = iam; i < dd; i += bdm)
        {
            for(int j = i + n; j < i + sz * n; j += n)
                tmpd[j] = tmpd[i];
        }

        // finally copy over all diagonal elements in ev. ev will be overwritten
        // by the new computed eigenvalues of the merged block
        for(int i = iam; i < sz; i += bdm)
            ev[i] = diag[i];
    }
}

template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeValues_Solve_kernel(const rocblas_int levs,
                                   const rocblas_int blks,
                                   const rocblas_int k,
                                   const rocblas_int n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   S* tmpzA,
                                   S* vecsA,
                                   rocblas_int* splitsA,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax,
                                   const rocblas_int groups_per_merge)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x / groups_per_merge;
    // thread id
    rocblas_int group_in_subblock = hipBlockIdx_x % groups_per_merge;
    rocblas_int tidb = hipThreadIdx_x + group_in_subblock * hipBlockDim_x;
    rocblas_int tid;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* idd  = ptr_idd(n, splits);
    rocblas_int* pers = ptr_pers(n, splits);
    rocblas_int* msz  = ptr_msz(n, splits);
    rocblas_int* dds  = ptr_dds(n, splits);
    rocblas_int* mps  = ptr_mps(n, splits);
    rocblas_int* bsz  = ptr_bsz(n, splits);
    rocblas_int* bps = ptr_bps(n, splits);
    
    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z   = ptr_z(n, tmpz);
    S* evs = ptr_evs(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // Work with merges on level k. A thread-group works with two leaves in the merge tree;
    // all threads work together to solve the secular equation.
    {
        // Find off-diagonal element of the merge
        // rank-1 modification component p correspond to the last element in the first sub-block
        rocblas_int p2 = bps[sid * 2 + 1];
        S p = 2 * E[p2 - 1];

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int sz = msz[sid];
        rocblas_int in = mps[sid];

        // 1. Organize data with non-deflated values to prepare secular equation
        // -----------------------------------------------------------------
        // All threads of the group participating in the merge will work together
        // to solve the correspondinbg secular eqn. Now 'iam' indexes those threads
        rocblas_int iam = tidb;
        rocblas_int bdm = hipBlockDim_x * groups_per_merge;

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* ev = evs + in;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = dds[in];

        solve_seq_eqns(dd, iam, bdm, sz, n, p, eps, ssfmin, ssfmax, mask, tmpd, ev, zz);
    }
}

template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeValues_Rescale_kernel(const rocblas_int levs,
                                     const rocblas_int blks,
                                     const rocblas_int k,
                                     const rocblas_int n,
                                     S* DD,
                                     const rocblas_stride strideD,
                                     S* EE,
                                     const rocblas_stride strideE,
                                     S* tmpzA,
                                     S* vecsA,
                                     rocblas_int* splitsA,
                                     const S eps,
                                     const S ssfmin,
                                     const S ssfmax)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* mps  = ptr_mps(n, splits);
    rocblas_int* idd  = ptr_idd(n, splits);
    rocblas_int* pers = ptr_pers(n, splits);
    rocblas_int* msz  = ptr_msz(n, splits);
    rocblas_int* dds  = ptr_dds(n, splits);
    
    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z   = ptr_z(n, tmpz);
    S* evs = ptr_evs(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // Work with merges on level k. A thread-group works with two leaves in the merge tree;
    // all threads work together to solve the secular equation.
    {
        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int sz = msz[sid];
        rocblas_int in = mps[sid];


        // 1. Organize data with non-deflated values to prepare secular equation
        // ----------------------------------------------------------------- 
        // All threads of the group participating in the merge will work together
        // to solve the correspondinbg secular eqn. Now 'iam' indexes those threads
        rocblas_int iam = tidb;
        rocblas_int bdm = hipBlockDim_x;

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* ev = evs + in;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = dds[in];

        rescale_z(dd, iam, bdm, sz, n, per, mask, tmpd, diag, zz);
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL prepares vectors from the secular equation for
    every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and as many groups as columns would 
          be in the matrix if its size is exact multiple of the number of sub-blocks 'blks'.
          Each group works with a column. Groups are size STEDC_BDIM.
        - If a group has an id larger than the actual number of columns it will do nothing. **/
template <bool USEGEMM, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeVectors_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* EE,
                              const rocblas_stride strideE,
                              S* CC,
                              const rocblas_int shiftC,
                              const rocblas_int ldc,
                              const rocblas_stride strideC,
                              S* tmpzA,
                              S* vecsA,
                              rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* ns   = ptr_ns(n, splits);
    rocblas_int* ps   = ptr_ps(n, splits);
    rocblas_int* idd  = ptr_idd(n, splits);
    rocblas_int* pers = ptr_pers(n, splits);

    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* z   = ptr_z(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    extern __shared__ rocblas_int lsmem[];
    S* inrms = reinterpret_cast<S*>(lsmem);

    // tn is max number of vectors in each sub-block
    rocblas_int bdm = 1 << (k + 1);
    rocblas_int tn = (n - 1) / blks + 1;

    // Work with merges on level k. Each thread-group works with one vector.
    if(sid < tn * blks)
    {
        rocblas_int iam, sz, p2;
        S valf, valg;

        // tid indexes the sub-blocks in the entire split block
        tid = sid / tn;
        p2 = ps[tid];
        // vidb indexes the vectors associated with each sub-block
        vidb = sid % tn;
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        iam = tid % bdm;

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position
        rocblas_int in = ps[tid - iam];
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        sz = ns[tid];
        for(int i = iam; i > 0; --i)
            sz += ns[tid - i];
        for(int i = bdm - 1 - iam; i > 0; --i)
            sz += ns[tid + i];

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = 0;
        for(int i = 0; i < sz; ++i)
        {
            if(mask[i] == 1)
                dd++;
        }
        __syncthreads();


        // Prepare vectors corresponding to non-deflated values
        S temp, nrm;
        rocblas_int j = vidb;
        bool go = (j < ns[tid]);
        S* putvec = USEGEMM ? vecs : temps;

        if(go)
        {
            if(idd[p2 + j] == 1)
            {
                // compute vectors of rank-1 perturbed system and their norms
                nrm = 0;
                for(int i = tidb; i < dd; i += dim)
                {
                    valf = zz[i] / temps[i + (p2 + j) * n];
                    nrm += valf * valf;
                    putvec[i + (p2 + j) * n] = valf;
                }
                inrms[tidb] = nrm;
                __syncthreads();

                // reduction (for the norms)
                for(int r = dim / 2; r > 0; r /= 2)
                {
                    if(tidb < r)
                    {
                        nrm += inrms[tidb + r];
                        inrms[tidb] = nrm;
                    }
                    __syncthreads();
                }
                nrm = sqrt(inrms[0]);
            }

            if(USEGEMM)
            {
                // when using external gemms for the update, we need to
                // put vectors in padded matrix 'temps'
                // (this is to compute 'vecs = C * temps' using external gemm call)
                for(int i = tidb; i < in + sz; i += dim)
                {
                    if(i >= in && idd[p2 + j] == 1 && idd[i] == 1)
                    {
                        dd = 0;
                        for(int k = in; k < i; ++k)
                        {
                            if(idd[k] == 0)
                                dd++;
                        }
                        temps[pers[i - dd] + in + (p2 + j) * n]
                            = vecs[i - dd - in + (p2 + j) * n] / nrm;
                    }
                    else
                        temps[i + (p2 + j) * n] = 0;
                }
            }
            else
            {
                // otherwise, use internal gemm-like procedure to
                // multiply by C (row by row)
                rocblas_int tsz = 1 << (levs - 1 - k);
                tsz = (n - 1) / tsz + 1;
                if(idd[p2 + j] == 1)
                {
                    for(int ii = 0; ii < tsz; ++ii)
                    {
                        rocblas_int i = in + ii;

                        // inner products
                        temp = 0;
                        if(ii < sz)
                        {
                            for(int kk = tidb; kk < dd; kk += dim)
                                temp += C[i + (per[kk] + in) * ldc] * temps[kk + (p2 + j) * n];
                        }
                        inrms[tidb] = temp;
                        __syncthreads();

                        // reduction
                        for(int r = dim / 2; r > 0; r /= 2)
                        {
                            if(ii < sz && tidb < r)
                            {
                                temp += inrms[tidb + r];
                                inrms[tidb] = temp;
                            }
                            __syncthreads();
                        }

                        // result
                        if(ii < sz && tidb == 0)
                            vecs[i + (p2 + j) * n] = temp / nrm;
                        __syncthreads();
                    }
                }
            }
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEUPDATE_KERNEL updates vectors and values after a merge is done. 
        - Call this kernel with batch_count groups in y, and as many groups as columns would 
          be in the matrix if its size is exact multiple of the number of sub-blocks 'blks'.
          Each group works with a column. Groups are size STEDC_BDIM.
        - If a group has an id larger than the actual number of columns it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeUpdate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* DD,
                             const rocblas_stride strideD,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* tmpzA,
                             S* vecsA,
                             rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * get_splits_size(n);
    rocblas_int* ns   = ptr_ns(n, splits);
    rocblas_int* ps   = ptr_ps(n, splits);
    rocblas_int* idd  = ptr_idd(n, splits);
    
    S* tmpz = tmpzA + bid * get_tmpz_size(n);
    S* evs = ptr_evs(n, tmpz);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);

    // tn is max number of vectors in each sub-block
    rocblas_int bdm = 1 << (k + 1);
    rocblas_int tn = (n - 1) / blks + 1;

    // Work with merges on level k. Each thread-group works with one vector.
    if(sid < tn * blks)
    {
        rocblas_int iam, sz, p2;

        // tid indexes the sub-blocks in the entire split block
        tid = sid / tn;
        p2 = ps[tid];
        // vidb indexes the vectors associated with each sub-block
        vidb = sid % tn;
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        iam = tid % bdm;

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position
        rocblas_int in = ps[tid - iam];
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        sz = ns[tid];
        for(int i = iam; i > 0; --i)
            sz += ns[tid - i];
        for(int i = bdm - 1 - iam; i > 0; --i)
            sz += ns[tid + i];

        // update D and C with computed values and vectors
        rocblas_int j = vidb;
        bool go = (j < ns[tid] && idd[p2 + j] == 1);
        if(go)
        {
            if(tidb == 0)
                D[p2 + j] = evs[p2 + j];
            for(int i = in + tidb; i < in + sz; i += dim)
                C[i + (p2 + j) * ldc] = vecs[i + (p2 + j) * n];
        }
    }
}


template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_copyD(const rocblas_int n,
                                                                S* DDin,
                                                                const rocblas_stride strideDin,
                                                                S* DDout,
                                                                const rocblas_stride strideDout)
{
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;

    S* Din  = DDin  + bid * strideDin;
    S* Dout = DDout + bid * strideDout;

    int tid = hipThreadIdx_x;

    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks = (n - 1) / chunk_width + 1;
    S bval[regs];

    for(int chunk = 0; chunk < n_chunks; chunk++) {
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                bval[i] = Din[x];
        }
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                Dout[x] = bval[i];
        }
    }
}

template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_copyC(const rocblas_int n,
                                                                S* CCin,
                                                                const rocblas_int shiftCin,
                                                                const rocblas_int ldcin,
                                                                const rocblas_stride strideCin,
                                                                S* CCout,
                                                                const rocblas_int shiftCout,
                                                                const rocblas_int ldcout,
                                                                const rocblas_stride strideCout)
{
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // group id
    rocblas_int gid = hipBlockIdx_x;

    S* Cin  = load_ptr_batch<S>(CCin,  bid, shiftCin,  strideCin);
    S* Cout = load_ptr_batch<S>(CCout, bid, shiftCout, strideCout);

    S* src =  Cin + ldcin  * gid;
    S* dst = Cout + ldcout * gid;
    
    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks = (n - 1) / chunk_width + 1;
    S bval[regs];
    
    for(int chunk = 0; chunk < n_chunks; chunk++) {
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                bval[i] = src[x];
        }
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                dst[x] = bval[i];
        }
    }
}

/** STEDC_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_sort(const rocblas_int n,
                                                              S* DDin,
                                                              const rocblas_stride strideDin,
                                                              S* DDout,
                                                              const rocblas_stride strideDout,
                                                              S* CCin,
                                                              const rocblas_int shiftCin,
                                                              const rocblas_int ldcin,
                                                              const rocblas_stride strideCin,
                                                              S* CCout,
                                                              const rocblas_int shiftCout,
                                                              const rocblas_int ldcout,
                                                              const rocblas_stride strideCout
                                                              
)
{
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // group id
    rocblas_int gid = hipBlockIdx_x;


    S* Din  = DDin  + bid * strideDin;
    S* Dout = DDout + bid * strideDout;

    int tid = hipThreadIdx_x;

    S d = Din[gid];

    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks = (n - 1) / chunk_width + 1;
    S bval[regs];

    int nan = 0;
    int lt = 0;
    int eq = 0;

    for(int chunk = 0; chunk < n_chunks; chunk++)
    {
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
            {
                bval[i] = Din[x];
            }
        }
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
            {
                nan += (bval[i] != bval[i]);
                // lt - how many values are less then the current
                // eq - how many values are equal to the current
                // all zero deflated values have to be grouped at the end
                // so we order any deflated value > any non-deflated value
                // dz == 0 - current is deflated, dz[i] == 1 - other value is not deflated
                lt += (bval[i] < d);
                eq += (bval[i] == d && x < gid);
            }
        }
    }

    int pos = lt + eq;
    __shared__ int lds[STEDC_BDIM];
    // Reduction (sum of pos across all lanes in a workgroup)
    // on each iteration reduction is done within a subgroup of size of 2*bit
    // by xoring corresponding bit of an address.
    // The faster implementation should use dpp + a single trip through lds
    // but keeping code simple for now.
    int bit = 1;
    while(bit < STEDC_BDIM) {
        lds[hipThreadIdx_x ^ bit] = pos;
        __syncthreads();
        pos += lds[hipThreadIdx_x];
        __syncthreads();
        bit *= 2;
    }

    if (hipThreadIdx_x == 0) {
        Dout[pos] = d;
    }

    __syncthreads();
    // The NAN fp value is unordered, so it is possible that with computed
    // new positions it would be silently overwriten with non NAN value.
    // Make sure we propagate NAN. It is likely to have more NANs in the output
    // than in the input, but the following computations are doomed anyway.
    if (nan) {
        Dout[gid] = NAN;
    }

    
    S* Cin  = load_ptr_batch<S>(CCin,  bid, shiftCin,  strideCin);
    S* Cout = load_ptr_batch<S>(CCout, bid, shiftCout, strideCout);

    S* src =  Cin + ldcin  * gid;
    S* dst = Cout + ldcout * pos;
    
    for(int chunk = 0; chunk < n_chunks; chunk++) {
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                bval[i] = src[x];
        }
        for(int i = 0; i < regs; i++) {
            int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
            if(x < n)
                dst[x] = bval[i];
        }
    }
}


/******************* Host functions *********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
    will be divided during the divide phase of divide & conquer algorithm 
    i.e. number of sub-blocks = 2^levels **/
inline rocblas_int stedc_num_levels(const rocblas_int n)
{
    rocblas_int levels;

    if(n <= 16)
        levels = 0;
    else
        levels = std::ceil(std::log2(n)) - 4;

    //return 3;

    return levels;
}

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack,
                                   size_t* size_tempvect,
                                   size_t* size_tempgemm,
                                   size_t* size_tmpz,
                                   size_t* size_splits_map,
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
        *size_splits_map = 0;
        *size_tmpz = 0;
        return;
    }

    // if no eigenvectors required with classic solver
    if(evect == rocblas_evect_none)
    {
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits_map = 0;
        *size_tmpz = 0;
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
    }

    // if size is too small with classic solver
    else if(n < STEDC_MIN_DC_SIZE)
    {
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits_map = 0;
        *size_tmpz = 0;
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // requirements for solver of small independent blocks
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);

        // extra requirements for original eigenvectors of small independent blocks
        if(evect != rocblas_evect_tridiagonal)
            *size_tempvect = sizeof(S) * (n * n) * batch_count;
        else
            *size_tempvect = 0;
        *size_tempgemm = sizeof(S) * 2 * (n * n) * batch_count;
        if(BATCHED && !COMPLEX)
            *size_workArr = sizeof(S*) * batch_count;
        else
            *size_workArr = 0;

        // size for split blocks and sub-blocks positions
        *size_splits_map = sizeof(rocblas_int) * get_splits_size(n) * batch_count;

        // size for temporary diagonal and rank-1 modif vector
        *size_tmpz = sizeof(S) * get_tmpz_size(n) * batch_count;
    }
}

//--------------------------------------------------------------------------------------//
/** This helper check argument correctness for stedc API **/
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
    if((n && !D) || (n > 1 && !E) || (evect != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//--------------------------------------------------------------------------------------//
/** STEDC templated function **/
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

#define DEBUG_OUTPUT 1
#if DEBUG_OUTPUT
    static int global_cnt = 0;
    global_cnt++;
#endif

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    auto const splits_map = splits;

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

    // if no eigenvectors required with the classic solver, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_template<S>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                    batch_count, static_cast<rocblas_int*>(work_stack));
    }

    // if size is too small with classic solver, use steqr
    else if(n < STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // initialize temporary array for vector updates
        size_t size_tempgemm = sizeof(S) * 2 * n * n * batch_count;
        HIP_CHECK(hipMemsetAsync((void*)tempgemm, 0, size_tempgemm, stream));

        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        S one = 1.0;
        S zero = 0.0;

        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);

        // find number of sub-blocks 
        rocblas_int levs = stedc_num_levels(n);
        char* env_levs = std::getenv("LEVS");
        if (env_levs) {
            levs = env_levs[0] - '0';
        }

        rocblas_int blks = 1 << levs;

        // initialize identity matrix in V
        // if evect is tridiagonal we can store V directly in C
        // otherwise, they must be kept separate to compute C*V
        S* V = tempvect;
        rocblas_int ldv = n;
        rocblas_stride strideV = n * n;
        if(evect == rocblas_evect_tridiagonal)
        {
            V = (S*)(C + shiftC);
            ldv = (rocblas_int)(sizeof(T) / sizeof(S)) * ldc;
            strideV = (rocblas_int)(sizeof(T) / sizeof(S)) * strideC;
        }
        rocblas_int groupsn = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, V, 0, ldv, strideV);

        // 1. divide phase
        //-----------------------------
        rocblas_int groups = (batch_count - 1) / STEDC_BDIM + 1;
        ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<S>),
                                dim3(groups), dim3(STEDC_BDIM), 0, stream, levs, blks, n, D + shiftD,
                                strideD, E + shiftE, strideE, batch_count, splits);

        // 2. solve phase
        //-----------------------------
        ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>),
                                dim3(blks, batch_count), dim3(64), 0, stream, levs, blks, 
                                n, D + shiftD, strideD, E + shiftE, strideE, 
                                V, 0, ldv, strideV, info, (S*)work_stack, splits, 
                                eps, ssfmin, ssfmax);

        ROCSOLVER_LAUNCH_KERNEL(stedc_init_splits, dim3(1, batch_count), dim3(STEDC_BDIM), 0,
                                stream, levs, n, splits);

        // 3. merge phase
        //----------------
        size_t lmemsize3 = sizeof(S) * STEDC_BDIM;
        rocblas_int numgrps3 = ((n - 1) / blks + 1) * blks;

        // launch merge for level k
        for(rocblas_int k = 0; k < levs; ++k)
        {
            rocblas_int n_merges = 1 << (levs - k - 1);
            ROCSOLVER_LAUNCH_KERNEL(stedc_update_splits, dim3(1, batch_count), dim3(STEDC_BDIM), 0,
                                    stream, levs, k, n, splits);

            // a. prepare secular equations
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_DeflateZero_kernel<S>),
                                    dim3(n_merges, batch_count), dim3(STEDC_BDIM), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, V, 0, ldv, strideV, tmpz, splits,
                                    eps);
#if DEBUG_OUTPUT
            //hipError_t status = hipStreamSynchronize(stream);
            if (env_levs && global_cnt == 1) {
                std::cout << "\nk=" << k << std::endl;
                //print_device_matrix(std::cout, "ns", 1, 1 << levs, ptr_ns(n, splits), 1);
                //print_device_matrix(std::cout, "ps", 1, 1 << levs, ptr_ps(n, splits), 1);
                print_device_matrix(std::cout, "msz", 1, n_merges, ptr_msz(n, splits), 1);
                print_device_matrix(std::cout, "mps", 1, n_merges, ptr_mps(n, splits), 1);
                print_device_matrix(std::cout, "em",  1, n, ptr_em(n, splits), 1);
                print_device_matrix(std::cout, "esz", 1, n, ptr_esz(n, splits), 1);
                print_device_matrix(std::cout, "eps", 1, n, ptr_eps(n, splits), 1);
                print_device_matrix(std::cout, "dbg", 1, n, ptr_dbg(n, splits), 1);

                //print_device_matrix(std::cout, "D", 1, n, D, 1);
                //print_device_matrix(std::cout, "mtols", 1, n, tempgemm + n * 4, 1);
                //print_device_matrix(std::cout, "mD", 1, n, tempgemm + n * 5, 1);
            }
#endif

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_SortD_kernel<S>),
                                    dim3(n, batch_count), dim3(STEDC_BDIM), 0, stream,
                                    levs, blks, k, n, D + shiftD, strideD,
                                    tmpz, splits);
            rocblas_int numgrps_deflate = (n - 1) / STEDC_BDIM + 1;
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_SetCandFlags_kernel<S>),
                                    dim3(numgrps_deflate, batch_count), dim3(STEDC_BDIM), 0, stream, levs,
                                    blks, k, n, D + shiftD, strideD, tmpz, splits);
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_DeflateCount_kernel<S>),
                                    dim3(numgrps_deflate, batch_count), dim3(STEDC_BDIM), 0, stream, levs,
                                    blks, k, n, D + shiftD, strideD, tmpz, splits);
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_DeflateApply_kernel<S>),
                                    dim3(numgrps_deflate, batch_count), dim3(STEDC_BDIM), 0, stream,
                                    levs, blks, k, n, D + shiftD, strideD,
                                    tmpz, splits);
                
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeRotate_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, V, 0, ldv,
                                    strideV, tmpz, splits);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_Organize_kernel<S>),
                                    dim3(n_merges, batch_count), dim3(STEDC_BDIM), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, V, 0, ldv, strideV, tmpz, tempgemm, splits,
                                    eps);

            // b. solve secular eq to find merged eigenvalues
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_Sort_kernel<S>), dim3(n_merges, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD,
                                    strideD, E + shiftE, strideE, tmpz, tempgemm, splits, eps,
                                    ssfmin, ssfmax);

            rocblas_int max_n_per_merge  = (n + n_merges - 1) / n_merges;
            rocblas_int groups_per_merge = (max_n_per_merge + STEDC_BDIM - 1) / STEDC_BDIM;
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_Solve_kernel<S>),
                                    dim3(n_merges * groups_per_merge, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD,
                                    strideD, E + shiftE, strideE, tmpz, tempgemm, splits, eps,
                                    ssfmin, ssfmax, groups_per_merge);
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_Rescale_kernel<S>),
                                    dim3(n_merges, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD,
                                    strideD, E + shiftE, strideE, tmpz, tempgemm, splits, eps,
                                    ssfmin, ssfmax);

            // c. find merged eigenvectors
            ROCSOLVER_LAUNCH_KERNEL(
                (stedc_mergeVectors_kernel<STEDC_EXTERNAL_GEMM, S>),
                dim3(numgrps3, batch_count), dim3(STEDC_BDIM), lmemsize3, stream, 
                levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE, V, 0, ldv, strideV, 
                tmpz, tempgemm, splits);

            if(STEDC_EXTERNAL_GEMM)
            {
                // using external gemms with padded matrices to do the vector update
                // One single full gemm of size n x n x n merges all the blocks in the level
                // TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
                // STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
                // external gemm based updates.
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                               &one, V, 0, ldv, strideV, tempgemm, n * n, n, 2 * n * n, &zero,
                               tempgemm, 0, n, 2 * n * n, batch_count, workArr);
            }

            // d. update level
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<S>),
                                    dim3(numgrps3, batch_count), dim3(STEDC_BDIM), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    V, 0, ldv, strideV, tmpz, tempgemm, splits);
        }

        // 4. update and sort
        //----------------------
        if(evect != rocblas_evect_tridiagonal)
        {
            // eigenvectors C <- C*V
            local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, V, tempgemm,
                                            tempgemm + strideV, 0, ldv, strideV, batch_count,
                                            workArr);
        }
        else if constexpr(rocblas_is_complex<T>)
        {
            // V is stored in C but is of type S; need to convert to type T
            // tempgemm = V
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                    0, stream, copymat_to_buffer, n, n, V, 0, ldv, strideV, tempgemm);

            // imag(C) = zeros
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

            // real(C) = tempgemm
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, C, shiftC,
                                    ldc, strideC, tempgemm);
        }

        ROCSOLVER_LAUNCH_KERNEL(stedc_copyD, dim3(1, batch_count), dim3(STEDC_BDIM), 0, stream, n,
                                D + shiftD, strideD, tmpz, n);

        ROCSOLVER_LAUNCH_KERNEL(stedc_copyC, dim3(n, batch_count), dim3(STEDC_BDIM), 0, stream, n,
                                (S*)C,    shiftC, ldc, strideC,
                                tempgemm, 0,      n,   n*n);

        ROCSOLVER_LAUNCH_KERNEL(stedc_sort, dim3(n, batch_count), dim3(STEDC_BDIM), 0, stream,
                                n, tmpz, n, D + shiftD, strideD,
                                tempgemm, 0,      n,   n*n,
                                (S*)C,    shiftC, ldc, strideC);

        rocblas_set_pointer_mode(handle, old_mode);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
