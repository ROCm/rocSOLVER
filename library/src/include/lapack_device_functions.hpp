/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "lib_device_helpers.hpp"
#include "lib_macros.hpp"
#include "rocsolver/rocsolver.h"

/*
 * ===========================================================================
 *    common location for device functions and kernels that reproduce LAPACK
 *    and BLAS functionality. Includes some reproduction of rocBLAS
 *    functionality since rocBLAS cannot be called from within a kernel.
 * ===========================================================================
 */

template <typename T>
__device__ void trtri_kernel_upper(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming upper triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = 1; j < n; j++)
    {
        if(i < j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i < j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i + 1; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trtri_kernel_lower(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming lower triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = n - 2; j >= 0; j--)
    {
        if(i > j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i > j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i - 1; ii > j; ii--)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_upper(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, upper triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = i + 1; k < m; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_lower(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, lower triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = 0; k < i; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_upper(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, upper triangular matrix from the right
    T ajj, bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = 0; k < j; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_lower(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, lower triangular matrix from the right
    T ajj, bij;
    for(int j = n - 1; j >= 0; j--)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = j + 1; k < n; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

/** GEMV device function to compute y = alpha * A * x + beta * y **/
template <int MAX_THDS, typename T>
__device__ void gemv(const rocblas_int tid,
                     const rocblas_int m,
                     const rocblas_int n,
                     const T* alpha,
                     T* A,
                     const rocblas_int lda,
                     T* x,
                     const rocblas_int incx,
                     const T* beta,
                     T* y,
                     const rocblas_int incy)
{
    // gemv function assuming no transpose
    T temp;
    for(int i = tid; i < m; i += MAX_THDS)
    {
        temp = 0;
        for(int j = 0; j < n; j++)
            temp += A[i + j * lda] * x[j * incx];
        y[i * incy] = *alpha * temp + *beta * y[i * incy];
    }
}

/** GEMM device function to compute C = alpha * A * B + beta * C **/
template <int MAX_THDS, typename T>
__device__ void gemm(const rocblas_int tid,
                     const rocblas_int m,
                     const rocblas_int n,
                     const rocblas_int k,
                     const T* alpha,
                     T* A,
                     const rocblas_int lda,
                     T* B,
                     const rocblas_int ldb,
                     const T* beta,
                     T* C,
                     const rocblas_int ldc)
{
    // gemm function assuming no transpose
    T temp;
    for(int e = tid; e < m * n; e += MAX_THDS)
    {
        int i = e % m;
        int j = e / m;
        temp = 0;
        for(int l = 0; l < k; l++)
            temp += A[i + l * lda] * B[l + j * ldb];
        C[i + j * ldc] = *alpha * temp + *beta * C[i + j * ldc];
    }
}

/** GEMM device function to compute C = alpha * A * B' + beta * C **/
template <int MAX_THDS, typename T>
__device__ void gemm_btrans(const rocblas_int tid,
                            const rocblas_int m,
                            const rocblas_int n,
                            const rocblas_int k,
                            const T* alpha,
                            T* A,
                            const rocblas_int lda,
                            T* B,
                            const rocblas_int ldb,
                            const T* beta,
                            T* C,
                            const rocblas_int ldc)
{
    // gemm function assuming B transpose
    T temp;
    for(int e = tid; e < m * n; e += MAX_THDS)
    {
        int i = e % m;
        int j = e / m;
        temp = 0;
        for(int l = 0; l < k; l++)
            temp += A[i + l * lda] * B[j + l * ldb];
        C[i + j * ldc] = *alpha * temp + *beta * C[i + j * ldc];
    }
}

/** LARTG device function computes the sine (s) and cosine (c) values
    to create a givens rotation such that:
    [  c s ]' * [ f ] = [ r ]
    [ -s c ]    [ g ]   [ 0 ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void lartg(T& f, T& g, T& c, T& s, T& r)
{
    if(g == 0)
    {
        c = 1;
        s = 0;
        r = f;
    }
    else if(f == 0)
    {
        c = 0;
        s = 1;
        r = -g;
    }
    else
    {
        T t;
        if(std::abs(g) > std::abs(f))
        {
            t = -f / g;
            s = 1 / std::sqrt(1 + t * t);
            c = s * t;
        }
        else
        {
            t = -g / f;
            c = 1 / std::sqrt(1 + t * t);
            s = c * t;
        }
        r = c * f - s * g;
    }
}

/** LASR device function applies a sequence of rotations P(i) i=1,2,...z
    to a m-by-n matrix A from either the left (P*A with z=m) or the right (A*P'
    with z=n). P = P(z-1)*...*P(1) if forward direction, P = P(1)*...*P(z-1) if
    backward direction. **/
template <typename T, typename W>
__device__ void lasr(const rocblas_side side,
                     const rocblas_direct direc,
                     const rocblas_int m,
                     const rocblas_int n,
                     W* c,
                     W* s,
                     T* A,
                     const rocblas_int lda)
{
    T temp;
    W cs, sn;

    if(side == rocblas_side_left)
    {
        if(direc == rocblas_forward_direction)
        {
            for(rocblas_int i = 0; i < m - 1; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    temp = A[i + j * lda];
                    cs = c[i];
                    sn = s[i];
                    A[i + j * lda] = cs * temp + sn * A[i + 1 + j * lda];
                    A[i + 1 + j * lda] = cs * A[i + 1 + j * lda] - sn * temp;
                }
            }
        }
        else
        {
            for(rocblas_int i = m - 1; i > 0; --i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    temp = A[i + j * lda];
                    cs = c[i - 1];
                    sn = s[i - 1];
                    A[i + j * lda] = cs * temp - sn * A[i - 1 + j * lda];
                    A[i - 1 + j * lda] = cs * A[i - 1 + j * lda] + sn * temp;
                }
            }
        }
    }

    else
    {
        if(direc == rocblas_forward_direction)
        {
            for(rocblas_int j = 0; j < n - 1; ++j)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    temp = A[i + j * lda];
                    cs = c[j];
                    sn = s[j];
                    A[i + j * lda] = cs * temp + sn * A[i + (j + 1) * lda];
                    A[i + (j + 1) * lda] = cs * A[i + (j + 1) * lda] - sn * temp;
                }
            }
        }
        else
        {
            for(rocblas_int j = n - 1; j > 0; --j)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    temp = A[i + j * lda];
                    cs = c[j - 1];
                    sn = s[j - 1];
                    A[i + j * lda] = cs * temp - sn * A[i + (j - 1) * lda];
                    A[i + (j - 1) * lda] = cs * A[i + (j - 1) * lda] + sn * temp;
                }
            }
        }
    }
}

/** LAE2 computes the eigenvalues of a 2x2 symmetric matrix
    [ a b ]
    [ b c ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void lae2(T& a, T& b, T& c, T& rt1, T& rt2)
{
    T sm = a + c;
    T adf = abs(a - c);
    T ab = abs(b + b);

    T rt, acmx, acmn;
    if(adf > ab)
    {
        rt = ab / adf;
        rt = adf * sqrt(1 + rt * rt);
    }
    else if(adf < ab)
    {
        rt = adf / ab;
        rt = ab * sqrt(1 + rt * rt);
    }
    else
        rt = ab * sqrt(2);

    // Compute the eigenvalues
    if(abs(a) > abs(c))
    {
        acmx = a;
        acmn = c;
    }
    else
    {
        acmx = c;
        acmn = a;
    }
    if(sm < 0)
    {
        rt1 = T(0.5) * (sm - rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else if(sm > 0)
    {
        rt1 = T(0.5) * (sm + rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else
    {
        rt1 = T(0.5) * rt;
        rt2 = T(-0.5) * rt;
    }
}

/** LAEV2 computes the eigenvalues and eigenvectors of a 2x2 symmetric matrix
    [ a b ]
    [ b c ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void laev2(T& a, T& b, T& c, T& rt1, T& rt2, T& cs1, T& sn1)
{
    int sgn1, sgn2;

    T sm = a + c;
    T df = a - c;
    T adf = abs(df);
    T tb = b + b;
    T ab = abs(tb);

    T rt, temp1, temp2;
    if(adf > ab)
    {
        rt = ab / adf;
        rt = adf * sqrt(1 + rt * rt);
    }
    else if(adf < ab)
    {
        rt = adf / ab;
        rt = ab * sqrt(1 + rt * rt);
    }
    else
        rt = ab * sqrt(2);

    // Compute the eigenvalues
    if(abs(a) > abs(c))
    {
        temp1 = a;
        temp2 = c;
    }
    else
    {
        temp1 = c;
        temp2 = a;
    }
    if(sm < 0)
    {
        sgn1 = -1;
        rt1 = T(0.5) * (sm - rt);
        rt2 = T((temp1 / (double)rt1) * temp2 - (b / (double)rt1) * b);
    }
    else if(sm > 0)
    {
        sgn1 = 1;
        rt1 = T(0.5) * (sm + rt);
        rt2 = T((temp1 / (double)rt1) * temp2 - (b / (double)rt1) * b);
    }
    else
    {
        sgn1 = 1;
        rt1 = T(0.5) * rt;
        rt2 = T(-0.5) * rt;
    }

    // Compute the eigenvector
    if(df >= 0)
    {
        temp1 = df + rt;
        sgn2 = 1;
    }
    else
    {
        temp1 = df - rt;
        sgn2 = -1;
    }

    if(abs(temp1) > ab)
    {
        // temp2 is cotan
        temp2 = -tb / temp1;
        sn1 = T(1) / sqrt(1 + temp2 * temp2);
        cs1 = temp2 * sn1;
    }
    else
    {
        if(ab == 0)
        {
            cs1 = 1;
            sn1 = 0;
        }
        else
        {
            // temp2 is tan
            temp2 = -temp1 / tb;
            cs1 = T(1) / sqrt(1 + temp2 * temp2);
            sn1 = temp2 * cs1;
        }
    }

    if(sgn1 == sgn2)
    {
        temp1 = cs1;
        cs1 = -sn1;
        sn1 = temp1;
    }
}

/** LASRT_INCREASING sorts an array D in increasing order.
    stack is a 32x2 array of integers on the device. **/
template <typename T>
__device__ void lasrt_increasing(const rocblas_int n, T* D, rocblas_int* stack)
{
    /** (TODO: Current implementation is failling for large sizes. Not removed for now
        as quick-sort methods could be required for performance purposes in the future.
        It should be debugged some time.) **/
    T d1, d2, d3, dmnmx, temp;
    constexpr rocblas_int select = 20;
    constexpr rocblas_int lds = 32;
    rocblas_int i, j, start, endd;
    rocblas_int stackptr = 0;

    // Initialize stack[0, 0] and stack[1, 0]
    stack[0 + 0 * lds] = 0;
    stack[1 + 0 * lds] = n - 1;
    while(stackptr >= 0)
    {
        start = stack[0 + stackptr * lds];
        endd = stack[1 + stackptr * lds];
        stackptr--;

        if(endd - start <= select && endd - start > 0)
        {
            // Insertion sort
            for(i = start + 1; i <= endd; i++)
            {
                for(j = i; j > start; j--)
                {
                    if(D[j] < D[j - 1])
                    {
                        dmnmx = D[j];
                        D[j] = D[j - 1];
                        D[j - 1] = dmnmx;
                    }
                    else
                        break;
                }
            }
        }
        else if(endd - start > select)
        {
            // Partition and add to stack
            d1 = D[start];
            d2 = D[endd];
            i = (start + endd) / 2;
            d3 = D[i];

            if(d1 < d2)
            {
                if(d3 < d1)
                    dmnmx = d1;
                else if(d3 < d2)
                    dmnmx = d3;
                else
                    dmnmx = d2;
            }
            else
            {
                if(d3 < d2)
                    dmnmx = d2;
                else if(d3 < d1)
                    dmnmx = d3;
                else
                    dmnmx = d1;
            }

            i = start;
            j = endd;
            while(i < j)
            {
                while(D[i] < dmnmx)
                    i++;
                while(D[j] > dmnmx)
                    j--;
                if(i < j)
                {
                    temp = D[i];
                    D[i] = D[j];
                    D[j] = temp;
                }
            }
            if(j - start > endd - j - 1)
            {
                stackptr++;
                stack[0 + stackptr * lds] = start;
                stack[1 + stackptr * lds] = j;
                stackptr++;
                stack[0 + stackptr * lds] = j + 1;
                stack[1 + stackptr * lds] = endd;
            }
            else
            {
                stackptr++;
                stack[0 + stackptr * lds] = j + 1;
                stack[1 + stackptr * lds] = endd;
                stackptr++;
                stack[0 + stackptr * lds] = start;
                stack[1 + stackptr * lds] = j;
            }
        }
    }
}

/** IAMAX finds the maximum element of a given vector and its index.
    MAX_THDS should be 128, 256, 512, or 1024, and sval and sidx should
    be shared arrays of size MAX_THDS. **/
template <int MAX_THDS, typename T, typename S>
__device__ void iamax(const rocblas_int tid,
                      const rocblas_int n,
                      T* A,
                      const rocblas_int incA,
                      S* sval,
                      rocblas_int* sidx)
{
    // local memory setup
    S val1, val2;
    rocblas_int idx1, idx2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    idx1 = INT_MAX;
    for(int i = tid; i < n; i += MAX_THDS)
    {
        val2 = aabs<S>(A[i * incA]);
        idx2 = i + 1; // add one to make it 1-based index
        if(val1 < val2 || idx1 == INT_MAX)
        {
            val1 = val2;
            idx1 = idx2;
        }
    }
    sval[tid] = val1;
    sidx[tid] = idx1;
    __syncthreads();

    if(n <= 1)
        return;

        /** <========= Next do the reduction on the shared memory array =========>
        (We halve the number of active threads at each step
        reducing two elements in the shared array. **/

#pragma unroll
    for(int i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
        {
            val2 = sval[tid + i];
            idx2 = sidx[tid + i];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        __syncthreads();
    }

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < warpSize)
    {
        if(warpSize >= 64)
        {
            val2 = sval[tid + 64];
            idx2 = sidx[tid + 64];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        val2 = sval[tid + 32];
        idx2 = sidx[tid + 32];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 16];
        idx2 = sidx[tid + 16];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 8];
        idx2 = sidx[tid + 8];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 4];
        idx2 = sidx[tid + 4];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 2];
        idx2 = sidx[tid + 2];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 1];
        idx2 = sidx[tid + 1];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }

    // after the reduction, the maximum of the elements is in sval[0] and sidx[0]
}

/** NRM2 finds the euclidean norm of a given vector.
    MAX_THDS should be 128, 256, 512, or 1024, and sval should
    be a shared array of size MAX_THDS. **/
template <int MAX_THDS, typename T>
__device__ void nrm2(const rocblas_int tid, const rocblas_int n, T* A, const rocblas_int incA, T* sval)
{
    // local memory setup
    T val = 0;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    for(int i = tid; i < n; i += MAX_THDS)
        val = val + A[i * incA] * A[i * incA];
    sval[tid] = val;
    __syncthreads();

    if(n <= 1)
    {
        if(tid == 0)
            sval[0] = sqrt(sval[0]);
        return;
    }

    /** <========= Next do the reduction on the shared memory array =========>
        (We halve the number of active threads at each step
        reducing two elements in the shared array. **/

#pragma unroll
    for(int i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
            val = val + sval[tid + i];
        __syncthreads();
        if(tid < i)
            sval[tid] = val;
        __syncthreads();
    }

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < warpSize)
    {
        if(warpSize >= 64)
        {
            sval[tid] = sval[tid] + sval[tid + 64];
            __threadfence();
        }
        sval[tid] = sval[tid] + sval[tid + 32];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 16];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 8];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 4];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 2];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 1];
        __threadfence();
    }

    // after the reduction, the euclidean norm of the elements is in sval[0]
    if(tid == 0)
        sval[0] = sqrt(sval[0]);
}

/** LAGTF computes an LU factorization of a matrix T - lambda*I, where T
    is a tridiagonal matrix and lambda is a scalar. **/
template <typename T>
__device__ void lagtf(rocblas_int n, T* a, T lambda, T* b, T* c, T tol, T* d, rocblas_int* in, T eps)
{
    T scale1, scale2, piv1, piv2, mult, temp;

    a[0] = a[0] - lambda;
    in[n - 1] = 0;
    if(n == 1)
    {
        if(a[0] == 0)
            in[0] = 1;
        return;
    }

    tol = max(tol, eps);
    scale1 = abs(a[0]) + abs(b[0]);
    for(rocblas_int k = 0; k < n - 1; k++)
    {
        temp = a[k + 1] - lambda;
        a[k + 1] = temp;
        scale2 = abs(c[k]) + abs(temp);
        if(k < n - 2)
            scale2 = scale2 + abs(b[k + 1]);
        piv1 = (a[k] == 0 ? 0 : abs(a[k]) / scale1);

        if(c[k] == 0)
        {
            in[k] = 0;
            piv2 = 0;
            scale1 = scale2;
            if(k < n - 2)
                d[k] = 0;
        }
        else
        {
            piv2 = abs(c[k]) / scale2;
            if(piv2 <= piv1)
            {
                in[k] = 0;
                scale1 = scale2;
                mult = c[k] / a[k];
                c[k] = mult;
                a[k + 1] = a[k + 1] - mult * b[k];
                if(k < n - 2)
                    d[k] = 0;
            }
            else
            {
                in[k] = 1;
                mult = a[k] / c[k];
                a[k] = c[k];
                a[k + 1] = b[k] - mult * temp;
                if(k < n - 2)
                {
                    d[k] = b[k + 1];
                    b[k + 1] = -mult * b[k + 1];
                }
                b[k] = temp;
                c[k] = mult;
            }
        }

        if(max(piv1, piv2) <= tol && in[n - 1] == 0)
            in[n - 1] = k + 1;
    }

    if(abs(a[n - 1]) <= scale1 * tol && in[n - 1] == 0)
        in[n - 1] = n;
}

/** LAGTS_TYPE1_PERTURB solves the system of equations (T - lambda*I)x = y,
    where T is a tridiagonal matrix and lambda is a scalar. If overflow were
    to occur, the diagonal elements are perturbed. **/
template <typename T>
__device__ void
    lagts_type1_perturb(rocblas_int n, T* a, T* b, T* c, T* d, rocblas_int* in, T* y, T tol, T eps, T ssfmin)
{
    rocblas_int k;
    T temp, pert, ak, absak;

    T bignum = T(1) / ssfmin;
    if(tol <= 0)
    {
        tol = abs(a[0]);
        if(n > 1)
            tol = max(tol, max(abs(a[1]), abs(b[0])));
        for(k = 2; k < n; k++)
            tol = max(max(tol, abs(a[k])), max(abs(b[k - 1]), abs(d[k - 2])));
        tol = tol * eps;
        if(tol == 0)
            tol = eps;
    }

    for(k = 1; k < n; k++)
    {
        if(in[k - 1] == 0)
            y[k] = y[k] - c[k - 1] * y[k - 1];
        else
        {
            temp = y[k - 1];
            y[k - 1] = y[k];
            y[k] = temp - c[k - 1] * y[k];
        }
    }

    for(k = n - 1; k >= 0; k--)
    {
        temp = y[k];
        if(k < n - 1)
            temp = temp - b[k] * y[k + 1];
        if(k < n - 2)
            temp = temp - d[k] * y[k + 2];

        ak = a[k];
        pert = (ak >= 0 ? abs(tol) : -abs(tol));
        while((absak = abs(ak)) < 1)
        {
            if(absak < ssfmin)
            {
                if(absak == 0 || abs(temp) * ssfmin > absak)
                {
                    ak = ak + pert;
                    pert = 2 * pert;
                }
                else
                {
                    temp = temp * bignum;
                    ak = ak * bignum;
                    break;
                }
            }
            else
            {
                if(abs(temp) > absak * bignum)
                {
                    ak = ak + pert;
                    pert = 2 * pert;
                }
                else
                    break;
            }
        }

        y[k] = temp / ak;
    }
}

/** AXPY computes a constant times a vector plus a vector. **/
template <typename T, typename U, typename V>
ROCSOLVER_KERNEL void axpy_kernel(const rocblas_int n,
                                  T* alpha,
                                  const rocblas_stride stride_alpha,
                                  U X,
                                  const rocblas_int shiftX,
                                  const rocblas_int incx,
                                  const rocblas_stride strideX,
                                  V Y,
                                  const rocblas_int shiftY,
                                  const rocblas_int incy,
                                  const rocblas_stride strideY)
{
    rocblas_int b = hipBlockIdx_x;
    rocblas_int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < n)
    {
        T* x = load_ptr_batch<T>(X, b, shiftX, strideX);
        T* y = load_ptr_batch<T>(Y, b, shiftY, strideY);
        T* a = alpha + b * stride_alpha;

        // axpy
        y[i * incy] = a[0] * x[i * incx] + y[i * incy];
    }
}

/** Optimized kernel that executes a simple gemm A = BC
    where A, B and C are sub blocks of the same matrix MM with
    leading dimension ldim and stride. A, B and C are
    located in MM by their respective shifts.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in x and y to cover all the 'm' rows and 'n' columns of C.
    Size of shared memory per group should be:
    lmemsize = k * (hipBlockDim_x + hipBlockDim_y) * sizeof(T); **/
template <typename T, typename U>
ROCSOLVER_KERNEL void gemm_kernel(const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int k,
                                  U MM,
                                  const rocblas_int shiftA,
                                  const rocblas_int shiftB,
                                  const rocblas_int shiftC,
                                  const rocblas_int ldim,
                                  const rocblas_stride stride)
{
    // indices
    int id = hipBlockIdx_z;
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int bdx = hipBlockDim_x;
    int bdy = hipBlockDim_y;
    int i = hipBlockIdx_x * bdx + tx;
    int j = hipBlockIdx_y * bdy + ty;

    // batch instance
    T* A = load_ptr_batch(MM, id, shiftA, stride);
    T* B = load_ptr_batch(MM, id, shiftB, stride);
    T* C = load_ptr_batch(MM, id, shiftC, stride);

    // shared mem setup
    extern __shared__ double lmem[];
    T* a = reinterpret_cast<T*>(lmem);
    T* b = a + k * bdx;
    T c;

    // local row and column of the shared arrays
    a += tx * k;
    b += ty * k;

    // read A and B into shared mem
    for(int kk = ty; kk < k; kk += bdy)
        a[kk] = i < m ? A[i + kk * ldim] : 0;
    for(int kk = tx; kk < k; kk += bdx)
        b[kk] = j < n ? B[kk + j * ldim] : 0;
    __syncthreads();

    if(i < m && j < n)
    {
        // update c
        c = C[i + j * ldim];
        for(int kk = 0; kk < k; ++kk)
            c -= a[kk] * b[kk];

        // write back to global memory
        C[i + j * ldim] = c;
    }
}

// **************** forward substitution kernels ************************//
///////////////////////////////////////////////////////////////////////////
/** The following kernels implement forward substitution for lower triangular L
    or upper triangular U matrices in the form
    LX = B
    U'X = B
    B = XU
    B = XL'

    nx is the number of variables and ny the number of right/left-hand-sides.
    Whether B is accessed by rows (left-hand-sides) or columns (right-hand-sides) is
    determined by the values of ldb1 and ldb2. Whether L/U is transposed or not is
    determined by the values of lda1 and lda2.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in y to cover all the 'ny' right/left-hand-sides (columns/rows of B).
    There should be only one group in x with hipBlockDim_x = nx.
    Size of shared memory per group should be:
    lmemsize = hipBlockDim_y * sizeof(T);

    There are 4 different forward substitution kernels; each one deals with
    a combination of unit and conjugate. In the non-unit case, the kernles DO NOT
    verify whether the diagonal element of L/U is non-zero.**/
template <typename T, typename U>
ROCSOLVER_KERNEL void unit_forward_substitution_kernel(const rocblas_int nx,
                                                       const rocblas_int ny,
                                                       U AA,
                                                       const rocblas_int lda1,
                                                       const rocblas_int lda2,
                                                       const rocblas_int shiftA,
                                                       const rocblas_stride strideA,
                                                       U BB,
                                                       const rocblas_int ldb1,
                                                       const rocblas_int ldb2,
                                                       const rocblas_int shiftB,
                                                       const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x > k) ? A[ida + k * lda2] * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_unit_forward_substitution_kernel(const rocblas_int nx,
                                                            const rocblas_int ny,
                                                            U AA,
                                                            const rocblas_int lda1,
                                                            const rocblas_int lda2,
                                                            const rocblas_int shiftA,
                                                            const rocblas_stride strideA,
                                                            U BB,
                                                            const rocblas_int ldb1,
                                                            const rocblas_int ldb2,
                                                            const rocblas_int shiftB,
                                                            const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x > k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void nonunit_forward_substitution_kernel(const rocblas_int nx,
                                                          const rocblas_int ny,
                                                          U AA,
                                                          const rocblas_int lda1,
                                                          const rocblas_int lda2,
                                                          const rocblas_int shiftA,
                                                          const rocblas_stride strideA,
                                                          U BB,
                                                          const rocblas_int ldb1,
                                                          const rocblas_int ldb2,
                                                          const rocblas_int shiftB,
                                                          const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / A[x * (lda1 + lda2)];
                b[ty] = c;
            }
            __syncthreads();

            c -= (x > k) ? A[ida + k * lda2] * b[ty] : 0;
        }
        if(x == nx - 1)
            c = c / A[x * (lda1 + lda2)];

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_nonunit_forward_substitution_kernel(const rocblas_int nx,
                                                               const rocblas_int ny,
                                                               U AA,
                                                               const rocblas_int lda1,
                                                               const rocblas_int lda2,
                                                               const rocblas_int shiftA,
                                                               const rocblas_stride strideA,
                                                               U BB,
                                                               const rocblas_int ldb1,
                                                               const rocblas_int ldb2,
                                                               const rocblas_int shiftB,
                                                               const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = 0; k < nx - 1; ++k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / conj(A[x * (lda1 + lda2)]);
                b[ty] = c;
            }
            __syncthreads();

            c -= (x > k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }
        if(x == nx - 1)
            c = c / conj(A[x * (lda1 + lda2)]);

        // move results back to global
        B[idb] = c;
    }
}

// **************** backward substitution kernels ************************//
////////////////////////////////////////////////////////////////////////////
/** The following kernels implement backward substitution for lower triangular L
    or upper triangular U matrices in the form
    L'X = B
    UX = B
    B = XU'
    B = XL

    nx is the number of variables and ny the number of right/left-hand-sides.
    Whether B is accessed by rows (left-hand-sides) or columns (right-hand-sides) is
    determined by the values of ldb1 and ldb2. Whether L/U is transposed or not is
    determined by the values of lda1 and lda2.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in y to cover all the 'ny' right/left-hand-sides (columns/rows of B).
    There should be only one group in x with hipBlockDim_x = nx.
    Size of shared memory per group should be:
    lmemsize = hipBlockDim_y * sizeof(T);

    There are 4 different backward substitution kernels; each one deals with
    a combination of unit and conjugate. In the non-unit case, the kernles DO NOT
    verify whether the diagonal element of L/U is non-zero.**/
template <typename T, typename U>
ROCSOLVER_KERNEL void unit_backward_substitution_kernel(const rocblas_int nx,
                                                        const rocblas_int ny,
                                                        U AA,
                                                        const rocblas_int lda1,
                                                        const rocblas_int lda2,
                                                        const rocblas_int shiftA,
                                                        const rocblas_stride strideA,
                                                        U BB,
                                                        const rocblas_int ldb1,
                                                        const rocblas_int ldb2,
                                                        const rocblas_int shiftB,
                                                        const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x < k) ? A[ida + k * lda2] * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_unit_backward_substitution_kernel(const rocblas_int nx,
                                                             const rocblas_int ny,
                                                             U AA,
                                                             const rocblas_int lda1,
                                                             const rocblas_int lda2,
                                                             const rocblas_int shiftA,
                                                             const rocblas_stride strideA,
                                                             U BB,
                                                             const rocblas_int ldb1,
                                                             const rocblas_int ldb2,
                                                             const rocblas_int shiftB,
                                                             const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
                b[ty] = c;
            __syncthreads();

            c -= (x < k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void nonunit_backward_substitution_kernel(const rocblas_int nx,
                                                           const rocblas_int ny,
                                                           U AA,
                                                           const rocblas_int lda1,
                                                           const rocblas_int lda2,
                                                           const rocblas_int shiftA,
                                                           const rocblas_stride strideA,
                                                           U BB,
                                                           const rocblas_int ldb1,
                                                           const rocblas_int ldb2,
                                                           const rocblas_int shiftB,
                                                           const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / A[x * (lda1 + lda2)];
                b[ty] = c;
            }
            __syncthreads();

            c -= (x < k) ? A[ida + k * lda2] * b[ty] : 0;
        }
        if(x == 0)
            c = c / A[x * (lda1 + lda2)];

        // move results back to global
        B[idb] = c;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void conj_nonunit_backward_substitution_kernel(const rocblas_int nx,
                                                                const rocblas_int ny,
                                                                U AA,
                                                                const rocblas_int lda1,
                                                                const rocblas_int lda2,
                                                                const rocblas_int shiftA,
                                                                const rocblas_stride strideA,
                                                                U BB,
                                                                const rocblas_int ldb1,
                                                                const rocblas_int ldb2,
                                                                const rocblas_int shiftB,
                                                                const rocblas_stride strideB)
{
    int bid = hipBlockIdx_z;
    int x = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * hipBlockDim_y + ty;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);

    // shared mem setup
    extern __shared__ double lmem[];
    T* b = reinterpret_cast<T*>(lmem);
    T c, d;

    if(y < ny)
    {
        int ida = x * lda1;
        int idb = x * ldb1 + y * ldb2;

        // read data
        c = B[idb];

        // solve for all y's
        for(int k = nx - 1; k > 0; --k)
        {
            __syncthreads();
            if(x == k)
            {
                c = c / conj(A[x * (lda1 + lda2)]);
                b[ty] = c;
            }
            __syncthreads();

            c -= (x < k) ? conj(A[ida + k * lda2]) * b[ty] : 0;
        }
        if(x == 0)
            c = c / conj(A[x * (lda1 + lda2)]);

        // move results back to global
        B[idb] = c;
    }
}
