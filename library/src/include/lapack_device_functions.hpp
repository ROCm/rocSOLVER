/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "lib_device_helpers.hpp"
#include "lib_macros.hpp"
#include "rocsolver.h"

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

template <typename T>
__device__ void gemv_kernel(const rocblas_int m,
                            const rocblas_int n,
                            T* alpha,
                            T* a,
                            const rocblas_int lda,
                            T* x,
                            const rocblas_int incX,
                            T* beta,
                            T* y,
                            const rocblas_int incY)
{
    // gemv kernel assuming no transpose
    T yi;
    for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
    {
        yi = 0;

        if(*alpha != 0)
        {
            for(int k = 0; k < n; k++)
                yi += a[i + k * lda] * x[k * incX];
        }

        y[i * incY] = *alpha * yi + *beta * y[i * incY];
    }
    __syncthreads();
}

template <typename T>
__device__ void gemm_kernel(const rocblas_int m,
                            const rocblas_int n,
                            const rocblas_int k,
                            T* alpha,
                            T* a,
                            const rocblas_int lda,
                            T* b,
                            const rocblas_int ldb,
                            T* beta,
                            T* c,
                            const rocblas_int ldc)
{
    // gemm kernel assuming no transpose
    T cij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            cij = 0;

            if(*alpha != 0)
            {
                for(int l = 0; l < k; l++)
                    cij += a[i + l * lda] * b[l + j * ldb];
            }

            c[i + j * ldc] = *alpha * cij + *beta * c[i + j * ldc];
        }
        __syncthreads();
    }
}

/** LARTG device function computes the sine (s) and cosine (c) values
    to create a givens rotation such that:
    [  c s ]' * [ f ] = [ r ]
    [ -s c ]    [ g ]   [ 0 ] **/
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
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
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
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
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
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
template <int MAX_THDS, typename T>
__device__ void iamax(const rocblas_int tid,
                      const rocblas_int n,
                      T* A,
                      const rocblas_int incA,
                      T* sval,
                      rocblas_int* sidx)
{
    using S = decltype(std::real(T{}));

    // local memory setup
    T val1, val2;
    rocblas_int idx1, idx2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    idx1 = INT_MAX;
    for(int i = tid; i < n; i += MAX_THDS)
    {
        val2 = A[i * incA];
        idx2 = i + 1; // add one to make it 1-based index
        if(aabs<S>(val1) < aabs<S>(val2) || idx1 == INT_MAX)
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
            if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
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
            if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        val2 = sval[tid + 32];
        idx2 = sidx[tid + 32];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 16];
        idx2 = sidx[tid + 16];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 8];
        idx2 = sidx[tid + 8];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 4];
        idx2 = sidx[tid + 4];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 2];
        idx2 = sidx[tid + 2];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 1];
        idx2 = sidx[tid + 1];
        if(aabs<S>(val1) < aabs<S>(val2) || (aabs<S>(val1) == aabs<S>(val2) && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }

    // after the reduction, the maximum of the elements is in sval[0] and sidx[0]
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
