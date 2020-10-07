/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORG2L_UNG2L_HPP
#define ROCLAPACK_ORG2L_UNG2L_HPP

//#include "rocauxiliary_lacgv.hpp"
#include "rocauxiliary_larf.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U>
__global__ void org2l_init_ident(const rocblas_int m,
                                 const rocblas_int n,
                                 const rocblas_int k,
                                 U A,
                                 const rocblas_int shiftA,
                                 const rocblas_int lda,
                                 const rocblas_stride strideA)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < m && j < n)
    {
        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        if(i == m - n + j)
            // ones along the (m-n)th subdiagonal
            Ap[i + j * lda] = 1.0;
        else if(i > m - n + j)
            // zero the lower triangular factor L
            Ap[i + j * lda] = 0.0;
        else if(j < n - k)
            // zero the left part of the matrix, leaving k Householder vectors
            Ap[i + j * lda] = 0.0;
    }
}

template <typename T, bool BATCHED>
void rocsolver_org2l_ung2l_getMemorySize(const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_Abyx,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_Abyx = 0;
        *size_workArr = 0;
        return;
    }

    // memory requirements to call larf
    rocsolver_larf_getMemorySize<T, BATCHED>(rocblas_side_left, m, n, batch_count, size_scalars,
                                             size_Abyx, size_workArr);
}

template <typename T, typename U>
rocblas_status rocsolver_org2l_orgql_argCheck(const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || m < n || k < 0 || k > n || lda < m)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((k && !ipiv) || (m * n && !A))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_org2l_ung2l_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* Abyx,
                                              T** workArr)
{
    // quick return
    if(!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // Initialize identity matrix (non used rows)
    rocblas_int blocksx = (m - 1) / 32 + 1;
    rocblas_int blocksy = (n - 1) / 32 + 1;
    hipLaunchKernelGGL(org2l_init_ident<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                       stream, m, n, k, A, shiftA, lda, strideA);

    for(rocblas_int j = 0; j < k; ++j)
    {
        rocblas_int jj = n - k + j;

        // apply H(i) to Q(1:m-k+i,1:n-k+i) from the left
        rocsolver_larf_template<T>(handle, rocblas_side_left, m - n + jj + 1, jj, A,
                                   shiftA + idx2D(0, jj, lda), 1, strideA, (ipiv + j), strideP, A,
                                   shiftA, lda, strideA, batch_count, scalars, Abyx, workArr);

        // set the diagonal element and negative tau
        hipLaunchKernelGGL(subtract_tau<T>, dim3(batch_count), dim3(1), 0, stream, m - n + jj, jj,
                           A, shiftA, lda, strideA, ipiv + j, strideP);

        // update i-th column -corresponding to H(i)-
        rocblasCall_scal<T>(handle, m - n + jj, ipiv + j, strideP, A, shiftA + idx2D(0, jj, lda), 1,
                            strideA, batch_count);
    }

    // restore values of tau
    blocksx = (k - 1) / 128 + 1;
    hipLaunchKernelGGL(restau<T>, dim3(blocksx, batch_count), dim3(128), 0, stream, k, ipiv, strideP);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif
