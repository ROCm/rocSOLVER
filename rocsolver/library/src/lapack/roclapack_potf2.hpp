/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_POTF2_HPP
#define ROCLAPACK_POTF2_HPP

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
__global__ void sqrtDiagOnward(U A,
                               const rocblas_int shiftA,
                               const rocblas_int strideA,
                               const size_t loc,
                               const rocblas_int j,
                               T* res,
                               rocblas_int* info)
{
    int id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    T t = M[loc] - res[id];

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    }

    else
    {
        // minor is positive definite
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}

template <typename T, typename U, std::enable_if_t<is_complex<T>, int> = 0>
__global__ void sqrtDiagOnward(U A,
                               const rocblas_int shiftA,
                               const rocblas_int strideA,
                               const size_t loc,
                               const rocblas_int j,
                               T* res,
                               rocblas_int* info)
{
    int id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    auto t = M[loc].real() - res[id].real();

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    }

    else
    {
        // minor is positive definite
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}

template <typename T>
void rocsolver_potf2_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_pivots)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_pivots = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of workspace
    *size_work = sizeof(T) * ((n - 1) / ROCBLAS_DOT_NB + 2) * batch_count;

    // size of array to store pivots
    *size_pivots = sizeof(T) * batch_count;
}

template <typename T>
rocblas_status rocsolver_potf2_potrf_argCheck(const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T* pivots)
{
    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a positive definite matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // (TODO: When the matrix is detected to be non positive definite, we need to
    //  prevent GEMV and SCAL to modify further the input matrix; ideally with no
    //  synchronizations.)

    if(uplo == rocblas_fill_upper)
    {
        // Compute the Cholesky factorization A = U'*U.
        for(rocblas_int j = 0; j < n; ++j)
        {
            // Compute U(J,J) and test for non-positive-definiteness.
            rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, batch_count, pivots,
                                        work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, A, shiftA,
                               strideA, idx2D(j, j, lda), j, pivots, info);

            // Compute elements J+1:N of row J
            if(j < n - 1)
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                                batch_count);

                rocblasCall_gemv<T>(handle, rocblas_operation_transpose, j, n - j - 1, scalars, 0,
                                    A, shiftA + idx2D(0, j + 1, lda), lda, strideA, A,
                                    shiftA + idx2D(0, j, lda), 1, strideA, scalars + 2, 0, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                    nullptr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                                batch_count);

                rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A, shiftA + idx2D(j, j + 1, lda),
                                    lda, strideA, batch_count);
            }
        }
    }

    else
    {
        // Compute the Cholesky factorization A = L'*L.
        for(rocblas_int j = 0; j < n; ++j)
        {
            // Compute L(J,J) and test for non-positive-definiteness.
            rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA, A,
                                        shiftA + idx2D(j, 0, lda), lda, strideA, batch_count,
                                        pivots, work);

            hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1), 0, stream, A, shiftA,
                               strideA, idx2D(j, j, lda), j, pivots, info);

            // Compute elements J+1:N of row J
            if(j < n - 1)
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);

                rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j, scalars, 0, A,
                                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                                    shiftA + idx2D(j, 0, lda), lda, strideA, scalars + 2, 0, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count, nullptr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);

                rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A, shiftA + idx2D(j + 1, j, lda),
                                    1, strideA, batch_count);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_POTF2_HPP */
