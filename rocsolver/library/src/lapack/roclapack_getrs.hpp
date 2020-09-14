/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRS_HPP
#define ROCLAPACK_GETRS_HPP

#include "../auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T>
rocblas_status rocsolver_getrs_argCheck(
    const rocblas_operation trans, const rocblas_int n, const rocblas_int nrhs,
    const rocblas_int lda, const rocblas_int ldb, T A, T B,
    const rocblas_int *ipiv, const rocblas_int batch_count = 1) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  if (trans != rocblas_operation_none && trans != rocblas_operation_transpose &&
      trans != rocblas_operation_conjugate_transpose)
    return rocblas_status_invalid_value;

  // 2. invalid size
  if (n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((n && !A) || (n && !ipiv) || (nrhs * n && !B))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_getrs_getMemorySize(const rocblas_int n, const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t *size_1, size_t *size_2,
                                   size_t *size_3, size_t *size_4) {
  rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, n, nrhs, batch_count,
                                   size_1, size_2, size_3, size_4);
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocsolver_getrs_template(
    rocblas_handle handle, const rocblas_operation trans, const rocblas_int n,
    const rocblas_int nrhs, U A, const rocblas_int shiftA,
    const rocblas_int lda, const rocblas_stride strideA,
    const rocblas_int *ipiv, const rocblas_stride strideP, U B,
    const rocblas_int shiftB, const rocblas_int ldb,
    const rocblas_stride strideB, const rocblas_int batch_count, void *x_temp,
    void *x_temp_arr, void *invA, void *invA_arr, bool optim_mem) {
  // quick return
  if (n == 0 || nrhs == 0 || batch_count == 0) {
    return rocblas_status_success;
  }

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // everything must be executed with scalars on the host
  rocblas_pointer_mode old_mode;
  rocblas_get_pointer_mode(handle, &old_mode);
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

  //// **** THIS SYNCHRONIZATION WILL BE REQUIRED UNTIL
  ////      TRSM_BATCH FUNCTIONALITY IS ENABLED. ****
  //#ifdef batched
  //  T *AA[batch_count];
  //  T *BB[batch_count];
  //  hipMemcpy(AA, A, batch_count * sizeof(T *), hipMemcpyDeviceToHost);
  //  hipMemcpy(BB, B, batch_count * sizeof(T *), hipMemcpyDeviceToHost);
  //#else
  //  T *AA = A;
  //  T *BB = B;
  //#endif

  // constants to use when calling rocablas functions
  T one = 1; // constant 1 in host

  //  T *Ap, *Bp;

  //  // **** TRSM_BATCH IS EXECUTED IN A FOR-LOOP UNTIL
  //  //      FUNCITONALITY IS ENABLED. ****

  if (trans == rocblas_operation_none) {

    // first apply row interchanges to the right hand sides
    rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n,
                                ipiv, 0, strideP, 1, batch_count);

    //     for (int b = 0; b < batch_count; ++b) {
    //      Ap = load_ptr_batch<T>(AA, b, shiftA, strideA);
    //      Bp = load_ptr_batch<T>(BB, b, shiftB, strideB);

    // solve L*X = B, overwriting B with X
    rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                 trans, rocblas_diagonal_unit, n, nrhs, &one, A,
                                 shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                 batch_count, optim_mem, x_temp, x_temp_arr,
                                 invA, invA_arr);

    // solve U*X = B, overwriting B with X
    rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper,
                                 trans, rocblas_diagonal_non_unit, n, nrhs,
                                 &one, A, shiftA, lda, strideA, B, shiftB, ldb,
                                 strideB, batch_count, optim_mem, x_temp,
                                 x_temp_arr, invA, invA_arr);
    //    }

  } else {

    //    for (int b = 0; b < batch_count; ++b) {
    //      Ap = load_ptr_batch<T>(AA, b, shiftA, strideA);
    //      Bp = load_ptr_batch<T>(BB, b, shiftB, strideB);

    // solve U**T *X = B or U**H *X = B, overwriting B with X
    rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper,
                                 trans, rocblas_diagonal_non_unit, n, nrhs,
                                 &one, A, shiftA, lda, strideA, B, shiftB, ldb,
                                 strideB, batch_count, optim_mem, x_temp,
                                 x_temp_arr, invA, invA_arr);

    // solve L**T *X = B, or L**H *X = B overwriting B with X
    rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                 trans, rocblas_diagonal_unit, n, nrhs, &one, A,
                                 shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                 batch_count, optim_mem, x_temp, x_temp_arr,
                                 invA, invA_arr);
    //    }

    // then apply row interchanges to the solution vectors
    rocsolver_laswp_template<T>(handle, nrhs, B, shiftB, ldb, strideB, 1, n,
                                ipiv, 0, strideP, -1, batch_count);
  }

  rocblas_set_pointer_mode(handle, old_mode);
  return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRS_HPP */
