/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_POTF2_HPP
#define ROCLAPACK_POTF2_HPP

#include <hip/hip_runtime.h>
#include <rocblas.hpp>

#include "rocsolver.h"

#include "definitions.h"
#include "helpers.h"

using namespace std;

#define POTF2_INPONE 0
#define POTF2_INPMINONE 1
#define POTF2_RESPOSDEF 2
#define POTF2_RESDOT 3
#define POTF2_RESINVDOT 4

template <typename T> __global__ void sqrtDiagFirst(T *a, size_t loc, T *res) {
  const T t = a[loc];
  if (t <= 0.0) {
    res[POTF2_RESPOSDEF] = -loc;
  } // error for non-positive definiteness
  a[loc] = sqrt(t);
  res[POTF2_RESINVDOT] = 1 / a[loc];
}

template <typename T> __global__ void sqrtDiagOnward(T *a, size_t loc, T *res) {
  const T t = a[loc] - res[POTF2_RESDOT];
  if (t <= 0.0) {
    res[POTF2_RESPOSDEF] = -loc;
  } // error for non-positive definiteness
  a[loc] = sqrt(t);
  res[POTF2_RESINVDOT] = 1 / a[loc];
}

template <typename T>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n, T *a,
                                        rocblas_int lda) {

  if (n == 0) {
    // quick return
    return rocblas_status_success;
  } else if (n < 0) {
    // less than zero dimensions in a matrix?!
    return rocblas_status_invalid_size;
  } else if (lda < max(1, n)) {
    // mismatch of provided first matrix dimension
    return rocblas_status_invalid_size;
  }

  rocblas_int oneInt = 1;
  T inpsResHost[5];
  inpsResHost[POTF2_INPONE] = static_cast<T>(1);
  inpsResHost[POTF2_INPMINONE] = static_cast<T>(-1);
  inpsResHost[POTF2_RESPOSDEF] = static_cast<T>(1);

  // allocate a tiny bit of memory on device to avoid going onto CPU and needing
  // to synchronize.
  T *inpsResGPU;
  hipMalloc(&inpsResGPU, 5 * sizeof(T));
  hipMemcpy(inpsResGPU, &inpsResHost[0], 5 * sizeof(T), hipMemcpyHostToDevice);

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // in order to get the indices right, we check what the fill mode is
  if (uplo == rocblas_fill_upper) {

    // Compute the Cholesky factorization A = U'*U.

    for (rocblas_int j = 0; j < n; ++j) {
      // Compute U(J,J) and test for non-positive-definiteness.
      if (j > 0) {
        rocblas_dot<T>(handle, j, &a[idx2D(0, j, lda)], oneInt,
                       &a[idx2D(0, j, lda)], oneInt, &inpsResGPU[POTF2_RESDOT]);
        hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(1), dim3(1), 0, stream, a,
                           idx2D(j, j, lda), inpsResGPU);
      } else {
        hipLaunchKernelGGL(sqrtDiagFirst<T>, dim3(1), dim3(1), 0, stream, a,
                           idx2D(j, j, lda), inpsResGPU);
      }

      // Compute elements J+1:N of row J

      if (j < n - 1) {
        rocblas_gemv<T>(handle, rocblas_operation_transpose, j, n - j - 1,
                        &(inpsResGPU[POTF2_INPMINONE]),
                        &a[idx2D(0, j + 1, lda)], lda, &a[idx2D(0, j, lda)],
                        oneInt, &(inpsResGPU[POTF2_INPONE]),
                        &a[idx2D(j, j + 1, lda)], lda);
        rocblas_scal<T>(handle, n - j - 1, &inpsResGPU[POTF2_RESINVDOT],
                        &a[idx2D(j, j + 1, lda)], lda);
      }
    }
  } else {

    // Compute the Cholesky factorization A = L'*L.

    for (rocblas_int j = 0; j < n; ++j) {
      // Compute L(J,J) and test for non-positive-definiteness.
      if (j > 0) {
        rocblas_dot<T>(handle, j, &a[idx2D(j, 0, lda)], lda,
                       &a[idx2D(j, 0, lda)], lda, &inpsResGPU[POTF2_RESDOT]);
        hipLaunchKernelGGL(sqrtDiagOnward<T>, dim3(1), dim3(1), 0, stream, a,
                           idx2D(j, j, lda), inpsResGPU);
      } else {
        hipLaunchKernelGGL(sqrtDiagFirst<T>, dim3(1), dim3(1), 0, stream, a,
                           idx2D(j, j, lda), inpsResGPU);
      }

      // Compute elements J+1:N of row J

      if (j < n - 1) {
        rocblas_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                        &(inpsResGPU[POTF2_INPMINONE]),
                        &a[idx2D(j + 1, 0, lda)], lda, &a[idx2D(j, 0, lda)],
                        lda, &(inpsResGPU[POTF2_INPONE]),
                        &a[idx2D(j + 1, j, lda)], oneInt);
        rocblas_scal<T>(handle, n - j - 1, &inpsResGPU[POTF2_RESINVDOT],
                        &a[idx2D(j + 1, j, lda)], oneInt);
      }
    }
  }

  // get the error code using memcpy and return internal error if there is one
  hipMemcpy(&inpsResHost[POTF2_RESPOSDEF], &inpsResGPU[POTF2_RESPOSDEF],
            sizeof(T), hipMemcpyDeviceToHost);
  if (inpsResHost[POTF2_RESPOSDEF] <= 0.0) {
    const size_t elem = static_cast<size_t>(fabs(inpsResHost[POTF2_RESPOSDEF]));
    cerr << "ERROR: Input matrix not strictly positive definite. Last "
            "occurrence of this in element "
         << elem << endl;
    hipFree(inpsResGPU);
    return rocblas_status_internal_error;
  }

  hipFree(inpsResGPU);

  return rocblas_status_success;
}

#undef POTF2_INPONE
#undef POTF2_INPMINONE
#undef POTF2_RESPOSDEF
#undef POTF2_RESDOT
#undef POTF2_RESINVDOT

#endif /* ROCLAPACK_POTF2_HPP */
