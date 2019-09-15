/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_GETF2_H
#define ROCLAPACK_GETF2_H

#include <hip/hip_runtime.h>
#include <rocblas.hpp>

#include "rocsolver.h"

#include "definitions.h"
#include "helpers.h"
#include "ideal_sizes.hpp"

using namespace std;

#define GETF2_INPMINONE 0
#define GETF2_RESSING 1

template <typename T>
__global__ void getf2_check_singularity(T *A, rocblas_int *jp, rocblas_int j,
                                        rocblas_int lda,
                                        T *inpsResGPUInt) {

  (*jp) = j + (*jp); // jp is 1 index, j is zero

  if (A[j * lda + (*jp) - 1] == 0) {
    inpsResGPUInt[GETF2_RESSING] = -j;
    // to not run into NaNs subsequently
    A[j * lda + (*jp) - 1] = static_cast<T>(1e-6);
  }
}

template <typename T>
__global__ void getf2_pivot(rocblas_int n, T *x, rocblas_int incx,
                            rocblas_int j, rocblas_int *jp) {
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if ((j != (*jp) - 1) && (*jp) > 0 && tid < n) {
    T tmp = x[tid * incx + j];
    x[tid * incx + j] = x[tid * incx + (*jp) - 1];
    x[tid * incx + (*jp) - 1] = tmp;
  }
}

template <typename T>
__global__ void getf2_scal(rocblas_int n, const T *alpha, T *x) {
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // bound
  if (tid < n) {
    x[tid] = (x[tid]) / (*alpha);
  }
}

template <typename T>
rocblas_status rocsolver_getf2_template(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, T *A, rocblas_int lda,
                                        rocblas_int *ipiv) {

  if (m == 0 || n == 0) {
    // quick return
    return rocblas_status_success;
  } else if (m < 0) {
    // less than zero dimensions in a matrix?!
    return rocblas_status_invalid_size;
  } else if (n < 0) {
    // less than zero dimensions in a matrix?!
    return rocblas_status_invalid_size;
  } else if (lda < max(1, m)) {
    // mismatch of provided first matrix dimension
    return rocblas_status_invalid_size;
  }

  rocblas_int oneInt = 1;
  T inpsResHost[2];
  inpsResHost[GETF2_INPMINONE] = static_cast<T>(-1);
  inpsResHost[GETF2_RESSING] = static_cast<T>(42);

  // allocate a tiny bit of memory on device to avoid going onto CPU and needing
  // to synchronize.
  T *inpsResGPU;
  hipMalloc(&inpsResGPU, 2 * sizeof(T));
  hipMemcpy(inpsResGPU, &inpsResHost[0], 2 * sizeof(T), hipMemcpyHostToDevice);

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  rocblas_int blocksPivot = (n - 1) / GETF2_BLOCKSIZE + 1;
  dim3 gridPivot(blocksPivot, 1, 1);
  dim3 threads(GETF2_BLOCKSIZE, 1, 1);

  for (rocblas_int j = 0; j < min(m, n); ++j) {

    // find pivot and test for singularity
    rocblas_iamax(handle, m - j, &A[idx2D(j, j, lda)], 1, &ipiv[j]);

    // use Fortran 1-based indexing for the ipiv array as iamax does that as
    // well!
    hipLaunchKernelGGL(getf2_check_singularity<T>, dim3(1), dim3(1), 0, stream,
                       A, &ipiv[j], j, lda, inpsResGPU);

    // Apply the interchange to columns 1:N
    hipLaunchKernelGGL(getf2_pivot<T>, gridPivot, threads, 0, stream, n, A, lda,
                       j, &ipiv[j]);

    // Compute elements J+1:M of J'th column

    rocblas_int blocksScal = (m - j - 2) / GETF2_BLOCKSIZE + 1;

    dim3 gridScal(blocksScal, 1, 1);
    hipLaunchKernelGGL(getf2_scal<T>, gridScal, threads, 0, stream, (m - j - 1),
                       &A[idx2D(j, j, lda)], &A[idx2D(j + 1, j, lda)]);

    if (j < min(m, n) - 1) {
      // update trailing submatrix
      rocblas_ger(handle, m - j - 1, n - j - 1, &inpsResGPU[GETF2_INPMINONE],
                  &A[idx2D(j + 1, j, lda)], oneInt, &A[idx2D(j, j + 1, lda)],
                  lda, &A[idx2D(j + 1, j + 1, lda)], lda);
    }
  }

  // let's see if we encountered any singularity
  hipMemcpy(&inpsResHost[GETF2_RESSING], &inpsResGPU[GETF2_RESSING], sizeof(T),
            hipMemcpyDeviceToHost);
  if (inpsResHost[GETF2_RESSING] <= 0.0) {
    const size_t elem = static_cast<size_t>(fabs(inpsResHost[GETF2_RESSING]));
    cerr << "ERROR: Input matrix has singularity/-ies. Last "
            "occurrence of this in element "
         << elem << endl;
    hipFree(inpsResGPU);
    return rocblas_status_internal_error;
  }

  hipFree(inpsResGPU);

  return rocblas_status_success;
}

#undef GETF2_INPMINONE
#undef GETF2_RESSING

#endif /* ROCLAPACK_GETF2_H */
