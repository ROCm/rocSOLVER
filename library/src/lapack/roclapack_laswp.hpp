/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_LASWP_HPP
#define ROCLAPACK_LASWP_HPP

#include "ideal_sizes.hpp"
#include <hip/hip_runtime.h>

using namespace std;

template <typename T>
__global__ void laswp_external(const rocblas_int n, T *a, const rocblas_int lda,
                               const rocblas_int exch1,
                               const rocblas_int *exch2) {

  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tid < n) {
    const T orig = a[exch1 + lda * tid];
    a[exch1 + lda * tid] = a[(*exch2 - 1) + lda * tid];
    a[(*exch2 - 1) + lda * tid] = orig;
  }
}

/**
 *  LASWP performs a series of row interchanges on the matrix A.
 *  One row interchange is initiated for each of rows K1 through K2 of A.
 *
 *  XXX Not a very efficient implementation currently.
 *
 *  Arguments
 *  =========
 *
 *  handle  (input) rocblas_handle
 *
 *  n       (input) rocblas_int
 *          The number of columns of the matrix A.
 *
 *  A       (input/output) T matrix, dimension (LDA,N)
 *          On entry, the matrix of column dimension N to which the row
 *          interchanges will be applied.
 *          On exit, the permuted matrix.
 *
 *  lda     (input) rocblas_int
 *          The leading dimension of the array A.
 *
 *  k1      (input) rocblas_int
 *          The first element of IPIV for which a row interchange will
 *          be done.
 *
 *  k2      (input) rocblas_int
 *          The last element of IPIV for which a row interchange will
 *          be done.
 *
 *  ipiv    (input) rocblas_int array, dimension (k2*abs(incx))
 *          The vector of pivot indices.  Only the elements in positions
 *          k1 through k2 of IPIV are accessed.
 *          IPIV(K) = L implies rows K and L are to be interchanged.
 *          Note: this implementation requires that the ipiv array must not
 *          contain concurrent swaps. I.e., ipiv(k) = l, ipiv(k+x) = l, or
 *          ipiv(k) = l and ipiv(k+x) = k. Calling code is required to sanitize
 *          for this before calling.
 *          Also, we assume this to be 1-indexed as in the reference LAPACK.
 *
 *  incx    (input) rocblas_int
 *          The increment between successive values of IPIV.  If IPIV
 *          is negative, the pivots are applied in reverse order.
 *
 */
template <typename T>
void roclapack_laswp_template(rocblas_handle handle, rocblas_int n, T *A,
                              rocblas_int lda, rocblas_int k1, rocblas_int k2,
                              const rocblas_int *ipiv, rocblas_int incx) {

  if (n == 0) {
    // quick return
    return;
  }

  rocblas_int start, end;
  if (incx < 0) {
    start = k2 - 1;
    end = k1 - 1;
  } else {
    start = k1;
    end = k2;
  }

  // XXX there is a lot to improve here...

  /*
   * to avoid kernel launch overhead and assuming that the number of actual
   * pivots << (end-start), we need to figure out whether the on-GPU ipiv
   * array and our running index are the same. For that: transfer the content
   * of ipiv to the host and check there. this causes the current limitation
   * that incx == 1
   */
  if (incx != 1 && incx != -1) {
    throw runtime_error("roclapack_laswp increment must be one.");
  }

  vector<int> ipivHost(abs(end - start));
  size_t startCpy = (incx < 0) ? end : start;
  hipMemcpy(ipivHost.data(), &ipiv[startCpy],
            sizeof(rocblas_int) * abs(end - start), hipMemcpyDeviceToHost);

  rocblas_int blocksPivot = (n - 1) / LASWP_BLOCKSIZE + 1;
  dim3 gridPivot(blocksPivot, 1, 1);
  dim3 threads(LASWP_BLOCKSIZE, 1, 1);

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  for (rocblas_int i = start; i != end; i += incx) {

    if (ipivHost[i - startCpy] == i + 1)
      continue;

    hipLaunchKernelGGL(laswp_external<T>, gridPivot, threads, 0, stream, n, A,
                       lda, i, &ipiv[i]);
  }
}

#endif /* ROCLAPACK_LASWP_HPP */
