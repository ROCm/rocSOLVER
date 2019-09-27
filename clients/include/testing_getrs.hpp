/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cmath> // std::abs
#include <fstream>
#include <iostream>
#include <limits> // std::numeric_limits<T>::epsilon();
#include <stdlib.h>
#include <string>
#include <vector>

#include "arg_check.h"
#include "cblas_interface.h"
#include "norm.h"
#include "rocblas_test_unique_ptr.hpp"
#include "rocsolver.hpp"
#include "unit.h"
#include "utility.h"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

// this is max error PER element after the solution
#define GETRF_ERROR_EPS_MULTIPLIER 500

using namespace std;

template <typename T> rocblas_status testing_getrs(Arguments argus) {

  rocblas_int M = argus.M;
  rocblas_int nhrs = argus.N;
  rocblas_int lda = argus.lda;
  rocblas_int ldb = argus.ldb;
  char trans = argus.transA_option;

  rocblas_operation transRoc;
  if (trans == 'N') {
    transRoc = rocblas_operation_none;
  } else if (trans == 'T') {
    transRoc = rocblas_operation_transpose;
  } else {
    throw runtime_error("Unsupported transpose operation.");
  }

  rocblas_int safe_size = 100; // arbitrarily set to 100

  rocblas_int size_A = max(lda, M) * M;
  rocblas_int size_B = max(ldb, M) * nhrs;

  rocblas_status status;

  std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(
      new rocblas_test::handle_struct);
  rocblas_handle handle = unique_ptr_handle->handle;

  // check here to prevent undefined memory allocation error
  if (M < 0 || nhrs < 0 || lda < std::max(1, M) || ldb < std::max(1, M)) {
    auto dA_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                           rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    if (!dA) {
      PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
      return rocblas_status_memory_error;
    }

    auto dB_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                           rocblas_test::device_free};
    T *dB = (T *)dB_managed.get();
    if (!dB) {
      PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
      return rocblas_status_memory_error;
    }

    auto dIpiv_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * M),
                           rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

    status =
        rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);

    getrs_arg_check(status, M, nhrs, lda, ldb);

    return status;
  }

  // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
  vector<T> hA(size_A);
  vector<T> hB(size_B);
  vector<T> hBRes(size_B);

  double gpu_time_used, cpu_time_used;
  T error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
  T eps = std::numeric_limits<T>::epsilon();

  // allocate memory on device
  auto dA_managed =
      rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                         rocblas_test::device_free};
  T *dA = (T *)dA_managed.get();
  if (!dA) {
    PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
    return rocblas_status_memory_error;
  }

  auto dB_managed =
      rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                         rocblas_test::device_free};
  T *dB = (T *)dB_managed.get();
  if (!dB) {
    PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
    return rocblas_status_memory_error;
  }

  //  initialize full random matrix hA, hB with all entries in [1, 10]
  rocblas_init<T>(hA.data(), M, M, lda);
  rocblas_init<T>(hB.data(), M, nhrs, ldb);

  //  pad untouched area into zero
  for (int i = M; i < lda; i++) {
    for (int j = 0; j < M; j++) {
      hA[i + j * lda] = 0.0;
    }
  }
  for (int i = M; i < ldb; i++) {
    for (int j = 0; j < nhrs; j++) {
      hB[i + j * ldb] = 0.0;
    }
  }

  // put it into [0, 1]
  for (int i = M; i < lda; i++) {
    for (int j = 0; j < M; j++) {
      hA[i + j * lda] = (hA[i + j * lda] - 1.0) / 10.0;
    }
  }

  // now make it diagonally dominant
  for (int i = 0; i < M; i++) {
    hA[i + i * lda] *= 420.0;
  }

  // allocate space for the pivoting array
  vector<int> hIpiv(M);
  auto dIpiv_managed = rocblas_unique_ptr{
      rocblas_test::device_malloc(sizeof(int) * M), rocblas_test::device_free};
  rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

  // do the LU decomposition of matrix A w/ the reference LAPACK routine
  int retCBLAS;
  cblas_getrf<T>(M, M, hA.data(), lda, hIpiv.data(), &retCBLAS);
  if (retCBLAS != 0) {
    // error encountered - unlucky pick of random numbers? no use to continue
    return rocblas_status_success;
  }

  // now copy pivoting indices and matrices to the GPU
  CHECK_HIP_ERROR(
      hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * M, hipMemcpyHostToDevice));

  T max_err_1 = 0.0;
  if (argus.unit_check || argus.norm_check) {
    const rocblas_status retGPU =
        rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);

    CHECK_HIP_ERROR(
        hipMemcpy(hBRes.data(), dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

    const int retCBLAS = cblas_getrs<T>(trans, M, nhrs, hA.data(), lda,
                                        hIpiv.data(), hB.data(), ldb);

    if (retCBLAS != 0) {
      // error encountered - we expect the same to happen from the GPU!
      if (retGPU == rocblas_status_success) {
        fprintf(stderr, "rocBLAS should fail also but doesn't!");
        return rocblas_status_internal_error;
      }
    } else {
      CHECK_ROCBLAS_ERROR(retGPU);

      // Error Check

      // hBRes contains calculated decomposition, so error is hBres - hB
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < nhrs; j++) {
          hBRes[i + j * ldb] = abs(hBRes[i + j * ldb] - hB[i + j * ldb]);
        }
      }

      for (int i = 0; i < M; i++) {
        for (int j = 0; j < nhrs; j++) {
          max_err_1 =
              max_err_1 > hBRes[i + j * ldb] ? max_err_1 : hBRes[i + j * ldb];
        }
      }
      getrs_err_res_check<T>(max_err_1, M, nhrs, error_eps_multiplier, eps);
    }
  }

  if (argus.timing) {
    // GPU rocBLAS
    gpu_time_used = get_time_us(); // in microseconds

    const rocblas_status retGPU =
        rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);

    gpu_time_used = get_time_us() - gpu_time_used;

    // CPU cblas
    cpu_time_used = get_time_us();

    const int retCBLAS = cblas_getrs<T>(trans, M, nhrs, hA.data(), lda,
                                        hIpiv.data(), hB.data(), ldb);

    if (retCBLAS != 0) {
      // error encountered - we expect the same to happen from the GPU!
      if (retGPU == rocblas_status_success) {
        fprintf(stderr, "rocBLAS should fail also but doesn't!");
      }
    } else {
      CHECK_ROCBLAS_ERROR(retGPU);
    }

    cpu_time_used = get_time_us() - cpu_time_used;

    // only norm_check return an norm error, unit check won't return anything
    cout << "M , nhrs , lda , ldb , us [gpu] , us [cpu]";

    if (argus.norm_check)
      cout << ", norm_error_host_ptr";

    cout << endl;

    cout << M << " , " << nhrs << " , " << lda << " , " << ldb << " , "
         << gpu_time_used << " , " << cpu_time_used;

    if (argus.norm_check)
      cout << " , " << max_err_1;

    cout << endl;
  }
  return rocblas_status_success;
}

#undef GETRF_ERROR_EPS_MULTIPLIER
