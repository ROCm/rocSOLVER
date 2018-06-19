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

// this is max error PER element after the LU
#define GETRF_ERROR_EPS_MULTIPLIER 500

using namespace std;

template <typename T> rocblas_status testing_getrf(Arguments argus) {

  rocblas_int M = argus.M;
  rocblas_int N = argus.N;
  rocblas_int lda = argus.lda;

  rocblas_int safe_size = 100; // arbitrarily set to 100

  rocblas_int size_A = max(lda, M) * N;

  rocblas_status status;

  std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(
      new rocblas_test::handle_struct);
  rocblas_handle handle = unique_ptr_handle->handle;

  // check here to prevent undefined memory allocation error
  if (M < 0 || N < 0 || lda < M) {
    auto dA_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                           rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    if (!dA) {
      PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
      return rocblas_status_memory_error;
    }

    auto dIpiv_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * min(M, N)),
                           rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

    status = rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);

    getrf_arg_check(status, M, N);

    return status;
  }

  // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
  vector<T> hA(size_A);
  vector<T> AAT(size_A);

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

  //  initialize full random matrix hA with all entries in [1, 10]
  rocblas_init<T>(hA, M, N, lda);

  //  pad untouched area into zero
  for (int i = M; i < lda; i++) {
    for (int j = 0; j < N; j++) {
      hA[i + j * lda] = 0.0;
    }
  }

  // put it into [0, 1]
  for (int i = M; i < lda; i++) {
    for (int j = 0; j < N; j++) {
      hA[i + j * lda] = (hA[i + j * lda] - 1.0) / 10.0;
    }
  }

  // now make it diagonally dominant
  for (int i = 0; i < min(M, N); i++) {
    hA[i + i * lda] *= 420.0;
  }

  // copy data from CPU to device
  CHECK_HIP_ERROR(
      hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

  // allocate space for the pivoting array
  vector<int> hIpiv(min(M, N));
  auto dIpiv_managed =
      rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * min(M, N)),
                         rocblas_test::device_free};
  rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

  T max_err_1 = 0.0;
  if (argus.unit_check || argus.norm_check) {
    // calculate dXorB <- A^(-1) B rocblas_pointer_mode_host
    const rocblas_status retGPU =
        rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);

    CHECK_HIP_ERROR(
        hipMemcpy(AAT.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

    const int retCBLAS = cblas_getrf<T>(M, N, hA.data(), lda, hIpiv.data());

    if (retCBLAS != 0) {
      // error encountered - we expect the same to happen from the GPU!
      if (retGPU == rocblas_status_success) {
        fprintf(stderr, "rocBLAS should fail also but doesn't!");
        return rocblas_status_internal_error;
      }
    } else {
      CHECK_ROCBLAS_ERROR(retGPU);

      // Error Check

      // check if the pivoting returned is identical
      vector<int> hIpivGPU(min(M, N));
      CHECK_HIP_ERROR(hipMemcpy(hIpivGPU.data(), dIpiv, sizeof(int) * min(M, N),
                                hipMemcpyDeviceToHost));
      for (int j = 0; j < min(M, N); j++) {
        const int refPiv = hIpiv[j];
        const int gpuPiv = hIpivGPU[j];
        if (refPiv != gpuPiv) {
          cerr << "reference pivot " << j << ": " << refPiv << " vs " << gpuPiv
               << endl;
          return rocblas_status_internal_error;
        }
      }

      // AAT contains calculated decomposition, so error is hA - AAT
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          AAT[i + j * lda] = abs(AAT[i + j * lda] - hA[i + j * lda]);
        }
      }

      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          max_err_1 =
              max_err_1 > AAT[i + j * lda] ? max_err_1 : AAT[i + j * lda];
        }
      }
      getrf_err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }
  }

  if (argus.timing) {
    // GPU rocBLAS
    gpu_time_used = get_time_us(); // in microseconds

    const rocblas_status retGPU =
        rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);

    gpu_time_used = get_time_us() - gpu_time_used;

    // CPU cblas
    cpu_time_used = get_time_us();

    const int retCBLAS = cblas_getrf<T>(M, N, hA.data(), lda, hIpiv.data());

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
    cout << "M , N , lda , us [gpu] , us [cpu]";

    if (argus.norm_check)
      cout << ",norm_error_host_ptr";

    cout << endl;

    cout << M << " , " << N << " , " << lda << " , " << gpu_time_used << " , "
         << cpu_time_used;

    if (argus.norm_check)
      cout << " , " << max_err_1;

    cout << endl;
  }
  return rocblas_status_success;
}

#undef GETRF_ERROR_EPS_MULTIPLIER
