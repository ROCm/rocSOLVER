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
#define GETRF_ERROR_EPS_MULTIPLIER 5000

using namespace std;

template <typename T> 
rocblas_status testing_getrf(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int safe_size = 100; // arbitrarily set to 100
    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if (M < 0 || N < 0 || lda < M) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        if (!dA || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);
        getrf_arg_check(status, M, N);

        return status;
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_piv = min(M, N);    

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<int> hIpiv(size_piv);
    vector<int> hIpivr(size_piv);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * min(M, N)), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
  
    if (!dA || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, M, N, lda);

/*for (int i = M; i < lda; i++) {   //pad untouched area to zero
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
  }*/

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    T error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
    T eps = std::numeric_limits<T>::epsilon();
    T max_err_1 = 0.0;
    T diff;


/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv));

        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hIpivr.data(), dIpiv, sizeof(int) * size_piv, hipMemcpyDeviceToHost));

        //CPU lapack
        cblas_getrf<T>(M, N, hA.data(), lda, hIpiv.data());

        // Error Check
        // check if the pivoting returned is identical
        for (int j = 0; j < size_piv; j++) {
            const int refPiv = hIpiv[j];
            const int gpuPiv = hIpivr[j];
            if (refPiv != gpuPiv) {
                cerr << "reference pivot " << j << ": " << refPiv << " vs " << gpuPiv << endl;
                max_err_1 = 10;
                break;
            }
        }
        // hAr contains calculated decomposition, so error is hA - hAr
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                diff = abs(hAr[i + j * lda] - hA[i + j * lda]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        getrf_err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;
        int hot_calls = 20;

        for(int iter = 0; iter < cold_calls; iter++)
        {
            rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);
        }
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
        {
            rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       

        // CPU cblas
        cpu_time_used = get_time_us();
        cblas_getrf<T>(M, N, hA.data(), lda, hIpiv.data());
        cpu_time_used = get_time_us() - cpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M , N , lda , gpu_time(us) , cpu_time(us)";

        if (argus.norm_check)
            cout << ", norm_error_host_ptr";

        cout << endl;
        cout << M << " , " << N << " , " << lda << " , " << gpu_time_used << " , "<< cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef GETRF_ERROR_EPS_MULTIPLIER
