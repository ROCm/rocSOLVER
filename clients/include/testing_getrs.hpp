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
#define GETRF_ERROR_EPS_MULTIPLIER 6000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, typename U> rocblas_status testing_getrs(Arguments argus) {

    rocblas_int M = argus.M;
    rocblas_int nhrs = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    char trans = argus.transA_option;
    int hot_calls = argus.iters;

    rocblas_operation transRoc;
    if (trans == 'N') {
        transRoc = rocblas_operation_none;
    } else if (trans == 'T') {
        transRoc = rocblas_operation_transpose;
    } else if (trans == 'C') {
        transRoc = rocblas_operation_conjugate_transpose;
    } else {
        throw runtime_error("Unsupported transpose operation.");
    }

    rocblas_int size_A = lda * M;
    rocblas_int size_B = ldb * nhrs;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if (M < 1 || nhrs < 1 || lda < M || ldb < M) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dB = (T *)dB_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        if (!dA || !dIpiv || !dB) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        return rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<int> hIpiv(M);
    vector<T> hBRes(size_B);

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();

    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B), rocblas_test::device_free};
    T *dB = (T *)dB_managed.get();

    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * M), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
  
    if (!dA || !dIpiv || !dB) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //  initialize full random matrix h and hB 
    rocblas_init<T>(hA.data(), M, M, lda);
    rocblas_init<T>(hB.data(), M, nhrs, ldb);

    // put it into [0, 1]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
          hA[i + j * lda] = (hA[i + j * lda] - 1.0) / 10.0;
        }
    }

    // now make it diagonally dominant
    for (int i = 0; i < M; i++) {
        hA[i + i * lda] *= 420.0;
    }

    // do the LU decomposition of matrix A w/ the reference LAPACK routine
    int retCBLAS;
    cblas_getrf<T>(M, M, hA.data(), lda, hIpiv.data(), &retCBLAS);
    if (retCBLAS != 0) {
        // error encountered - unlucky pick of random numbers? no use to continue
        return rocblas_status_success;
    }

    // now copy pivoting indices and matrices to the GPU
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * M, hipMemcpyHostToDevice));

    double max_err_1 = 0.0, max_val = 0.0, diff;

/* =====================================================================
           ROCSOLVER
    =================================================================== */
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb));
        CHECK_HIP_ERROR(hipMemcpy(hBRes.data(), dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_getrs<T>(trans, M, nhrs, hA.data(), lda, hIpiv.data(), hB.data(), ldb);
        cpu_time_used = get_time_us() - cpu_time_used;


        // Error Check
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < nhrs; j++) {
                diff = abs(hB[i + j * ldb]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs(hBRes[i + j * ldb] - hB[i + j * ldb]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = max_err_1 / max_val;

        getrs_err_res_check<U>(max_err_1, M, nhrs, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_getrs<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, dB, ldb);
        gpu_time_used = get_time_us() - gpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        cout << "trans , M , nhrs , lda , ldb , us [gpu] , us [cpu]";

        if (argus.norm_check)
            cout << ", norm_error_host_ptr";

        cout << endl;

        cout << trans << " , " << M << " , " << nhrs << " , " << lda << " , " << ldb << " , " << gpu_time_used << " , " << cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef GETRF_ERROR_EPS_MULTIPLIER
