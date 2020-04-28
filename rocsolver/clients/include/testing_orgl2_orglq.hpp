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

#define ERROR_EPS_MULTIPLIER 8000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, int orglq> 
rocblas_status testing_orgl2_orglq(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;
    rocblas_int lda = argus.lda;
    int hot_calls = argus.iters;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (M < 1 || N < 1 || N < M || K < 1 || K > M || lda < M) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dIpiv = (T *)dIpiv_managed.get();

        if (!dA || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        if(orglq) {
            return rocsolver_orglq<T>(handle, M, N, K, dA, lda, dIpiv);
        }
        else { 
            return rocsolver_orgl2<T>(handle, M, N, K, dA, lda, dIpiv);
        }
    }

    rocblas_int size_A = lda * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<T> hW(M);
    vector<T> hIpiv(M);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*M), rocblas_test::device_free};
    T *dIpiv = (T *)dIpiv_managed.get();

    if ((size_A > 0 && !dA) || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random data 
    rocblas_init<T>(hA.data(), M, N, lda);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            hA[i + j*lda] = (hA[i + j*lda] - 5.0) / 5.0;    //entries in [-1, 1]
        }
    }
    //Compute QR factorization
    cblas_gelqf<T>(M, N, hA.data(), lda, hIpiv.data(), hW.data(), M); 

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(T) * K, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<T>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double diff;
    int piverr = 0;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        if(orglq) {
            CHECK_ROCBLAS_ERROR(rocsolver_orglq<T>(handle, M, N, K, dA, lda, dIpiv));
        }
        else {
            CHECK_ROCBLAS_ERROR(rocsolver_orgl2<T>(handle, M, N, K, dA, lda, dIpiv));
        }   
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        if(orglq) {
            cblas_orglq<T>(M, N, K, hA.data(), lda, hIpiv.data(), hW.data());
        }
        else {
            cblas_orgl2<T>(M, N, K, hA.data(), lda, hIpiv.data(), hW.data());
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        // hAr contains calculated matrix, so error is hA - hAr
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                diff = abs(hA[i + j * lda]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs(hAr[i + j * lda] - hA[i + j * lda]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = max_err_1 / max_val;

        if(argus.unit_check)
            getf2_err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        if(orglq) {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_orglq<T>(handle, M, N, K, dA, lda, dIpiv);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_orglq<T>(handle, M, N, K, dA, lda, dIpiv);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }
        else {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_orgl2<T>(handle, M, N, K, dA, lda, dIpiv);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_orgl2<T>(handle, M, N, K, dA, lda, dIpiv);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "M , N , K , lda , gpu_time(us) , cpu_time(us)";

        if (argus.norm_check)
            cout << " , norm_error_host_ptr";

        cout << endl;
        cout << M << " , " << N << " , " << K << " , " << lda << " , " << gpu_time_used << " , "<< cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
