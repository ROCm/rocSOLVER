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

#define ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T> 
rocblas_status testing_orgbr(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;
    rocblas_storev storev;
    char storevC = argus.storev;
    rocblas_int lda = argus.lda;
    int hot_calls = argus.iters;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    bool invalid = false;
    bool column = false;
    rocblas_int size_P; 
    rocblas_int size_A; 
    rocblas_int size_W = max(max(M,N),K);
    if (M < 1 || N < 1 || K < 1 || lda < M)
        invalid = true;
    if (storevC == 'C') {
        column = true;
        size_A = lda*max(K,N);
        storev = rocblas_column_wise;
        size_P = min(M,K);;
        if (N > M || N < size_P)
            invalid = true;
    } else if (storevC == 'R') {
        size_A = lda*N;
        storev = rocblas_row_wise;
        size_P = min(N,K);
        if (M > N || M < size_P)
            invalid = true;
    } else {
        throw runtime_error("Unsupported store option.");
    }    

    // check invalid size and quick return
    if (invalid) {
        size_t t;
        t = size_A > 0 ? size_A : 1;
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*t), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        t = size_P > 0 ? size_P : 1;
        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*t), rocblas_test::device_free};
        T *dIpiv = (T *)dIpiv_managed.get();

        if (!dA || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        return rocsolver_orgbr<T>(handle, storev, M, N, K, dA, lda, dIpiv);
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<T> hW(size_W);
    vector<T> hIpiv(size_P);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_P), rocblas_test::device_free};
    T *dIpiv = (T *)dIpiv_managed.get();

    if ((size_A > 0 && !dA) || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random data and compute bi-diagonal form
    vector<T> E(size_P-1);
    vector<T> D(size_P);
    vector<T> P(size_P);
    if (column) {
        rocblas_init<T>(hA.data(), M, K, lda);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                if (i == j)
                    hA[i + j*lda] += 400;
                else
                    hA[i + j*lda] -= 4;
            }
        }
        cblas_gebrd<T>(M, K, hA.data(), lda, D.data(), E.data(), hIpiv.data(), P.data(), hW.data(), size_W); 
    } else {
        rocblas_init<T>(hA.data(), K, N, lda);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                if(i == j)
                    hA[i + j*lda] += 400;
                else
                    hA[i + j*lda] -= 4;
            }
        }
        cblas_gebrd<T>(K, N, hA.data(), lda, D.data(), E.data(), P.data(), hIpiv.data(), hW.data(), size_W); 
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(T) * size_P, hipMemcpyHostToDevice));

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
        CHECK_ROCBLAS_ERROR(rocsolver_orgbr<T>(handle, storev, M, N, K, dA, lda, dIpiv));
        
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_orgbr<T>(storevC, M, N, K, hA.data(), lda, hIpiv.data(), hW.data(), size_W);
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
           err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_orgbr<T>(handle, storev, M, N, K, dA, lda, dIpiv);
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_orgbr<T>(handle, storev, M, N, K, dA, lda, dIpiv);
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        
        // only norm_check return an norm error, unit check won't return anything
        cout << "storev, M , N , K , lda , gpu_time(us) , cpu_time(us)";

        if (argus.norm_check)
            cout << " , norm_error_host_ptr";

        cout << endl;
        cout << storevC << " , " << M << " , " << N << " , " << K << " , " << lda << " , " << gpu_time_used << " , "<< cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
