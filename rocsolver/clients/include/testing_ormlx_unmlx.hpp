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

#include "rocsolver_test.hpp"

#define ERROR_EPS_MULTIPLIER 3000


// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, typename U, int mlq> 
rocblas_status testing_ormlx_unmlx(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;
    rocblas_int lda = argus.lda;
    rocblas_int ldc = argus.ldc;
    int hot_calls = argus.iters;
    char sideC = argus.side_option;
    char transA = argus.transA_option;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    bool invalid = false;
    rocblas_side side;
    rocblas_operation trans;
    rocblas_int size_W, order, size_A;

    if (M < 1 || N < 1 || K < 1 || ldc < M || lda < K || !check_transpose<T>(transA)) 
        invalid = true;

    if (sideC == 'L') {
        side = rocblas_side_left;
        order = M;
        size_A = lda*M;
        size_W = max(N,K);
        if (K > M)
            invalid = true;
    } else if (sideC == 'R') {
        side = rocblas_side_right;
        order = N;
        size_A = lda*N;
        size_W = max(M,K);
        if (K > N)
            invalid = true;
    } else {
        throw runtime_error("Unsoported side option");
    }

    if (transA == 'N') {
        trans = rocblas_operation_none;
    } else if (transA == 'T') {
        trans = rocblas_operation_transpose;
    } else if (transA == 'C') {
        trans = rocblas_operation_conjugate_transpose;
    } else {
        throw runtime_error("Unsupported operation option.");
    } 

    // check invalid size and quick return
    if (invalid) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dC = (T *)dC_managed.get();
        
        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dIpiv = (T *)dIpiv_managed.get();

        if (!dA || !dC || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        if(mlq) {
            return rocsolver_ormlq_unmlq<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
        }
        else { 
            return rocsolver_orml2_unml2<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
        }
    }

    rocblas_int size_C = ldc * N;
    rocblas_int size_P = K;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hC(size_C);
    vector<T> hCr(size_C);
    vector<T> hW(size_W);
    vector<T> hIpiv(size_P);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_C), rocblas_test::device_free};
    T *dC = (T *)dC_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*size_P), rocblas_test::device_free};
    T *dIpiv = (T *)dIpiv_managed.get();

    if (!dA || !dC || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random data 
    rocblas_init<T>(hA.data(), K, order, lda);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < order; j++) {
            if (i == j)
                hA[i + j*lda] += 400;
            else
                hA[i + j*lda] -= 4;
        }
    }
    rocblas_init<T>(hC.data(), M, N, ldc);
    
    //Compute QR factorization
    cblas_gelqf<T>(K, order, hA.data(), lda, hIpiv.data(), hW.data(), size_W); 

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(T) * size_P, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double diff;
    int piverr = 0;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        if(mlq) {
            CHECK_ROCBLAS_ERROR(rocsolver_ormlq_unmlq<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc));
        }
        else {
            CHECK_ROCBLAS_ERROR(rocsolver_orml2_unml2<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc));
        }   
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hCr.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        if(mlq) {
            cblas_ormlq_unmlq<T>(side, trans, M, N, K, hA.data(), lda, hIpiv.data(), hC.data(), ldc, hW.data(), size_W);
        }
        else {
            cblas_orml2_unml2<T>(side, trans, M, N, K, hA.data(), lda, hIpiv.data(), hC.data(), ldc, hW.data());
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        // hCr contains calculated matrix, so error is hC - hCr
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                diff = abs(hC[i + j * ldc]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs(hCr[i + j * ldc] - hC[i + j * ldc]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = max_err_1 / max_val;

        if(argus.unit_check)
            err_res_check<U>(max_err_1, M, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        if(mlq) {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_ormlq_unmlq<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_ormlq_unmlq<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }
        else {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_orml2_unml2<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_orml2_unml2<T>(handle, side, trans, M, N, K, dA, lda, dIpiv, dC, ldc);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "side, trans, M , N , K , lda , ldc , gpu_time(us) , cpu_time(us)";

        if (argus.norm_check)
            cout << " , norm_error_host_ptr";

        cout << endl;
        cout << sideC << " , " << transA << " , " << M << " , " << N << " , " << K << " , " << lda << " , " << ldc << " , " << gpu_time_used << " , "<< cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
