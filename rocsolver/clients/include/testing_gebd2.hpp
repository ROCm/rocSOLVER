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

// this is max error PER element
#define ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, typename U> 
rocblas_status testing_gebd2(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    int hot_calls = argus.iters;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (M < 1 || N < 1 || lda < M) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U)), rocblas_test::device_free};
        U *dD = (U *)dD_managed.get();

        auto dE_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U)), rocblas_test::device_free};
        U *dE = (U *)dE_managed.get();

        auto dTauq_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dTauq = (T *)dTauq_managed.get();

        auto dTaup_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dTaup = (T *)dTaup_managed.get();

        if (!dA || !dD || !dE || !dTauq || !dTaup) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        return rocsolver_gebd2(handle, M, N, dA, lda, dD, dE, dTauq, dTaup);
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_D = min(M, N);
    rocblas_int size_E = min(M, N) - 1;
    rocblas_int size_piv = min(M, N);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<U> hD(size_D);
    vector<U> hDr(size_D);
    vector<U> hE(size_E);
    vector<U> hEr(size_E);
    vector<T> hw(max(M, N));
    vector<T> hTauq(size_piv);
    vector<T> hTauqr(size_piv);
    vector<T> hTaup(size_piv);
    vector<T> hTaupr(size_piv);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U) * size_D), rocblas_test::device_free};
    U *dD = (U *)dD_managed.get();
    auto dE_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U) * size_E), rocblas_test::device_free};
    U *dE = (U *)dE_managed.get();
    auto dTauq_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_piv), rocblas_test::device_free};
    T *dTauq = (T *)dTauq_managed.get();
    auto dTaup_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_piv), rocblas_test::device_free};
    T *dTaup = (T *)dTaup_managed.get();
  
    if ((size_A > 0 && !dA) || (size_D > 0 && !dD) || (size_E > 0 && !dE) || (size_piv > 0 && !dTauq) || (size_piv > 0 && !dTaup)) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA.data(), M, N, lda);
    for (rocblas_int i = 0; i < M; ++i) {
        for (rocblas_int j = 0; j < N; ++j) {
            if (i == j)
                hA[i+j*lda] += 400;
            else
                hA[i+j*lda] -= 4;
        }
    }


    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double max_err_d = 0.0, max_val_d = 0.0;
    double diff, diff_d;
    bool   flipped = false;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_gebd2(handle, M, N, dA, lda, dD, dE, dTauq, dTaup));

        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hDr.data(), dD, sizeof(U) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hEr.data(), dE, sizeof(U) * size_E, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hTauqr.data(), dTauq, sizeof(T) * size_piv, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hTaupr.data(), dTaup, sizeof(T) * size_piv, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_gebd2<U,T>(M, N, hA.data(), lda, hD.data(), hE.data(), hTauq.data(), hTaup.data(), hw.data());
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        // check if the pivoting returned is identical
        for (int j = 0; j < min(M, N); j++) {
            diff = abs(hTauq[j]);
            max_val = max_val > diff ? max_val : diff;
            diff = abs(hTauqr[j] - hTauq[j]);
            max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            diff = abs(hTaup[j]);
            max_val = max_val > diff ? max_val : diff;
            diff = abs(hTaupr[j] - hTaup[j]);
            max_err_1 = max_err_1 > diff ? max_err_1 : diff;
        }
        // check if the diagonal elements returned are identical
        for (int j = 0; j < min(M, N); j++) {
            diff = abs(hD[j]);
            max_val = max_val > diff ? max_val : diff;
            max_val_d = max_val_d > diff ? max_val_d : diff;
            diff  =  abs(hDr[j] - hD[j]);
            diff_d = abs(hDr[j] + hD[j]);
            max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            
            if (diff_d < diff)
            {
                flipped = true;
                max_err_d = max_err_d > diff_d ? max_err_d : diff_d;
            }
            else
                max_err_d = max_err_d > diff ? max_err_d : diff;
        }
        // check if the off-diagonal elements returned are identical
        for (int j = 0; j < min(M, N) - 1; j++) {
            diff = abs(hE[j]);
            max_val = max_val > diff ? max_val : diff;
            max_val_d = max_val_d > diff ? max_val_d : diff;
            diff   = abs(hEr[j] - hE[j]);
            diff_d = abs(hEr[j] + hE[j]);
            max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            
            if (diff_d < diff)
            {
                flipped = true;
                max_err_d = max_err_d > diff_d ? max_err_d : diff_d;
            }
            else
                max_err_d = max_err_d > diff ? max_err_d : diff;
        }
        // hAr contains calculated bidiagonal form, so error is hA - hAr
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                diff = abs(hA[i + j * lda]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs(hAr[i + j * lda] - hA[i + j * lda]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = flipped ? max_err_d / max_val_d : max_err_1 / max_val;

        if(argus.unit_check)
        {
            if (flipped)
                cout << "WARNING: Diagonal element has flipped sign; checking diagonal magnitudes only" << endl;
            gebd2_err_res_check<U>(max_err_1, M, N, error_eps_multiplier, eps);
        }
    }
 
    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_gebd2(handle, M, N, dA, lda, dD, dE, dTauq, dTaup);
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_gebd2(handle, M, N, dA, lda, dD, dE, dTauq, dTaup);
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << M << "," << N << "," << lda << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
  
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
