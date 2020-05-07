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
rocblas_status testing_laswp(Arguments argus) {
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int k1 = argus.k1;
    rocblas_int k2 = argus.k2;
    rocblas_int inc = argus.incx;
    int hot_calls = argus.iters;
    
    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // invalid size and quick return
    if (N < 1 || lda < 1 || !inc || k1 < 1 || k2 < 1 || k2 < k1) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();
    
        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        if (!dA || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }    

        return rocsolver_laswp<T>(handle,N,dA,lda,k1,k2,dIpiv,inc);
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_P = k1 + (k2-k1)*abs(inc);
    
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<int> hIpiv(size_P);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * size_P), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

    if (!dA || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }
    
    //initialize full random matrix hA with all entries in [1, 10
    //for sdimplicity, consider M = lda
    rocblas_init<T>(hA.data(), lda, N, lda);

    //initialize full random vector of pivots in [1, x]
    //for simplicity, consider x = lda as this is the number of rows
    rocblas_init<int>(hIpiv.data(), size_P, 1, 1, lda);
 
    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * size_P, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double max_err_1 = 0.0, diff;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        rocsolver_laswp<T>(handle,N,dA,lda,k1,k2,dIpiv,inc);
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_laswp<T>(N,hA.data(),lda,k1,k2,hIpiv.data(),inc);
        cpu_time_used = get_time_us() - cpu_time_used;

        //++++++++++++ error check ++++++++++++++++
        if (argus.unit_check) {
            unit_check_general(lda,N,lda,hA.data(),hAr.data());  
        } else {
            for (int i = 0; i < lda; i++) {
                for (int j = 0; j < N; j++) {
                    diff = abs(hAr[i + j * lda] - hA[i + j *lda]);
                    max_err_1 = max_err_1 > diff ? max_err_1 : diff;
                }
            }
        }              
    }

    if (argus.timing) {
        int cold_calls = 2;

            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_laswp<T>(handle,N,dA,lda,k1,k2,dIpiv,inc);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_laswp<T>(handle,N,dA,lda,k1,k2,dIpiv,inc);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       

        // only norm_check return an norm error, unit check won't return anything
        cout << "N,lda,k1,k2,inc,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << N << "," << lda << "," << k1 << "," << k2 << "," << inc << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }

    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
