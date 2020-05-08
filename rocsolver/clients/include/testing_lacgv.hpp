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
rocblas_status testing_lacgv(Arguments argus) {
    rocblas_int N = argus.N;
    rocblas_int inc = argus.incx;
    int hot_calls = argus.iters;
    
    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // invalid size and quick return
    if (N < 1 || !inc) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        if (!dA) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }    

        return rocsolver_lacgv<T>(handle,N,dA,inc);
    }

    rocblas_int size_A = inc * N;
    
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();

    if (!dA) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }
    
    //initialize full random matrix hA with all entries in [1, 10
    //for sdimplicity, consider M = lda = inc
    rocblas_init<T>(hA.data(), inc, N, inc);
 
    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double max_err_1 = 0.0, diff;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        rocsolver_lacgv<T>(handle,N,dA,inc);
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_lacgv<T>(N,hA.data(),inc);
        cpu_time_used = get_time_us() - cpu_time_used;

        //++++++++++++ error check ++++++++++++++++
        if (argus.unit_check) {
            unit_check_general(inc,N,inc,hA.data(),hAr.data());  
        } else {
            for (int i = 0; i < N; i++) {
                diff = abs(hAr[i * inc] - hA[i * inc]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }              
    }

    if (argus.timing) {
        int cold_calls = 2;

            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_lacgv<T>(handle,N,dA,inc);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_lacgv<T>(handle,N,dA,inc);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       

        // only norm_check return an norm error, unit check won't return anything
        cout << "N,inc,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << N << "," << inc << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }

    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
