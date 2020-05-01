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
#define ERROR_EPS_MULTIPLIER 8000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, typename U> 
rocblas_status testing_larfg(Arguments argus) {
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    int hot_calls = argus.iters;
    rocblas_int safe_size = 100; // arbitrarily set to 100
    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (N < 1 || incx < 1) {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size), rocblas_test::device_free};
        T *dx = (T *)dx_managed.get();

        auto dalpha_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dalpha = (T *)dalpha_managed.get();

        auto dtau_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dtau = (T *)dtau_managed.get();
        
        if (!dx || !dalpha || !dtau) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        return rocsolver_larfg<T>(handle, N, dalpha, dx, incx, dtau);
    }

    rocblas_int sizex = 1;
    if (N > 1)    
        sizex += (N - 2) * incx;    

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hx(sizex);
    vector<T> hx_r(sizex);
    T htau, htau_r;
    T halpha, halpha_r;

    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * sizex), rocblas_test::device_free};
    T *dx = (T *)dx_managed.get();
    auto dalpha_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T *dalpha = (T *)dalpha_managed.get();
    auto dtau_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T *dtau = (T *)dtau_managed.get();
  
    if ((sizex > 0 && !dx) || !dalpha || !dtau) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random inputs, all entries in [1, 10]
    rocblas_init<T>(hx.data(), 1, N-1, incx);
    rocblas_init<T>(&halpha, 1, 1, 1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizex, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));

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
        CHECK_ROCBLAS_ERROR(rocsolver_larfg<T>(handle, N, dalpha, dx, incx, dtau));
        
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hx_r.data(), dx, sizeof(T) * sizex, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&halpha_r, dalpha, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&htau_r, dtau, sizeof(T), hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_larfg<T>(N, &halpha, hx.data(), incx, &htau);
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        //check v
        for (int i = 0; i < N-1; i++) {
            diff = abs(hx[i * incx]);
            max_val = max_val > diff ? max_val : diff;
            diff = abs(hx_r[i * incx] - hx[i * incx]);
            max_err_1 = max_err_1 > diff ? max_err_1 : diff;
        }
        //check beta
        max_val = max_val > abs(halpha) ? max_val : abs(halpha);
        diff = abs(halpha_r - halpha);
        max_err_1 = max_err_1 > diff ? max_err_1 : diff;
        //check tau
        max_val = max_val > abs(htau) ? max_val : abs(htau);
        diff = abs(htau_r - htau);
        max_err_1 = max_err_1 > diff ? max_err_1 : diff;

        max_err_1 = max_err_1 / max_val;
        
        if(argus.unit_check)
            getf2_err_res_check<U>(max_err_1, 1, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_larfg<T>(handle, N, dalpha, dx, incx, dtau);
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_larfg<T>(handle, N, dalpha, dx, incx, dtau);
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       

        // only norm_check return an norm error, unit check won't return anything
        cout << "N,incx,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << N << "," << incx << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
