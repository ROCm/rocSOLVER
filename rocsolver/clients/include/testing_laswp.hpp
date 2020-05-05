/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//#include <cmath> // std::abs
//#include <fstream>
//#include <iostream>
//#include <limits> // std::numeric_limits<T>::epsilon();
//#include <stdlib.h>
//#include <string>
//#include <vector>

//#include "arg_check.h"
//#include "norm.h"
#include "rocblas_test_unique_ptr.hpp"
#include "unit.h"
//#include "utility.h"
//#ifdef GOOGLE_TEST
//#include <gtest/gtest.h>
//#endif

#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"

#define ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

//using namespace std;

template <typename T>
void testing_laswp_bad_arg()
{
    rocblas_local_handle handle;  
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int k1 = 1;
    rocblas_int k2 = 2;
    rocblas_int inc = 1;

/*    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
    if (!dA || !dIpiv) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }*/

    device_vector<T,0> dA(1);
    device_vector<rocblas_int,0> dIpiv(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp<T>(nullptr,n,dA,lda,k1,k2,dIpiv,inc),
                          rocblas_status_invalid_handle); 

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp<T>(handle,n,nullptr,lda,k1,k2,dIpiv,inc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp<T>(handle,n,dA,lda,k1,k2,nullptr,inc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp<T>(handle,0,nullptr,lda,k1,k2,dIpiv,inc),
                          rocblas_status_success);
}

template <typename T> 
void testing_laswp(Arguments argus) 
{
    /***** 1. get arguments *****/
    rocblas_local_handle handle;  
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int k1 = argus.k1;
    rocblas_int k2 = argus.k2;
    rocblas_int inc = argus.incx;

    /***** 2. check non-supported values *****/
    // N/A

    /***** 3. check invalid sizes *****/
    bool invalid_size = (n < 0 || lda < 1 || !inc || k1 < 1 || k2 < 1 || k2 < k1);
    if (invalid_size) {
        EXPECT_ROCBLAS_STATUS(rocsolver_laswp<T>(handle,n,nullptr,lda,k1,k2,nullptr,inc),
                              rocblas_status_invalid_size);

        if (argus.timing) { //in benchmark
            rocblas_cout << "Invalid size arguments..." << std::endl;
            rocblas_cout << "No performance data to collect." << std::endl;
            if (argus.norm_check)
                rocblas_cout << "No computations to verify." << std::endl;
        }

        return;
    }             

    /***** 4. memory allocations *****/
    size_t size_A = size_t(lda) * n;
    size_t size_P = k1 + size_t(k2-k1)*abs(inc);

    host_vector<T> hA(size_A);
    host_vector<T> hAr(size_A);
    host_vector<rocblas_int> hIpiv(size_P); 
//    std::vector<T> hA(size_A);
//    std::vector<T> hAr(size_A);
//    std::vector<int> hIpiv(size_P);
    device_vector<T,0> dA(size_A);
    device_vector<rocblas_int,0> dIpiv(size_P);

    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());
    if (size_A > 0) 
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
    
//    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
//    T *dA = (T *)dA_managed.get();
//    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * size_P), rocblas_test::device_free};
//    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

//    if ((n && !dA) || !dIpiv) {
//        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
    
    /***** 5. check quick return *****/
    if (n == 0) {
        CHECK_ROCBLAS_ERROR(rocsolver_laswp<T>(handle,n,dA,lda,k1,k2,dIpiv,inc));

        if (argus.timing) { //in benchmark
            rocblas_cout << "Quick return..." << std::endl;
            rocblas_cout << "No performance data to collect." << std::endl;
            if (argus.norm_check)
                rocblas_cout << "No computations to verify." << std::endl;
        }
        
        return;
    }
    
    /***** 6. input data initialization *****/
    //initialize full random matrix hA with all entries in [1, 10]
    //for sdimplicity, consider m = lda
    rocblas_init<T>(hA.data(), lda, n, lda);

    //initialize full random vector of pivots in [1, x]
    //for simplicity, consider x = lda as this is the number of rows
    rocblas_init<int>(hIpiv.data(), size_P, 1, 1, lda);
 
    // copy data from CPU to device
//    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
//    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * size_P, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));


    /***** 7. check computations *****/
    double max_err_1 = 0.0, diff;

    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_laswp<T>(handle,n,dA,lda,k1,k2,dIpiv,inc));
        // (TODO) to fix: transfer_from(device) only works with padded vectors!
        //CHECK_HIP_ERROR(hAr.transfer_from(dA));
        CHECK_HIP_ERROR(hipMemcpy(hAr, dA, sizeof(T)*size_A, hipMemcpyDeviceToHost));

        //CPU lapack
//        cblas_laswp<T>(n,hA.data(),lda,k1,k2,hIpiv.data(),inc);
        cblas_laswp<T>(n,hA,lda,k1,k2,hIpiv,inc);

        //++++++++++++ error check ++++++++++++++++
        if (argus.unit_check) {
            unit_check_general(lda,n,lda,hA.data(),hAr.data());  
        } else {
            for (int i = 0; i < lda; i++) {
                for (int j = 0; j < n; j++) {
                    diff = std::abs(hAr[i + j * lda] - hA[i + j *lda]);
                    max_err_1 = max_err_1 > diff ? max_err_1 : diff;
                }
            }
        }              
    }

    /***** 8. collect performance data *****/
    if (argus.timing) {
        double gpu_time_used, cpu_time_used;
        
        cpu_time_used = get_time_us();
        cblas_laswp<T>(n,hA.data(),lda,k1,k2,hIpiv.data(),inc);
        cpu_time_used = get_time_us() - cpu_time_used;
        
        int hot_calls = argus.iters;
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            CHECK_ROCBLAS_ERROR(rocsolver_laswp<T>(handle,n,dA,lda,k1,k2,dIpiv,inc));
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_laswp<T>(handle,n,dA,lda,k1,k2,dIpiv,inc);
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "n,lda,k1,k2,inc,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            rocblas_cout << ",norm_error_host_ptr";

        rocblas_cout << std::endl;
        rocblas_cout << n << "," << lda << "," << k1 << "," << k2 << "," << inc << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            rocblas_cout << "," << max_err_1;

        rocblas_cout << std::endl;
    }
}

#undef ERROR_EPS_MULTIPLIER
