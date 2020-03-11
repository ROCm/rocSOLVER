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
#define ERROR_EPS_MULTIPLIER 5000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

// **** THIS FUNCTION ONLY TESTS NORMNAL USE CASE
//      I.E. WHEN STRIDEA >= LDA*N AND STRIDEP >= MIN(M,N) ****


template <typename T, int gelqf> 
rocblas_status testing_gelq2_gelqf_strided_batched(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int stridep = argus.bsp;
    rocblas_int strideA = argus.bsa;
    rocblas_int batch_count = argus.batch_count;
    int hot_calls = argus.iters;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (M < 1 || N < 1 || lda < M || batch_count < 0) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dIpiv = (T *)dIpiv_managed.get();

        if (!dA || !dIpiv) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        if(gelqf) {
            return rocsolver_gelqf_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
        }
        else { 
            return rocsolver_gelq2_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
        }
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_piv = min(M, N);    
    size_A += strideA * (batch_count - 1);
    size_piv += stridep * (batch_count  - 1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hAr(size_A);
    vector<T> hw(M);
    vector<T> hIpiv(size_piv);
    vector<T> hIpivr(size_piv);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_piv), rocblas_test::device_free};
    T *dIpiv = (T *)dIpiv_managed.get();
  
    if ((size_A > 0 && !dA) || (size_piv > 0 && !dIpiv)) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random matrix hA with all entries in [1, 10]
    for(int b=0; b < batch_count; ++b)
        rocblas_init<T>((hA.data() + b*strideA), M, N, lda);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<T>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double diff, err;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        if(gelqf) {
            CHECK_ROCBLAS_ERROR(rocsolver_gelqf_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count));
        }
        else {
            CHECK_ROCBLAS_ERROR(rocsolver_gelq2_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count));
        }
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hIpivr.data(), dIpiv, sizeof(T) * size_piv, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        if(gelqf) {
            for (int b = 0; b < batch_count; ++b) 
                cblas_gelqf<T>(M, N, (hA.data() + b*strideA), lda, (hIpiv.data() + b*stridep), hw.data(), M);
        }
        else {
            for (int b = 0; b < batch_count; ++b) 
                cblas_gelq2<T>(M, N, (hA.data() + b*strideA), lda, (hIpiv.data() + b*stridep), hw.data());
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        for (int b = 0; b < batch_count; ++b) {
            err = 0.0;
            max_val = 0.0;
            // check if the pivoting returned is identical
            for (int j = 0; j < min(M,N); j++) {
                diff = fabs((hIpiv.data() + b*stridep)[j]);
                max_val = max_val > diff ? max_val : diff;
                diff = (hIpiv.data() + b*stridep)[j];
                diff = fabs((hIpivr.data() + b*stridep)[j] - diff);
                err = err > diff ? err : diff;
            }
            // hAr contains calculated decomposition, so error is hA - hAr
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    diff = fabs((hA.data() + b*strideA)[i + j * lda]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = (hA.data() + b*strideA)[i + j * lda];
                    diff = fabs((hAr.data() + b*strideA)[i + j * lda] - diff);
                    err = err > diff ? err : diff;
                }
            }
            err = err / max_val;
            max_err_1 = max_err_1 > err ? max_err_1 : err;
        }

        if(argus.unit_check)
            getf2_err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        if(gelqf) {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_gelqf_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_gelqf_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;
        }
        else {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_gelq2_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_gelq2_strided_batched<T>(handle, M, N, dA, lda, strideA, dIpiv, stridep, batch_count);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,strideA,strideP,batch_count,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << M << "," << N << "," << lda << "," << strideA << "," << stridep << "," << batch_count << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
