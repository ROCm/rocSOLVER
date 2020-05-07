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

// this is max error PER element after the LU
#define ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

// **** THIS FUNCTION ONLY TESTS NORMNAL USE CASE
//      I.E. WHEN STRIDEA >= LDA*N AND STRIDEP >= MIN(M,N) **** 

template <typename T, typename U, int getrf> 
rocblas_status testing_getf2_getrf_batched(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int stridep = argus.bsp;
    rocblas_int batch_count = argus.batch_count;
    int hot_calls = argus.iters;
    rocblas_int safe_size = 100; // arbitrarily set to 100
    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (M < 1 || N < 1 || lda < M || batch_count < 1) {
        T **dA;
        hipMalloc(&dA,sizeof(T*));

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        auto dinfo_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dinfo = (rocblas_int *)dinfo_managed.get();
        
        if (!dA || !dIpiv || !dinfo) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        if(getrf) {
            return rocsolver_getrf_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
        }
        else { 
            return rocsolver_getf2_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
        }
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_piv = min(M, N);    
    size_piv += stridep * (batch_count - 1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA[batch_count];
    vector<T> hAr[batch_count];
    for(int b=0; b < batch_count; ++b) {
        hA[b] = vector<T>(size_A);
        hAr[b] = vector<T>(size_A);
    }
    vector<int> hIpiv(size_piv);
    vector<int> hIpivr(size_piv);
    vector<int> hinfo(batch_count);
    vector<int> hinfor(batch_count);

    T* A[batch_count];
    for(int b=0; b < batch_count; ++b) 
        hipMalloc(&A[b], sizeof(T) * size_A);
    
    T **dA;
    hipMalloc(&dA,sizeof(T*) * batch_count);
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * size_piv), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
    auto dinfo_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * batch_count), rocblas_test::device_free};
    rocblas_int *dinfo = (rocblas_int *)dinfo_managed.get();
  
    if (!dA || (size_A > 0 && !A[batch_count-1]) || (size_piv > 0 && !dIpiv) || !dinfo) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random matrix hA with all entries in [1, 10]
    for(int b=0; b < batch_count; ++b) {
        T* a = hA[b].data();
        rocblas_init<T>(a, M, N, lda);
        for (rocblas_int i = 0; i < M; ++i) {
            for (rocblas_int j = 0; j < N; ++j) {
                if (i == j)
                    a[i+j*lda] += 400;
                else
                    a[i+j*lda] -= 4;
            }
        }
    }

    // copy data from CPU to device
    for(int b=0;b<batch_count;b++) 
        CHECK_HIP_ERROR(hipMemcpy(A[b], hA[b].data(), sizeof(T)*size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, A, sizeof(T*)*batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();
    double max_err_1 = 0.0, max_val;
    double diff, err;
    int piverr = 0;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        if(getrf) {
            CHECK_ROCBLAS_ERROR(rocsolver_getrf_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count));
        }
        else {
            CHECK_ROCBLAS_ERROR(rocsolver_getf2_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count));
        }   
        
        //copy output from device to cpu
        for(int b=0;b<batch_count;b++) 
            CHECK_HIP_ERROR(hipMemcpy(hAr[b].data(), A[b], sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hIpivr.data(), dIpiv, sizeof(int) * size_piv, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hinfor.data(), dinfo, sizeof(int) * batch_count, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        if(getrf) {
            for(int b = 0; b < batch_count; ++b)
                cblas_getrf<T>(M, N, hA[b].data(), lda, (hIpiv.data() + b*stridep), (hinfo.data() + b));
        }
        else {
            for(int b = 0; b < batch_count; ++b)
                cblas_getf2<T>(M, N, hA[b].data(), lda, (hIpiv.data() + b*stridep), (hinfo.data() + b));
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        for(int b = 0; b < batch_count; ++b) {
            err = 0.0;
            max_val = 0.0;
            // check singularity
            if (hinfo[b] != hinfor[b]) {
                piverr = 1;
                cerr << "error singular pivot (batch " << b << "): " << hinfo[b] << " vs " << hinfor[b] << endl; 
            }
            // check if the pivoting returned is identical
            for (int j = 0; j < min(M,N); j++) {
                const int refPiv = (hIpiv.data() + b*stridep)[j];
                const int gpuPiv = (hIpivr.data() + b*stridep)[j];
                if (refPiv != gpuPiv) {
                    piverr = 1;
                    cerr << "error reference pivot " << j << " (batch " << b << "): " << refPiv << " vs " << gpuPiv << endl;
                    break;
                }
            }
            // hAr contains calculated decomposition, so error is hA - hAr
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    diff = abs(hA[b][i + j * lda]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = abs(hAr[b][i + j * lda] - hA[b][i + j * lda]);
                    err = err > diff ? err : diff;
                }
            }
            err = err / max_val;
            max_err_1 = max_err_1 > err ? max_err_1 : err;
        }

        if(argus.unit_check && !piverr)
            getf2_err_res_check<U>(max_err_1, M, N, error_eps_multiplier, eps);
    }
 

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        if(getrf) {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_getrf_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_getrf_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }
        else {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_getf2_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_getf2_batched<T>(handle, M, N, dA, lda, dIpiv, stridep, dinfo, batch_count);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;       
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,striep,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << M << "," << N << "," << lda << "," << stridep << "," << gpu_time_used << "," << cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
   

    for(int b=0;b<batch_count;++b) 
        hipFree(A[b]);
    hipFree(dA);
 
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
