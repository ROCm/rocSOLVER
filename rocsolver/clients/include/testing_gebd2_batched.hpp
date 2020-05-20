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
rocblas_status testing_gebd2_batched(Arguments argus) {
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int strideP = argus.bsp;
    rocblas_int batch_count = argus.batch_count;
    int hot_calls = argus.iters;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (M < 1 || N < 1 || lda < M || batch_count < 0) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T*)), rocblas_test::device_free};
        T **dA = (T **)dA_managed.get();

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
        
        return rocsolver_gebd2_batched(handle, M, N, dA, lda,
            dD, strideP, dE, strideP, dTauq, strideP, dTaup, strideP, batch_count);
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_D = min(M, N);
    rocblas_int size_E = min(M, N) - 1;
    rocblas_int size_piv = min(M, N);
    size_D += strideP * (batch_count  - 1);
    size_E += strideP * (batch_count  - 1);
    size_piv += strideP * (batch_count  - 1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA[batch_count];
    vector<T> hAr[batch_count];
    for (int b = 0; b < batch_count; ++b)
    {
        hA[b] = vector<T>(size_A);
        hAr[b] = vector<T>(size_A);
    }
    vector<U> hD(size_D);
    vector<U> hDr(size_D);
    vector<U> hE(size_E);
    vector<U> hEr(size_E);
    vector<T> hw(max(M, N));
    vector<T> hTauq(size_piv);
    vector<T> hTauqr(size_piv);
    vector<T> hTaup(size_piv);
    vector<T> hTaupr(size_piv);

    T* A[batch_count];
    for (int b = 0; b < batch_count; ++b)
        hipMalloc(&A[b], sizeof(T) * size_A);

    T **dA;
    hipMalloc(&dA, sizeof(T*) * batch_count);
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U) * size_D), rocblas_test::device_free};
    U *dD = (U *)dD_managed.get();
    auto dE_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(U) * size_E), rocblas_test::device_free};
    U *dE = (U *)dE_managed.get();
    auto dTauq_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_piv), rocblas_test::device_free};
    T *dTauq = (T *)dTauq_managed.get();
    auto dTaup_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_piv), rocblas_test::device_free};
    T *dTaup = (T *)dTaup_managed.get();
  
    if ((size_A > 0 && !dA) || (size_D > 0 && !dD) || (size_E > 0 && !dE) || (size_piv > 0 && !dTauq) || (size_piv > 0 && !dTaup) || !A[batch_count - 1]) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random matrix hA with all entries in [1, 10]
    for(int b = 0; b < batch_count; ++b)
    {
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
    for(int b = 0; b < batch_count; ++b)
        CHECK_HIP_ERROR(hipMemcpy(A[b], hA[b].data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, A, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0, max_val_d = 0.0;
    double diff, diff_d, err, err_d;
    bool   flipped = false;

/* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_gebd2_batched(handle, M, N, dA, lda,
            dD, strideP, dE, strideP, dTauq, strideP, dTaup, strideP, batch_count));

        //copy output from device to cpu
        for(int b = 0; b < batch_count; b++)
            CHECK_HIP_ERROR(hipMemcpy(hAr[b].data(), A[b], sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hDr.data(), dD, sizeof(U) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hEr.data(), dE, sizeof(U) * size_E, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hTauqr.data(), dTauq, sizeof(T) * size_piv, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hTaupr.data(), dTaup, sizeof(T) * size_piv, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        for (int b = 0; b < batch_count; ++b)
        {
            cblas_gebd2<U,T>(M, N, hA[b].data(), lda, (hD.data() + b*strideP), (hE.data() + b*strideP),
                (hTauq.data() + b*strideP), (hTaup.data() + b*strideP), hw.data());
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        for (int b = 0; b < batch_count; ++b)
        {
            err = 0.0;
            max_val = 0.0;
            // check if the pivoting returned is identical
            for (int j = 0; j < min(M, N); j++) {
                diff = abs((hTauq.data() + b*strideP)[j]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs((hTauqr.data() + b*strideP)[j] - (hTauq.data() + b*strideP)[j]);
                err = err > diff ? err : diff;
                diff = abs((hTaup.data() + b*strideP)[j]);
                max_val = max_val > diff ? max_val : diff;
                diff = abs((hTaupr.data() + b*strideP)[j] - (hTaup.data() + b*strideP)[j]);
                err = err > diff ? err : diff;
            }
            // check if the diagonal elements returned are identical
            for (int j = 0; j < min(M, N); j++) {
                diff = abs((hD.data() + b*strideP)[j]);
                max_val = max_val > diff ? max_val : diff;
                max_val_d = max_val_d > diff ? max_val_d : diff;
                diff   = abs((hDr.data() + b*strideP)[j] - (hD.data() + b*strideP)[j]);
                diff_d = abs((hDr.data() + b*strideP)[j] + (hD.data() + b*strideP)[j]);
                err = err > diff ? err : diff;
            
                if (diff_d < diff)
                {
                    flipped = true;
                    err_d = err_d > diff_d ? err_d : diff_d;
                }
                else
                    err_d = err_d > diff ? err_d : diff;
            }
            // check if the off-diagonal elements returned are identical
            for (int j = 0; j < min(M, N) - 1; j++) {
                diff = abs((hE.data() + b*strideP)[j]);
                max_val = max_val > diff ? max_val : diff;
                max_val_d = max_val_d > diff ? max_val_d : diff;
                diff   = abs((hEr.data() + b*strideP)[j] - (hE.data() + b*strideP)[j]);
                diff_d = abs((hEr.data() + b*strideP)[j] + (hE.data() + b*strideP)[j]);
                err = err > diff ? err : diff;
            
                if (diff_d < diff)
                {
                    flipped = true;
                    err_d = err_d > diff_d ? err_d : diff_d;
                }
                else
                    err_d = err_d > diff ? err_d : diff;
            }
            // hAr contains calculated bidiagonal form, so error is hA - hAr
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    diff = abs(hA[b][i + j * lda]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = abs(hAr[b][i + j * lda] - hA[b][i + j * lda]);
                    err = err > diff ? err : diff;
                }
            }
            err = flipped ? err_d / max_val_d : err / max_val;
            max_err_1 = max_err_1 > err ? max_err_1 : err;
        }

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
        {
            rocsolver_gebd2_batched(handle, M, N, dA, lda,
                dD, strideP, dE, strideP, dTauq, strideP, dTaup, strideP, batch_count);
        }
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
        {
            rocsolver_gebd2_batched(handle, M, N, dA, lda,
                dD, strideP, dE, strideP, dTauq, strideP, dTaup, strideP, batch_count);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,strideP,batch_count,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << M << "," << N << "," << lda << "," << strideP << "," << batch_count << "," << gpu_time_used << ","<< cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
  
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
