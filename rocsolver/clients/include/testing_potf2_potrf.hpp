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

#define ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  

using namespace std;

template <typename T, int potrf> 
rocblas_status testing_potf2_potrf(Arguments argus) {
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    char char_uplo = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(char_uplo);
    rocblas_int size_A = lda * N;
    rocblas_status status;
    int hot_calls = argus.iters;
    
    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (N < 1 || lda < N) {

        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();
        auto dinfo_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dinfo = (rocblas_int *)dinfo_managed.get();

        if (!dA || !dinfo) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        if (potrf)
            return rocsolver_potrf<T>(handle, uplo, N, dA, lda, dinfo);
        else
            return rocsolver_potf2<T>(handle, uplo, N, dA, lda, dinfo);
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> AAT(size_A);
    int hinfo, hinfor;

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    auto dinfo_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
    rocblas_int *dinfo = (rocblas_int *)dinfo_managed.get();

    if (!dA || !dinfo) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }
  
    //  We start with full random matrix A. Calculate symmetric AAT = A*A^T.
    //  Make AAT strictly diagonal dominant. A strictly diagonal dominant matrix
    //  is SPD so we can use Cholesky.

    //  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA.data(), N, N, lda);

    // put it into [0, 1]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hA[i + j * lda] = (hA[i + j * lda] - 1.0) / 10.0;
        }
    }

    //  calculate AAT = hA * hA ^ T
    cblas_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, N, N, N,
               (T)1.0, hA.data(), lda, hA.data(), lda, (T)0.0, AAT.data(), lda);

    //  copy AAT into hA, and make it positive-definite
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hA[i + j * lda] = AAT[i + j * lda];
        }
        hA[i + i * lda] += 100;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double max_err_1 = 0.0, max_val = 0.0;
    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<T>::epsilon();
    double diff;
    int pderror = 0, last = N, ii, fi;

    /* =====================================================================
           ROCSOLVER
    =================================================================== */  
    if (argus.unit_check || argus.norm_check) {
        // GPU lapack
        if (potrf) {
            CHECK_ROCBLAS_ERROR(rocsolver_potrf<T>(handle, uplo, N, dA, lda, dinfo));
        } else {
            CHECK_ROCBLAS_ERROR(rocsolver_potf2<T>(handle, uplo, N, dA, lda, dinfo));
        }

        //copy result to cpu
        CHECK_HIP_ERROR(hipMemcpy(AAT.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hinfor, dinfo, sizeof(int), hipMemcpyDeviceToHost));
        
        //CPU lapack
        cpu_time_used = get_time_us();
        if (potrf) {
            cblas_potrf<T>(uplo, N, hA.data(), lda, &hinfo);
        } else {
            cblas_potf2<T>(uplo, N, hA.data(), lda, &hinfo);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
    
        // +++++++++++++ Error Check +++++++++++++++++++++
        // check positive definiteness
        if (hinfo != hinfor) {
            pderror = 1;
            cerr << "Error detecting positive definiteness: " << hinfo << "vs" << hinfor <<endl;
        } else {
            if (hinfo > 0)
                last = hinfo;
            // AAT contains calculated decomposition, so error is hA - AAT
            for (int j = 0; j < last; j++) {
                if (char_uplo == 'U') {
                    ii = 0;
                    fi = j + 1;
                } else {
                    ii = j;
                    fi = last;
                }
                for (int i = ii; i < fi; i++) {
                    diff = fabs(hA[i + j * lda]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = hA[i + j * lda];
                    diff = fabs(AAT[i + j * lda] - diff);
                    max_err_1 = max_err_1 > diff ? max_err_1 : diff;                    
                }
            }
            max_err_1 = max_err_1 / max_val;
        }

        if(argus.unit_check && !pderror)
            potf2_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;
        
        if (potrf) {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_potrf<T>(handle, uplo, N, dA, lda, dinfo);
            gpu_time_used = get_time_us(); // in microseconds
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_potrf<T>(handle, uplo, N, dA, lda, dinfo);
            gpu_time_used = get_time_us() - gpu_time_used;
        } else {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_potf2<T>(handle, uplo, N, dA, lda, dinfo);
            gpu_time_used = get_time_us(); // in microseconds
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_potf2<T>(handle, uplo, N, dA, lda, dinfo);
            gpu_time_used = get_time_us() - gpu_time_used;
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "N , lda , uplo , us [gpu] , us [cpu]";

        if (argus.norm_check)
            cout << " , norm_error_host_ptr";

        cout << endl;
        cout << N << " , " << lda << " , " << char_uplo << " , " << gpu_time_used
             << " , " << cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER 
