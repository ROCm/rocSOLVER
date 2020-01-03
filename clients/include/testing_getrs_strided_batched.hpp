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

// this is max error PER element after the solution
#define GETRF_ERROR_EPS_MULTIPLIER 5000

using namespace std;

// **** THIS FUNCTION ONLY TESTS NORMNAL USE CASE
//      I.E. WHEN STRIDEA >= LDA*M,
//      STRIDEB >= LDB*NRHS, AND STRIDEP >= M ****

template <typename T> rocblas_status testing_getrs_strided_batched(Arguments argus) {

    rocblas_int M = argus.M;
    rocblas_int nhrs = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int strideA = argus.bsa;
    rocblas_int strideB = argus.bsb;
    rocblas_int strideP = argus.bsp;
    char trans = argus.transA_option;
    int hot_calls = argus.iters;
    rocblas_int batch_count = argus.batch_count;

    rocblas_operation transRoc;
    if (trans == 'N') {
        transRoc = rocblas_operation_none;
    } else if (trans == 'T') {
        transRoc = rocblas_operation_transpose;
    } else {
        throw runtime_error("Unsupported transpose operation.");
    }

    rocblas_int size_A = lda * M;
    rocblas_int size_B = ldb * nhrs;
    rocblas_int size_P = M;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if (batch_count < 1 || M < 1 || nhrs < 1 || lda < M || ldb < M) {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();

        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dB = (T *)dB_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        if (!dA || !dIpiv || !dB) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        return rocsolver_getrs_strided_batched<T>(handle, transRoc, M, nhrs, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, batch_count);
    }

    size_A += strideA * (batch_count -1);
    size_B += strideB * (batch_count -1);
    size_P += strideP * (batch_count -1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<int> hIpiv(size_P);
    vector<T> hBRes(size_B);

    double gpu_time_used, cpu_time_used;
    T error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
    T eps = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();

    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B), rocblas_test::device_free};
    T *dB = (T *)dB_managed.get();

    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * size_P), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
  
    if (!dA || !dIpiv || !dB) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //  initialize full random matrix h and hB 
    for(int b=0; b < batch_count; ++b) {
        rocblas_init<T>((hA.data() + b*strideA), M, M, lda);
        rocblas_init<T>((hB.data() + b*strideB), M, nhrs, ldb);

        // put it into [0, 1]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                (hA.data() + b*strideA)[i + j * lda] = ((hA.data() + b*strideA)[i + j * lda] - 1.0) / 10.0;
            }
        }

        // now make it diagonally dominant
        for (int i = 0; i < M; i++) {
            (hA.data() + b*strideA)[i + i * lda] *= 420.0;
        }
    }


    // do the LU decomposition of matrix A w/ the reference LAPACK routine
    int retCBLAS;
    for(int b=0; b < batch_count; ++b) {
        retCBLAS = 0;
        cblas_getrf<T>(M, M, (hA.data() + b*strideA), lda, (hIpiv.data() + b*strideP), &retCBLAS);
        if (retCBLAS != 0) {
            // error encountered - unlucky pick of random numbers? no use to continue
            return rocblas_status_success;
        }
    }

    // now copy pivoting indices and matrices to the GPU
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * size_P, hipMemcpyHostToDevice));

    double max_err_1 = 0.0, max_val = 0.0, diff, err;

/* =====================================================================
           ROCSOLVER
    =================================================================== */
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_getrs_strided_batched<T>(handle, transRoc, M, nhrs, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hBRes.data(), dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        for(int b=0; b < batch_count; ++b) 
            cblas_getrs<T>(trans, M, nhrs, (hA.data() + b*strideA), lda, (hIpiv.data() + b*strideP), (hB.data() + b*strideB), ldb);
        cpu_time_used = get_time_us() - cpu_time_used;


        // Error Check
        for(int b=0; b < batch_count; ++b) {
            err = 0.0;
            max_val = 0.0;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < nhrs; j++) {
                    diff = fabs((hB.data() + b*strideB)[i + j * ldb]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = (hB.data() + b*strideB)[i + j * ldb];
                    diff = fabs((hBRes.data() + b*strideB)[i + j * ldb] - diff);
                    err = err > diff ? err : diff;
                }
            }
            err = err / max_val;
            max_err_1 = max_err_1 > err ? max_err_1 : err;
        }

        getrs_err_res_check<T>(max_err_1, M, nhrs, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_getrs_strided_batched<T>(handle, transRoc, M, nhrs, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, batch_count);
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_getrs_strided_batched<T>(handle, transRoc, M, nhrs, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, batch_count);
        gpu_time_used = get_time_us() - gpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M , nhrs , lda , strideA , strideP , ldb , strideB , batch_count , us [gpu] , us [cpu]";

        if (argus.norm_check)
            cout << ", norm_error_host_ptr";

        cout << endl;

        cout << M << " , " << nhrs << " , " << lda << " , " << strideA << " , " << strideP << " , " << ldb << " , " << strideB << " , " << batch_count << " , " << gpu_time_used << " , " << cpu_time_used;

        if (argus.norm_check)
            cout << " , " << max_err_1;

        cout << endl;
    }
    
    return rocblas_status_success;
}

#undef GETRF_ERROR_EPS_MULTIPLIER
