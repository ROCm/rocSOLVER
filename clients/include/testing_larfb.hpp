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

#define ERROR_EPS_MULTIPLIER 5000

using namespace std;

template <typename T> 
rocblas_status testing_larfb(Arguments argus) 
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;
    rocblas_int lda = argus.lda;
    rocblas_int ldv = argus.ldv;
    rocblas_int ldt = argus.ldt;
    char sideC = argus.side_option;    
    char directC = argus.direct_option;
    char transC = argus.transH_option;
    int hot_calls = argus.iters;
    
    rocblas_side side;
    rocsolver_direct direct;
    rocblas_operation trans;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check invalid size and quick return
    if (N < 1 || M < 1 || K < 1 || lda < M || ldt < K
        || (sideC == 'L' && ldv < M) || (sideC == 'R' && ldv < N)) {

        auto dV_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dV = (T *)dV_managed.get();
        auto dF_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dF = (T *)dF_managed.get();
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
        T *dA = (T *)dA_managed.get();
        if (!dV || !dF || !dA) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
        
        return rocsolver_larfb<T>(handle, side, trans, direct, M, N, K, dV, ldv, dF, ldt, dA, lda);
    }

    rocblas_int sizeF = ldt * K;
    rocblas_int sizeV = ldv * K;
    rocblas_int sizeA = lda * N;
    rocblas_int ldw, sizeW;
    if (directC == 'F') {
        direct = rocsolver_forward_direction;
    } else if (directC == 'B') {
        direct = rocsolver_backward_direction;
    } else {
        throw runtime_error("Unsupported direct option.");
    }
    if (sideC == 'L') {
        side = rocblas_side_left;
        ldw = N;
    } else if (sideC == 'R') {
        side = rocblas_side_right;
        ldw = M;
    } else {
        throw runtime_error("Unsupported side option.");
    }
    if (transC == 'N') {
        trans = rocblas_operation_none;
    } else if (transC == 'T') {
        trans = rocblas_operation_transpose;
    } else {
        throw runtime_error("Unsupported operation option.");
    }
    sizeW = ldw * K;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hV(sizeV);
    vector<T> hF(sizeF);
    vector<T> hA(sizeA);
    vector<T> hA_r(sizeA);
    vector<T> hW(sizeW);

    auto dV_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*sizeV), rocblas_test::device_free};
    T *dV = (T *)dV_managed.get();
    auto dF_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*sizeF), rocblas_test::device_free};
    T *dF = (T *)dF_managed.get();
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)*sizeA), rocblas_test::device_free};
    T *dA = (T *)dA_managed.get();
    if (!dV || !dF || !dA) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //initialize full random inputs with reals from -1 to 1
    if (sideC == 'L') {
        rocblas_init<T>(hV.data(), M, K, ldv);
        for (int i=0; i<M; ++i) {
            for (int j=0; j<K; ++j) {
                hV[i+j*ldv] = (hV[i+j*ldv] - 5) / 5;
            }
        } 
    } else {
        rocblas_init<T>(hV.data(), N, K, ldv);
        for (int i=0; i<N; ++i) {
            for (int j=0; j<K; ++j) {
                hV[i+j*ldv] = (hV[i+j*ldv] - 5) / 5;
            }
        } 
    }
    rocblas_init<T>(hF.data(), K, K, ldt);
    for (int i=0; i<K; ++i) {
        for (int j=0; j<K; ++j) {
            hF[i+j*ldt] = (hF[i+j*ldt] - 5) / 5;
        }
    } 
    rocblas_init<T>(hA.data(), M, N, lda);
    for (int i=0; i<M; ++i) {
        for (int j=0; j<N; ++j) {
            hA[i+j*lda] = (hA[i+j*lda] - 5) / 5;
        }
    } 
    

/*    printf("\n");
    for (int i=0;i<M;++i) {
        for (int j=0;j<K;++j) {
            printf("%2.15f ",hV[i+j*ldv]);
        }
        printf("\n");    
    }
    printf("\n");
    for (int i=0;i<K;++i) {
        for (int j=0;j<K;++j) {
            printf("%2.15f ",hF[i+j*ldt]);
        }
        printf("\n");    
    }
    printf("\n");
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            printf("%2.15f ",hA[i+j*lda]);
        }
        printf("\n");    
    }*/

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dV, hV.data(), sizeof(T) * sizeV, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dF, hF.data(), sizeof(T) * sizeF, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * sizeA, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<T>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double diff;

/* =====================================================================
           ROCSOLVER
    =================================================================== */

    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_larfb<T>(handle, side, trans, direct, M, N, K, dV, ldv, dF, ldt, dA, lda));

        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hA_r.data(), dA, sizeof(T) * sizeA, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        cblas_larfb<T>(sideC,transC,directC,'C',M,N,K,hV.data(),ldv,hF.data(),ldt,hA.data(),lda,hW.data(),ldw);
        cpu_time_used = get_time_us() - cpu_time_used;

/*    printf("\n");
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            printf("%2.15f ",hA[i+j*lda]);
        }
        printf("\n");    
    }
    printf("\n");
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            printf("%2.15f ",hA_r[i+j*lda]);
        }
        printf("\n");    
    }*/

        // +++++++++ Error Check +++++++++++++
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                diff = fabs(hA[i + j * lda]);
                max_val = max_val > diff ? max_val : diff;
                diff = hA[i + j * lda];
                diff = fabs(hA_r[i + j * lda] - diff);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }

        max_err_1 = max_err_1 / max_val;

        if(argus.unit_check)
            getf2_err_res_check<T>(max_err_1, M, N, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_larfb<T>(handle, side, trans, direct, M, N, K, dV, ldv, dF, ldt, dA, lda);
        gpu_time_used = get_time_us();
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_larfb<T>(handle, side, trans, direct, M, N, K, dV, ldv, dF, ldt, dA, lda);
        gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;

        // only norm_check return an norm error, unit check won't return anything
        cout << "side,trans,direct,M,N,K,ldv,ldt,lda,gpu_time(us),cpu_time(us)";

        if (argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << sideC << "," << transC << "," << directC << "," << M << "," << N << "," << K << "," << ldv << "," << ldt << "," << lda << "," << gpu_time_used << "," << cpu_time_used;

        if (argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
   
    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
