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

    //TO BE IMPLEMENTED...

    return rocblas_status_success;
}

#undef ERROR_EPS_MULTIPLIER
