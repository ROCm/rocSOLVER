/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "unit.h"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


// this is max error PER element after the solution
#define GETRF_ERROR_EPS_MULTIPLIER 10000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  


template <typename T>
void testing_getrs_bad_arg()
{
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_operation trans = rocblas_operation_none;

    device_vector<T,0> dA(1);
    device_vector<T,0> dB(1);
    device_vector<rocblas_int,0> dIpiv(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(nullptr, trans, m, nrhs, dA, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, rocblas_operation(-1), m, nrhs, dA, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, nullptr, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, nullptr, dB, ldb),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, nullptr, ldb),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, 0, nrhs, nullptr, lda, nullptr, nullptr, ldb),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, 0, dA, lda, dIpiv, nullptr, ldb),
                          rocblas_status_success);
}


template <typename T, typename U> 
void testing_getrs(Arguments argus) 
{
    /***** 1. get arguments *****/
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int nrhs = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    char transC = argus.transA_option;
    rocblas_operation trans = char2rocblas_operation(transC);

    /***** 2. check non-supported values *****/
    // N/A

    /***** 3. check invalid sizes *****/
    bool invalid_size = (m < 0 || nrhs < 0 || lda < m || ldb < m);
    if (invalid_size) {
        EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, nullptr, lda, nullptr, nullptr, ldb),
                              rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    /***** 4. memory allocations *****/
    size_t size_A = size_t(lda) * m;
    size_t size_B = size_t(ldb) * nrhs;
    size_t size_P = size_t(m);

    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<int> hIpiv(size_P);
    host_vector<T> hBRes(size_B);
    device_vector<T,0> dA(size_A);
    device_vector<T,0> dB(size_B);
    device_vector<rocblas_int,0> dIpiv(size_P);
    if (size_A) CHECK_DEVICE_ALLOCATION(dA.memcheck());
    if (size_B) CHECK_DEVICE_ALLOCATION(dB.memcheck());
    if (size_P) CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());
    
    /***** 5. check quick return *****/
    if (m == 0 || nrhs == 0) {
        CHECK_ROCBLAS_ERROR(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, dB, ldb));

        if (argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    /***** 6. input data initialization *****/
    rocblas_init<T>(hA.data(), m, m, lda);
    rocblas_init<T>(hB.data(), m, nrhs, ldb);

    // put it into [0, 1]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          hA[i + j * lda] = (hA[i + j * lda] - 1.0) / 10.0;
        }
    }

    // now make it diagonally dominant
    for (int i = 0; i < m; i++) {
        hA[i + i * lda] *= 420.0;
    }

    // do the LU decomposition of matrix A w/ the reference LAPACK routine
    int retCBLAS;
    cblas_getrf<T>(m, m, hA, lda, hIpiv, &retCBLAS);
    if (retCBLAS != 0) {
        // error encountered - unlucky pick of random numbers? no use to continue
        return;
    }

    // now copy pivoting indices and matrices to the GPU
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));


    /***** 7. check computations *****/
    double max_err_1 = 0.0, max_val = 0.0, diff;
    double error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();

    if (argus.unit_check || argus.norm_check) {
        // GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, dB, ldb));
        // (TODO) to fix: transfer_from(device) only works with padded vectors!
        //CHECK_HIP_ERROR(hAr.transfer_from(dA));
        CHECK_HIP_ERROR(hipMemcpy(hBRes.data(), dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // CPU lapack
        cblas_getrs<T>(transC, m, nrhs, hA, lda, hIpiv, hB, ldb);

        // Error Check
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < nrhs; j++) {
                diff = std::abs(hB[i + j * ldb]);
                max_val = max_val > diff ? max_val : diff;
                diff = std::abs(hBRes[i + j * ldb] - hB[i + j * ldb]);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = max_err_1 / max_val;

        getrs_err_res_check<U>(max_err_1, m, nrhs, error_eps_multiplier, eps);
    }

    /***** 8. collect performance data *****/
    if (argus.timing) {
        double gpu_time_used, cpu_time_used;

        cpu_time_used = get_time_us();
        cblas_getrs<T>(transC, m, nrhs, hA, lda, hIpiv, hB, ldb);
        cpu_time_used = get_time_us() - cpu_time_used;

        int cold_calls = 2;
        int hot_calls = argus.iters;

        for(int iter = 0; iter < cold_calls; iter++)
            CHECK_ROCBLAS_ERROR(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, dB, ldb));
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, dB, ldb);
        gpu_time_used = get_time_us() - gpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "trans , m , nrhs , lda , ldb , us [gpu] , us [cpu]";

        if (argus.norm_check)
            rocblas_cout << ", norm_error_host_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << transC << " , " << m << " , " << nrhs << " , " << lda << " , " << ldb << " , " << gpu_time_used << " , " << cpu_time_used;

        if (argus.norm_check)
            rocblas_cout << " , " << max_err_1;

        rocblas_cout << std::endl;
    }
}

#undef GETRF_ERROR_EPS_MULTIPLIER
