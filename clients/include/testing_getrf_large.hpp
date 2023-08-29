/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

// I dont think we need to check for bad arguments, do we ?
template <bool STRIDED, bool GETRF, typename T, typename U>
void getrf_large_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf(STRIDED, GETRF, nullptr, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA, lda, stA,
                                                    dIpiv, stP, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T) nullptr, lda, stA,
                                                dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA, lda, stA,
                                                (U) nullptr, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA, lda, stA, dIpiv,
                                                stP, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, 0, n, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, dinfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, 0, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, dinfo, bc),
                          rocblas_status_success);
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA, lda, stA,
                                                    dIpiv, stP, (U) nullptr, 0),
                              rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getrf_large_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getrf_large_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getrf_large_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
}


// This is our initialization function, we have to modify it in order to initialize multiple Matrices at once.
template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_large_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < m / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp = hA[b][i + j * lda];
                    hA[b][i + j * lda] = hA[b][m - 1 - i + j * lda];
                    hA[b][m - 1 - i + j * lda] = tmp;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes).
                // The algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_large_getError(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          Td& dA,
                          Td& dB,
                          Td& dX,
                          const rocblas_int lda,
                          const rocblas_int ldb,
                          const rocblas_int ldx,
                          const rocblas_stride stA,
                          const rocblas_stride stB,
                          const rocblas_stride stX,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hB,
                          Th& hX,
                          Th& hARes,
                          Th& hBRes,
                          Th& hXRes,
                          size_t size_A,
                          size_t size_B,
                          size_t size_X,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{   

    // Modify the init data method itself to initialize both A and B together.
    // Input data initialization for Matrix A
    getrf_large_initData<true, true, T>(handle, n, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                        hIpiv, hInfo, singular);
    
    // Input data initialization for Matrix B
    getrf_large_initData<true, true, T>(handle, n, nrhs, dB, ldb, stB, dIpiv, stP, dInfo, bc, hB,
                                        hIpiv, hInfo, singular);
    

    //copy from host to device memory for matrice X
    CHECK_HIP_ERROR(dX.transfer_from(hB));  


    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));
    // CHECK_HIP_ERROR(hARes.transfer_from(dA));
    // CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    // CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // Calling the GETRS function with modififed value of dA and result of the operation will be stored in dA.
    // Perhaps it would be better to use CHECK_ROCBLAS_ERROR()
    auto re = rocsolver_getrs(STRIDED,handle, rocblas_operation_none,
                        n, nrhs, dA, lda, stA, dIpiv, stP, dX, ldx, stX, bc);
    if (re != rocblas_status_success)
        std::cout << "getrs failed" << std::endl;
    
    // Resetting the value of dA.
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    // Calling the GEMM function
    T alpha = 1., beta = 0.;
    auto he = rocblas_gemm(STRIDED, handle, rocblas_operation_none, rocblas_operation_none,
                          n, nrhs, n, &alpha, dA, lda, stA, dX, ldx, stX, &beta, dB, ldb, stB, bc);
    if (he != rocblas_status_success) //HIPBLAS_STATUS_SUCCESS)
    std::cout << "gemm failed" << std::endl;

    // Copying the result of the dgemm operation to the host memory
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    
    // Checking for errors.
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {   // Pass the matrices here
        err = norm_error('F', n, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

// Function for perfromance data.
template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_large_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             Uh& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const bool singular)
{
    return;
}


template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getrf_large(Arguments& argus)
{   
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs",n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_int ldx = ldb;
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);
    rocblas_stride stX = argus.get<rocblas_stride>("strideX", ldx * nrhs);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP",n);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    rocblas_stride stXRes = (argus.unit_check || argus.norm_check) ? stX : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // determine sizes using the leading Dimensions, which are typically greater than the rows
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    size_t size_X = size_B;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, stP,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        
        

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T* const*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T*)nullptr, lda,
                                                    stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hX(size_X, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_batch_vector<T> hXRes(size_XRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dX(size_X, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);

        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());

        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());


        // check computations
        if(argus.unit_check || argus.norm_check)
            // Modify the parameters passed.
            getrf_large_getError<STRIDED, GETRF, T>(handle, n, nrhs, dA, dB, dX, lda, ldb, ldx,
                                                     stA, stB, stX, dIpiv, stP, dInfo,
                                                    bc, hA, hB, hX, hARes, hBRes, hXRes, size_A, size_B, size_X, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // This should be returning NAN
        // collect performance data
        if(argus.timing)
            getrf_large_getPerfData<STRIDED, GETRF, T>(
                handle, n, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }


    // These are the allocations for the non batched version
    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hX(size_X, 1, stX, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stARes, bc);
        host_strided_batch_vector<T> hXRes(size_XRes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dX(size_X, 1, stX, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);

        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());


        // check computations
        if(argus.unit_check || argus.norm_check)
            getrf_large_getError<STRIDED, GETRF, T>(handle, n, nrhs, dA, dB, dX, lda, ldb, ldx,
                                                     stA, stB, stX, dIpiv, stP, dInfo,
                                                    bc, hA, hB, hX, hARes, hBRes, hXRes, size_A, size_B, size_X, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // The perf function must return NAN
        // collect performance data
        if(argus.timing)
            getrf_large_getPerfData<STRIDED, GETRF, T>(
                handle, n, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }


    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(n, n));
    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(n, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(n, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(n, n, lda);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
