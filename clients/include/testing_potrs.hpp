/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T>
void potrs_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        const rocblas_int nrhs,
                        T dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        T dB,
                        const rocblas_int ldb,
                        const rocblas_stride stB,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potrs(STRIDED, nullptr, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, rocblas_fill_full, n, nrhs, dA, lda, stA,
                                          dB, ldb, stB, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, (T) nullptr, lda, stA, dB, ldb, stB, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, (T) nullptr, ldb, stB, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, uplo, 0, nrhs, (T) nullptr, lda, stA,
                                          (T) nullptr, ldb, stB, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potrs(STRIDED, handle, uplo, n, 0, dA, lda, stA, (T) nullptr, ldb, stB, bc),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_potrs_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_int bc = 1;
    rocblas_fill uplo = rocblas_fill_upper;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());

        // check bad arguments
        potrs_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dB.data(), ldb, stB,
                                    bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());

        // check bad arguments
        potrs_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dB.data(), ldb, stB,
                                    bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void potrs_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nrhs,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_stride stA,
                    Td& dB,
                    const rocblas_int ldb,
                    const rocblas_stride stB,
                    const rocblas_int bc,
                    Th& hA,
                    Th& hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);
        int info;

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to ensure positive definiteness
            for(rocblas_int i = 0; i < n; i++)
                hA[b][i + i * lda] = hA[b][i + i * lda]*sconj(hA[b][i + i * lda])*400;

            // do the Cholesky factorization of matrix A w/ the reference LAPACK routine
            cblas_potrf<T>(uplo, n, hA[b], lda, &info);
        }
    }

    if(GPU)
    {
        // now copy matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void potrs_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nrhs,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_stride stA,
                    Td& dB,
                    const rocblas_int ldb,
                    const rocblas_stride stB,
                    const rocblas_int bc,
                    Th& hA,
                    Th& hB,
                    Th& hBRes,
                    double* max_err)
{
    // input data initialization
    potrs_initData<true, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                        dB.data(), ldb, stB, bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cblas_potrs<T>(uplo, n, nrhs, hA[b], lda, hB[b], ldb);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('I', n, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void potrs_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int nrhs,
                       Td& dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       Td& dB,
                       const rocblas_int ldb,
                       const rocblas_stride stB,
                       const rocblas_int bc,
                       Th& hA,
                       Th& hB,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool perf)
{
    if(!perf)
    {
        potrs_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_potrs<T>(uplo, n, nrhs, hA[b], lda, hB[b], ldb);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    potrs_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        potrs_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        CHECK_ROCBLAS_ERROR(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                            dB.data(), ldb, stB, bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        potrs_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        start = get_time_us_sync(stream);
        rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA, dB.data(), ldb, stB, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_potrs(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr,
                                                  lda, stA, (T* const*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda,
                                                  stA, (T*)nullptr, ldb, stB, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr,
                                              lda, stA, (T* const*)nullptr, ldb, stB, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda, stA,
                                              (T*)nullptr, ldb, stB, bc));

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
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                  stA, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            potrs_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA,
                                       hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing)
            potrs_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA,
                                          hB, &gpu_time_used, &cpu_time_used, hot_calls,
                                          argus.profile, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_potrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                  stA, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            potrs_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA,
                                       hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing)
            potrs_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA,
                                          hB, &gpu_time_used, &cpu_time_used, hot_calls,
                                          argus.profile, argus.perf);
    }

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb", "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb", "strideA", "strideB",
                                       "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, stA, stB, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
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
