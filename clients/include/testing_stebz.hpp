/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T, typename U>
void stebz_checkBadArgs(const rocblas_handle handle, const rocblas_int n, T dD, T dE, U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(nullptr, n, dD, dE, dInfo), rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, n, (T) nullptr, dE, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, n, dD, (T) nullptr, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, n, dD, dE, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, 0, (T) nullptr, (T) nullptr, dInfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stebz_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;

    // memory allocations
    device_strided_batch_vector<T> dD(1, 1, 1, 1);
    device_strided_batch_vector<T> dE(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    stebz_checkBadArgs(handle, n, dD.data(), dE.data(), dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dD,
                    Td& dE,
                    Ud& dInfo,
                    Th& hD,
                    Th& hE,
                    Uh& hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hD, true);
        rocblas_init<T>(hE, true);

        // scale matrix and add random splits
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 400;
            hE[0][i] -= 5;
        }

        // add fixed splits in the matrix to test split handling
        rocblas_int k = n / 2;
        hE[0][k] = 0;
        hE[0][k - 1] = 0;
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dD,
                    Td& dE,
                    Ud& dInfo,
                    Th& hD,
                    Th& hDRes,
                    Th& hE,
                    Th& hERes,
                    Uh& hInfo,
                    double* max_err)
{
    // input data initialization
    stebz_initData<true, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, n, dD.data(), dE.data(), dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));

    // CPU lapack
    cblas_stebz<T>(n, hD[0], hE[0]);

    // error is ||hD - hDRes|| / ||hD||
    // using frobenius norm
    *max_err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       Td& dD,
                       Td& dE,
                       Ud& dInfo,
                       Th& hD,
                       Th& hE,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool perf)
{
    if(!perf)
    {
        stebz_initData<true, false, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_stebz<T>(n, hD[0], hE[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stebz_initData<true, false, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stebz_initData<false, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, n, dD.data(), dE.data(), dInfo.data()));
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
        stebz_initData<false, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        start = get_time_us_sync(stream);
        rocsolver_stebz(handle, n, dD.data(), dE.data(), dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stebz(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;

    // check invalid sizes
    bool invalid_size = (n < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_stebz(handle, n, (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stebz(handle, n, (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<T> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<T> hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<T> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    device_strided_batch_vector<T> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<T> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, n, dD.data(), dE.data(), dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stebz_getError<T>(handle, n, dD, dE, dInfo, hD, hDRes, hE, hERes, hInfo, &max_error);

    // collect performance data
    if(argus.timing)
        stebz_getPerfData<T>(handle, n, dD, dE, dInfo, hD, hE, hInfo, &gpu_time_used,
                             &cpu_time_used, hot_calls, argus.profile, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n");
            rocsolver_bench_output(n);

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
