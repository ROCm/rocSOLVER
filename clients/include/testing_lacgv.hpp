/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T>
void lacgv_checkBadArgs(const rocblas_handle handle, const rocblas_int n, T dA, const rocblas_int inc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lacgv(nullptr, n, dA, inc), rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lacgv(handle, n, (T) nullptr, inc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lacgv(handle, 0, (T) nullptr, inc), rocblas_status_success);
}

template <typename T>
void testing_lacgv_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int inc = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());

    // check bad arguments
    lacgv_checkBadArgs(handle, n, dA.data(), inc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void lacgv_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int inc,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Th>
void lacgv_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int inc,
                    Th& hA,
                    Th& hAr,
                    double* max_err)
{
    // initialize data
    lacgv_initData<true, true, T>(handle, n, dA, inc, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lacgv(handle, n, dA.data(), inc));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_lacgv<T>(n, hA[0], inc);

    // error |hA - hAr| (elements must be identical)
    *max_err = 0;
    double diff;
    for(int j = 0; j < n; j++)
    {
        diff = std::abs(hAr[0][j * abs(inc)] - hA[0][j * abs(inc)]);
        *max_err = diff > *max_err ? diff : *max_err;
    }
}

template <typename T, typename Td, typename Th>
void lacgv_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       Td& dA,
                       const rocblas_int inc,
                       Th& hA,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        lacgv_initData<true, false, T>(handle, n, dA, inc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_lacgv<T>(n, hA[0], inc);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lacgv_initData<true, false, T>(handle, n, dA, inc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lacgv_initData<false, true, T>(handle, n, dA, inc, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_lacgv(handle, n, dA.data(), inc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(int iter = 0; iter < hot_calls; iter++)
    {
        lacgv_initData<false, true, T>(handle, n, dA, inc, hA);

        start = get_time_us_sync(stream);
        rocsolver_lacgv(handle, n, dA.data(), inc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_lacgv(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int inc = argus.get<rocblas_int>("incx");

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(n) * abs(inc);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || !inc);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lacgv(handle, n, (T*)nullptr, inc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lacgv(handle, n, (T*)nullptr, inc));

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
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lacgv(handle, n, dA.data(), inc), rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lacgv_getError<T>(handle, n, dA, inc, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        lacgv_getPerfData<T>(handle, n, dA, inc, hA, &gpu_time_used, &cpu_time_used, hot_calls,
                             argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // no tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 0);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "inc");
            rocsolver_bench_output(n, inc);

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

#define EXTERN_TESTING_LACGV(...) extern template void testing_lacgv<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LACGV, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
