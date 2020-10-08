/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
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
                       const bool perf)
{
    if(!perf)
    {
        lacgv_initData<true, false, T>(handle, n, dA, inc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_lacgv<T>(n, hA[0], inc);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    lacgv_initData<true, false, T>(handle, n, dA, inc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lacgv_initData<false, true, T>(handle, n, dA, inc, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_lacgv(handle, n, dA.data(), inc));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        lacgv_initData<false, true, T>(handle, n, dA, inc, hA);

        start = get_time_us();
        rocsolver_lacgv(handle, n, dA.data(), inc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_lacgv(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int inc = argus.incx;
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
            ROCSOLVER_BENCH_INFORM(1);

        return;
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
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lacgv_getError<T>(handle, n, dA, inc, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        lacgv_getPerfData<T>(handle, n, dA, inc, hA, &gpu_time_used, &cpu_time_used, hot_calls,
                             argus.perf);

    // validate results for rocsolver-test
    // no tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, 0);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("n", "inc");
            rocsolver_bench_output(n, inc);

            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Results:\n";
            rocblas_cout << "============================================\n";
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
            rocblas_cout << std::endl;
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }
}
