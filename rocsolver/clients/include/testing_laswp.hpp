/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T, typename U>
void laswp_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int n,
                        T dA,
                        const rocblas_int lda,
                        const rocblas_int k1,
                        const rocblas_int k2,
                        U dIpiv,
                        const rocblas_int inc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp(nullptr, n, dA, lda, k1, k2, dIpiv, inc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp(handle, n, (T) nullptr, lda, k1, k2, dIpiv, inc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp(handle, n, dA, lda, k1, k2, (U) nullptr, inc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_laswp(handle, 0, (T) nullptr, lda, k1, k2, dIpiv, inc),
                          rocblas_status_success);
}

template <typename T>
void testing_laswp_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int k1 = 1;
    rocblas_int k2 = 2;
    rocblas_int inc = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());

    // check bad arguments
    laswp_checkBadArgs(handle, n, dA.data(), lda, k1, k2, dIpiv.data(), inc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void laswp_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_int k1,
                    const rocblas_int k2,
                    Ud& dIpiv,
                    const rocblas_int inc,
                    Th& hA,
                    Uh& hIpiv)
{
    if(CPU)
    {
        // for simplicity consider number of rows m = lda
        rocblas_init<T>(hA, true);
        rocblas_init<rocblas_int>(hIpiv, true);

        // put indices in range [1, x]
        // for simplicity, consider x = lda as this is the number of rows
        for(rocblas_int i = 0; i < hIpiv.n(); ++i)
            hIpiv[0][i] *= lda / 10;
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void laswp_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_int k1,
                    const rocblas_int k2,
                    Ud& dIpiv,
                    const rocblas_int inc,
                    Th& hA,
                    Th& hAr,
                    Uh& hIpiv,
                    double* max_err)
{
    // initialize data
    laswp_initData<true, true, T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_laswp(handle, n, dA.data(), lda, k1, k2, dIpiv.data(), inc));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_laswp<T>(n, hA[0], lda, k1, k2, hIpiv[0], inc);

    // error |hA - hAr| (elements must be identical)
    *max_err = 0;
    double diff;
    for(int i = 0; i < lda; i++)
    {
        for(int j = 0; j < n; j++)
        {
            diff = std::abs(hAr[0][i + j * lda] - hA[0][i + j * lda]);
            *max_err = diff > *max_err ? diff : *max_err;
        }
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void laswp_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       Td& dA,
                       const rocblas_int lda,
                       const rocblas_int k1,
                       const rocblas_int k2,
                       Ud& dIpiv,
                       const rocblas_int inc,
                       Th& hA,
                       Uh& hIpiv,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    if(!perf)
    {
        laswp_initData<true, false, T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_laswp<T>(n, hA[0], lda, k1, k2, hIpiv[0], inc);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    laswp_initData<true, false, T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        laswp_initData<false, true, T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_laswp(handle, n, dA.data(), lda, k1, k2, dIpiv.data(), inc));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        laswp_initData<false, true, T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv);

        start = get_time_us();
        rocsolver_laswp(handle, n, dA.data(), lda, k1, k2, dIpiv.data(), inc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_laswp(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int k1 = argus.k1;
    rocblas_int k2 = argus.k2;
    rocblas_int inc = argus.incx;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = k1 + size_t(k2 - k1) * abs(inc);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < 1 || !inc || k1 < 1 || k2 < 1 || k2 < k1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_laswp(handle, n, (T*)nullptr, lda, k1, k2, (rocblas_int*)nullptr, inc),
            rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, size_P, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_laswp(handle, n, dA.data(), lda, k1, k2, dIpiv.data(), inc),
                              rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        laswp_getError<T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hAr, hIpiv, &max_error);

    // collect performance data
    if(argus.timing)
        laswp_getPerfData<T>(handle, n, dA, lda, k1, k2, dIpiv, inc, hA, hIpiv, &gpu_time_used,
                             &cpu_time_used, hot_calls, argus.perf);

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
            rocsolver_bench_output("n", "lda", "k1", "k2", "inc");
            rocsolver_bench_output(n, lda, k2, k2, inc);

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
