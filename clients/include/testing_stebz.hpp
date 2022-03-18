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
void stebz_checkBadArgs(const rocblas_handle handle,
                        const rocblas_eval_range range,
                        const rocblas_eval_order order,
                        const rocblas_int n,
                        const T vlow,
                        const T vup,
                        const rocblas_int ilow,
                        const rocblas_int iup,
                        const T abstol,
                        U dD,
                        U dE,
                        rocblas_int* dnev,
                        rocblas_int* dnsplit,
                        U dW,
                        rocblas_int* dIB,
                        rocblas_int* dIS,
                        rocblas_int* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(nullptr, range, order, n, vlow, vup, ilow, iup, abstol,
                                          dD, dE, dnev, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, rocblas_eval_range(-1), order, n, vlow, vup, ilow,
                                          iup, abstol, dD, dE, dnev, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, rocblas_eval_order(-1), n, vlow, vup, ilow,
                                          iup, abstol, dD, dE, dnev, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol,
                                          (U) nullptr, dE, dnev, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          (U) nullptr, dnev, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, (rocblas_int*)nullptr, dnsplit, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, dnev, (rocblas_int*)nullptr, dW, dIB, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, dnev, dnsplit, (U) nullptr, dIB, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, dnev, dnsplit, dW, (rocblas_int*)nullptr, dIS, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, dnev, dnsplit, dW, dIB, (rocblas_int*)nullptr, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD,
                                          dE, dnev, dnsplit, dW, dIB, dIS, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, 0, vlow, vup, ilow, iup, abstol,
                                          (U) nullptr, (U) nullptr, dnev, dnsplit, (U) nullptr,
                                          (rocblas_int*)nullptr, (rocblas_int*)nullptr, dinfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stebz_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_eval_range range = rocblas_range_all;
    rocblas_eval_order order = rocblas_order_entire;
    T vlow = 0;
    T vup = 0;
    rocblas_int ilow = 0;
    rocblas_int iup = 0;
    T abstol = 0;

    // memory allocations
    device_strided_batch_vector<T> dD(1, 1, 1, 1);
    device_strided_batch_vector<T> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnsplit(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIB(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIS(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dnsplit.memcheck());
    CHECK_HIP_ERROR(dIB.memcheck());
    CHECK_HIP_ERROR(dIS.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    stebz_checkBadArgs(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD.data(), dE.data(),
                       dnev.data(), dnsplit.data(), dW.data(), dIB.data(), dIS.data(), dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void stebz_initData(const rocblas_handle handle, const rocblas_int n, Td& dD, Td& dE, Th& hD, Th& hE)
{
    if(CPU)
    {
        rocblas_init<T>(hD, true);
        rocblas_init<T>(hE, true);

        // scale matrix and add fixed splits in the matrix to test split handling
        // (scaling ensures that all eigenvalues are in [-30, 30])
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 10;
            hE[0][i] -= 5;
            if(i == n / 4 || i == n / 2 || i == n - 1)
                hE[0][i] = 0;
            if(i == n / 7 || i == n / 5 || i == n / 3)
                hD[0][i] *= -1;
        }
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
                    const rocblas_eval_range range,
                    const rocblas_eval_order order,
                    const rocblas_int n,
                    const T vlow,
                    const T vup,
                    const rocblas_int ilow,
                    const rocblas_int iup,
                    const T abstol,
                    Td& dD,
                    Td& dE,
                    Ud& dnev,
                    Ud& dnsplit,
                    Td& dW,
                    Ud& dIB,
                    Ud& dIS,
                    Ud& dinfo,
                    Th& hD,
                    Th& hE,
                    Uh& hnev,
                    Uh& hnevRes,
                    Uh& hnsplit,
                    Uh& hnsplitRes,
                    Th& hW,
                    Th& hWRes,
                    Uh& hIB,
                    Uh& hIBRes,
                    Uh& hIS,
                    Uh& hISRes,
                    Uh& hinfo,
                    Uh& hinfoRes,
                    double* max_err)
{
    std::vector<T> work(4 * n);
    std::vector<int> iwork(3 * n);

    // input data initialization
    stebz_initData<true, true, T>(handle, n, dD, dE, hD, hE);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol,
                                        dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                        dW.data(), dIB.data(), dIS.data(), dinfo.data()));
    CHECK_HIP_ERROR(hnevRes.transfer_from(dnev));
    CHECK_HIP_ERROR(hnsplitRes.transfer_from(dnsplit));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hIBRes.transfer_from(dIB));
    CHECK_HIP_ERROR(hISRes.transfer_from(dIS));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    double atol = (abstol == 0) ? 2 * get_safemin<T>() : abstol;
    cblas_stebz<T>(range, order, n, vlow, vup, ilow, iup, atol, hD[0], hE[0], hnev[0], hnsplit[0],
                   hW[0], hIB[0], hIS[0], work.data(), iwork.data(), hinfo[0]);

    // check info
    if(hinfo[0][0] != hinfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    // check number of split blocks
    rocblas_int ns = hnsplit[0][0];
    *max_err += std::abs(ns - hnsplitRes[0][0]);

    // check split blocks limits
    for(int k = 0; k < ns; ++k)
        *max_err += std::abs(hIS[0][k] - hISRes[0][k]);

    // if finding eigenvalues succeded, check values
    if(hinfo[0][0] == 0)
    {
        // check number of computed eigenvalues
        rocblas_int nn = hnev[0][0];
        *max_err += std::abs(nn - hnevRes[0][0]);

        // check block indices
        for(int k = 0; k < nn; ++k)
            *max_err += std::abs(hIB[0][k] - hIBRes[0][k]);

        // error is ||hW - hWRes|| / ||hW||
        // using frobenius norm
        double err = norm_error('F', 1, nn, 1, hW[0], hWRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_getPerfData(const rocblas_handle handle,
                       const rocblas_eval_range range,
                       const rocblas_eval_order order,
                       const rocblas_int n,
                       const T vlow,
                       const T vup,
                       const rocblas_int ilow,
                       const rocblas_int iup,
                       const T abstol,
                       Td& dD,
                       Td& dE,
                       Ud& dnev,
                       Ud& dnsplit,
                       Td& dW,
                       Ud& dIB,
                       Ud& dIS,
                       Ud& dinfo,
                       Th& hD,
                       Th& hE,
                       Uh& hnev,
                       Uh& hnsplit,
                       Th& hW,
                       Uh& hIB,
                       Uh& hIS,
                       Uh& hinfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool perf)
{
    if(!perf)
    {
        std::vector<T> work(4 * n);
        std::vector<int> iwork(3 * n);
        // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
        double atol = (abstol == 0) ? 2 * get_safemin<T>() : abstol;

        stebz_initData<true, false, T>(handle, n, dD, dE, hD, hE);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_stebz<T>(range, order, n, vlow, vup, ilow, iup, atol, hD[0], hE[0], hnev[0],
                       hnsplit[0], hW[0], hIB[0], hIS[0], work.data(), iwork.data(), hinfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stebz_initData<true, false, T>(handle, n, dD, dE, hD, hE);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stebz_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol,
                                            dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                            dW.data(), dIB.data(), dIS.data(), dinfo.data()));
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
        stebz_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        start = get_time_us_sync(stream);
        rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD.data(), dE.data(),
                        dnev.data(), dnsplit.data(), dW.data(), dIB.data(), dIS.data(), dinfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stebz(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char rangeC = argus.get<char>("range");
    char orderC = argus.get<char>("order");
    rocblas_int n = argus.get<rocblas_int>("n");
    T vlow = T(argus.get<double>("vlow", 0));
    T vup = T(argus.get<double>("vup", rangeC == 'V' ? 1 : 0));
    rocblas_int ilow = argus.get<rocblas_int>("ilow", rangeC == 'I' ? 1 : 0);
    rocblas_int iup = argus.get<rocblas_int>("iup", rangeC == 'I' ? 1 : 0);
    T abstol = T(argus.get<double>("abstol"));

    rocblas_eval_range range = char2rocblas_eval_range(rangeC);
    rocblas_eval_order order = char2rocblas_eval_order(orderC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_W = n;
    size_t size_IB = n;
    size_t size_IS = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;
    size_t size_IBRes = (argus.unit_check || argus.norm_check) ? size_IB : 0;
    size_t size_ISRes = (argus.unit_check || argus.norm_check) ? size_IS : 0;

    // check invalid sizes
    bool invalid_size = (n < 0) || (range == rocblas_range_value && vlow >= vup)
        || (range == rocblas_range_index && (ilow < 1 || iup < 0))
        || (range == rocblas_range_index && (iup > n || (n > 0 && ilow > iup)));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol, (T*)nullptr,
                            (T*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr,
                            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol,
                                          (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (rocblas_int*)nullptr));

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
    host_strided_batch_vector<T> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, size_WRes, 1);
    host_strided_batch_vector<rocblas_int> hIB(size_IB, 1, size_IB, 1);
    host_strided_batch_vector<rocblas_int> hIBRes(size_IBRes, 1, size_IBRes, 1);
    host_strided_batch_vector<rocblas_int> hIS(size_IS, 1, size_IS, 1);
    host_strided_batch_vector<rocblas_int> hISRes(size_ISRes, 1, size_ISRes, 1);
    host_strided_batch_vector<rocblas_int> hnev(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnevRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnsplit(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnsplitRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<T> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dW(size_W, 1, size_W, 1);
    device_strided_batch_vector<rocblas_int> dIB(size_IB, 1, size_IB, 1);
    device_strided_batch_vector<rocblas_int> dIS(size_IS, 1, size_IS, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnsplit(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);

    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_IB)
        CHECK_HIP_ERROR(dIB.memcheck());
    if(size_IS)
        CHECK_HIP_ERROR(dIS.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dnsplit.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, range, order, n, vlow, vup, ilow, iup, abstol,
                                              dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                              dW.data(), dIB.data(), dIS.data(), dinfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stebz_getError<T>(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD, dE, dnev,
                          dnsplit, dW, dIB, dIS, dinfo, hD, hE, hnev, hnevRes, hnsplit, hnsplitRes,
                          hW, hWRes, hIB, hIBRes, hIS, hISRes, hinfo, hinfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        stebz_getPerfData<T>(handle, range, order, n, vlow, vup, ilow, iup, abstol, dD, dE, dnev,
                             dnsplit, dW, dIB, dIS, dinfo, hD, hE, hnev, hnsplit, hW, hIB, hIS, hinfo,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.perf);

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
            rocsolver_bench_output("range", "order", "n", "vlow", "vup", "ilow", "iup", "abstol");
            rocsolver_bench_output(rangeC, orderC, n, vlow, vup, ilow, iup, abstol);

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
