/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
void csrrf_analysis_checkBadArgs(rocblas_handle handle,
                                 const rocblas_int n,
                                 const rocblas_int nnzM,
                                 rocblas_int* ptrM,
                                 rocblas_int* indM,
                                 T valM,
                                 const rocblas_int nnzT,
                                 rocblas_int* ptrT,
                                 rocblas_int* indT,
                                 T valT,
                                 rocblas_int* pivP,
                                 rocblas_int* pivQ,
                                 rocsolver_rfinfo rfinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(nullptr, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, (rocblas_int*)nullptr, indM,
                                                   valM, nnzT, ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, (rocblas_int*)nullptr,
                                                   valM, nnzT, ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, (T) nullptr, nnzT,
                                                   ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT,
                                                   (rocblas_int*)nullptr, indT, valT, pivP, pivQ,
                                                   rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   (rocblas_int*)nullptr, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, (T) nullptr, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, valT, (rocblas_int*)nullptr, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, valT, pivP, (rocblas_int*)nullptr, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, valT, pivP, pivQ, nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, 0, nnzM, ptrM, indM, valM, nnzT, ptrT,
                                                   indT, valT, (rocblas_int*)nullptr,
                                                   (rocblas_int*)nullptr, rfinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_analysis(handle, 0, 0, ptrM, (rocblas_int*)nullptr, (T) nullptr, nnzT, ptrT,
                                 indT, valT, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_analysis(handle, 0, nnzM, ptrM, indM, valM, 0, ptrT, (rocblas_int*)nullptr,
                                 (T) nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    // N/A
}

template <typename T>
void testing_csrrf_analysis_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = 1;
    rocblas_int nnzM = 1;
    rocblas_int nnzT = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> ptrM(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indM(1, 1, 1, 1);
    device_strided_batch_vector<T> valM(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T> valT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivP(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivQ(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrM.memcheck());
    CHECK_HIP_ERROR(indM.memcheck());
    CHECK_HIP_ERROR(valM.memcheck());
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(pivP.memcheck());
    CHECK_HIP_ERROR(pivQ.memcheck());

    // check bad arguments
    csrrf_analysis_checkBadArgs(handle, n, nnzM, ptrM.data(), indM.data(), valM.data(), nnzT,
                                ptrT.data(), indT.data(), valT.data(), pivP.data(), pivQ.data(),
                                rfinfo);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_analysis_initData(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nnzM,
                             Ud& dptrM,
                             Ud& dindM,
                             Td& dvalM,
                             const rocblas_int nnzT,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Ud& dpivP,
                             Ud& dpivQ,
                             Uh& hptrM,
                             Uh& hindM,
                             Th& hvalM,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ)
{
    if(CPU)
    {
    }

    if(GPU)
    {
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_analysis_getError(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nnzM,
                             Ud& dptrM,
                             Ud& dindM,
                             Td& dvalM,
                             const rocblas_int nnzT,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Ud& dpivP,
                             Ud& dpivQ,
                             rocsolver_rfinfo rfinfo,
                             Uh& hptrM,
                             Uh& hindM,
                             Th& hvalM,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             double* max_err)
{
    // input data initialization
    csrrf_analysis_initData<true, true, T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT, dindT,
                                           dvalT, dpivP, dpivQ, hptrM, hindM, hvalM, hptrT, hindT,
                                           hvalT, hpivP, hpivQ);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(handle, n, nnzM, dptrM.data(), dindM.data(),
                                                 dvalM.data(), nnzT, dptrT.data(), dindT.data(),
                                                 dvalT.data(), dpivP.data(), dpivQ.data(), rfinfo));

    // No error to calculate...
    // (analysis phase is required by refact and solve, so its results are validated there)
    *max_err = 0;
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_analysis_getPerfData(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nnzM,
                                Ud& dptrM,
                                Ud& dindM,
                                Td& dvalM,
                                const rocblas_int nnzT,
                                Ud& dptrT,
                                Ud& dindT,
                                Td& dvalT,
                                Ud& dpivP,
                                Ud& dpivQ,
                                rocsolver_rfinfo rfinfo,
                                Uh& hptrM,
                                Uh& hindM,
                                Th& hvalM,
                                Uh& hptrT,
                                Uh& hindT,
                                Th& hvalT,
                                Uh& hpivP,
                                Uh& hpivQ,
                                double* gpu_time_used,
                                double* cpu_time_used,
                                const rocblas_int hot_calls,
                                const int profile,
                                const bool profile_kernels,
                                const bool perf)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrrf_analysis_initData<true, false, T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT,
                                            dindT, dvalT, dpivP, dpivQ, hptrM, hindM, hvalM, hptrT,
                                            hindT, hvalT, hpivP, hpivQ);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrrf_analysis_initData<false, true, T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT,
                                                dindT, dvalT, dpivP, dpivQ, hptrM, hindM, hvalM,
                                                hptrT, hindT, hvalT, hpivP, hpivQ);

        CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
            handle, n, nnzM, dptrM.data(), dindM.data(), dvalM.data(), nnzT, dptrT.data(),
            dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), rfinfo));
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

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        csrrf_analysis_initData<false, true, T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT,
                                                dindT, dvalT, dpivP, dpivQ, hptrM, hindM, hvalM,
                                                hptrT, hindT, hvalT, hpivP, hpivQ);

        start = get_time_us_sync(stream);
        rocsolver_csrrf_analysis(handle, n, nnzM, dptrM.data(), dindM.data(), dvalM.data(), nnzT,
                                 dptrT.data(), dindT.data(), dvalT.data(), dpivP.data(),
                                 dpivQ.data(), rfinfo);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_csrrf_analysis(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnzM = argus.get<rocblas_int>("nnzM");
    rocblas_int hot_calls = argus.iters;

    // determine valid test case
    if(n > 0)
    {
        if(n <= 35)
            n = 20;
        else if(n <= 75)
            n = 50;
        else if(n <= 200)
            n = 100;
        else
            n = 300;
    }
    if(nnzM > 0)
    {
        if(nnzM <= 30)
            nnzM = 20;
        else if(nnzM <= 55)
            nnzM = 40;
        else if(nnzM <= 110)
            nnzM = 75;
        else if(nnzM <= 200)
            nnzM = 150;
        else
            nnzM = 250;
    }

    // read nnzT
    rocblas_int nnzT = 0;
    if(n > 0 && nnzM > 0)
    {
        std::string testcase = fmt::format("case_{}_{}/", n, nnzM);
    }

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzM < 0 || nnzT < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(
                                  handle, n, nnzM, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                  (T*)nullptr, nnzT, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                  (T*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query if necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_csrrf_analysis(
            handle, n, nnzM, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, nnzT,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
            (rocblas_int*)nullptr, rfinfo));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // determine sizes
    size_t size_ptrM = size_t(n) + 1;
    size_t size_indM = size_t(nnzM);
    size_t size_valM = size_t(nnzM);
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrM(size_ptrM, 1, size_ptrM, 1);
    host_strided_batch_vector<rocblas_int> hindM(size_indM, 1, size_indM, 1);
    host_strided_batch_vector<T> hvalM(size_valM, 1, size_valM, 1);
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<rocblas_int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<rocblas_int> hpivQ(size_pivQ, 1, size_pivQ, 1);
    device_strided_batch_vector<rocblas_int> dptrM(size_ptrM, 1, size_ptrM, 1);
    device_strided_batch_vector<rocblas_int> dindM(size_indM, 1, size_indM, 1);
    device_strided_batch_vector<T> dvalM(size_valM, 1, size_valM, 1);
    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    device_strided_batch_vector<rocblas_int> dpivP(size_pivP, 1, size_pivP, 1);
    device_strided_batch_vector<rocblas_int> dpivQ(size_pivQ, 1, size_pivQ, 1);
    CHECK_HIP_ERROR(dptrM.memcheck());
    CHECK_HIP_ERROR(dptrT.memcheck());
    if(size_indM)
        CHECK_HIP_ERROR(dindM.memcheck());
    if(size_valM)
        CHECK_HIP_ERROR(dvalM.memcheck());
    if(size_indT)
        CHECK_HIP_ERROR(dindT.memcheck());
    if(size_valT)
        CHECK_HIP_ERROR(dvalT.memcheck());
    if(size_pivP)
        CHECK_HIP_ERROR(dpivP.memcheck());
    if(size_pivQ)
        CHECK_HIP_ERROR(dpivQ.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_analysis(handle, n, nnzM, dptrM.data(), dindM.data(),
                                                       dvalM.data(), nnzT, dptrT.data(),
                                                       dindT.data(), dvalT.data(), dpivP.data(),
                                                       dpivQ.data(), rfinfo),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_analysis_getError<T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT, dindT, dvalT,
                                   dpivP, dpivQ, rfinfo, hptrM, hindM, dvalM, hptrT, hindT, dvalT,
                                   hpivP, hpivQ, &max_error);

    // collect performance data
    if(argus.timing)
        csrrf_analysis_getPerfData<T>(handle, n, nnzM, dptrM, dindM, dvalM, nnzT, dptrT, dindT,
                                      dvalT, dpivP, dpivQ, rfinfo, hptrM, hindM, dvalM, hptrT,
                                      hindT, dvalT, hpivP, hpivQ, &gpu_time_used, &cpu_time_used,
                                      hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // N/A

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nnzM", "nnzT");
            rocsolver_bench_output(n, nnzM, nnzT);

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

#define EXTERN_TESTING_CSRRF_ANALYSIS(...) \
    extern template void testing_csrrf_analysis<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CSRRF_ANALYSIS, FOREACH_SCALAR_TYPE, APPLY_STAMP)
