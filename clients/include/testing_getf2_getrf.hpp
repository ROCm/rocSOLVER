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

template <bool STRIDED, bool GETRF, typename T, typename U>
void getf2_getrf_checkBadArgs(const rocblas_handle handle,
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
void testing_getf2_getrf_bad_arg()
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
        getf2_getrf_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
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
        getf2_getrf_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_initData(const rocblas_handle handle,
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
void getf2_getrf_getError(const rocblas_handle handle,
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
                          Th& hARes,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    // input data initialization
    getf2_getrf_initData<true, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                        hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GETRF ? cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b])
              : cpu_getf2(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA|| (ideally ||LU - Lres Ures|| / ||LU||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;

        // also check pivoting (count the number of incorrect pivots)
        err = 0;
        for(rocblas_int i = 0; i < min(m, n); ++i)
        {
            EXPECT_EQ(hIpiv[b][i], hIpivRes[b][i]) << "where b = " << b << ", i = " << i;
            if(hIpiv[b][i] != hIpivRes[b][i])
                err++;
        }
        *max_err = err > *max_err ? err : *max_err;
    }

    // also check info for singularities
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_getPerfData(const rocblas_handle handle,
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
    if(!perf)
    {
        getf2_getrf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                             hIpiv, hInfo, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GETRF ? cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b])
                  : cpu_getf2(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getf2_getrf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                         hIpiv, hInfo, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getf2_getrf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                             hIpiv, hInfo, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA,
                                                  dIpiv.data(), stP, dInfo.data(), bc));
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
        getf2_getrf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                             hIpiv, hInfo, singular);

        start = get_time_us_sync(stream);
        rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP,
                              dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getf2_getrf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", min(m, n));

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T*)nullptr,
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
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T*)nullptr, lda,
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
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                        stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo,
                                                    bc, hA, hARes, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                        stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo,
                                                    bc, hA, hARes, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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

#define EXTERN_TESTING_GETF2_GETRF(...) \
    extern template void testing_getf2_getrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GETF2_GETRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
