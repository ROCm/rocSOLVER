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

template <bool STRIDED, bool POTRF, typename T, typename U>
void potf2_potrf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potf2_potrf(STRIDED, POTRF, nullptr, uplo, n, dA, lda, stA, dinfo, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, rocblas_fill_full, n, dA,
                                                lda, stA, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA, lda, stA, dinfo, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T) nullptr, lda, stA, dinfo, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA, lda, stA, (U) nullptr, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, 0, (T) nullptr, lda, stA, dinfo, bc),
        rocblas_status_success);
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA, lda, stA, (U) nullptr, 0),
            rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA, lda, stA, dinfo, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool POTRF, typename T>
void testing_potf2_potrf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        potf2_potrf_checkBadArgs<STRIDED, POTRF>(handle, uplo, n, dA.data(), lda, stA, dinfo.data(),
                                                 bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        potf2_potrf_checkBadArgs<STRIDED, POTRF>(handle, uplo, n, dA.data(), lda, stA, dinfo.data(),
                                                 bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potf2_potrf_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to ensure positive definiteness
            for(rocblas_int i = 0; i < n; i++)
                hA[b][i + i * lda] = hA[b][i + i * lda] * sconj(hA[b][i + i * lda]) * 400;

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // make some matrices not positive definite
                // always the same elements for debugging purposes
                // the algorithm must detect the lower order of the principal minors <= 0
                // in those matrices in the batch that are non positive definite
                rocblas_int i = n / 4 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
                i = n / 2 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
                i = n - 1 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool POTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potf2_potrf_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    // input data initialization
    potf2_potrf_initData<true, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo,
                                        singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(), lda, stA,
                                              dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        POTRF ? cblas_potrf<T>(uplo, n, hA[b], lda, hInfo[b])
              : cblas_potf2<T>(uplo, n, hA[b], lda, hInfo[b]);
    }

    // error is ||hA - hARes|| / ||hA|| (ideally ||LL' - Lres Lres'|| / ||LL'||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    rocblas_int nn;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        nn = hInfoRes[b][0] == 0 ? n : hInfoRes[b][0];
        // (TODO: For now, the algorithm is modifying the whole input matrix even when
        //  it is not positive definite. So we only check the principal nn-by-nn submatrix.
        //  Once this is corrected, nn could be always equal to n.)
        *max_err = (uplo == rocblas_fill_lower)
            ? norm_error_lowerTr('F', nn, nn, lda, hA[b], hARes[b])
            : norm_error_upperTr('F', nn, nn, lda, hA[b], hARes[b]);
    }

    // also check info for non positive definite cases
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    *max_err += err;
}

template <bool STRIDED, bool POTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potf2_potrf_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
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
        potf2_potrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo,
                                             singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            POTRF ? cblas_potrf<T>(uplo, n, hA[b], lda, hInfo[b])
                  : cblas_potf2<T>(uplo, n, hA[b], lda, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    potf2_potrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo,
                                         singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        potf2_potrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo,
                                             singular);

        CHECK_ROCBLAS_ERROR(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(), lda,
                                                  stA, dInfo.data(), bc));
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
        potf2_potrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo,
                                             singular);

        start = get_time_us_sync(stream);
        rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(), lda, stA, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool POTRF, typename T>
void testing_potf2_potrf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n,
                                                        (T* const*)nullptr, lda, stA,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n,
                                                        (T* const*)nullptr, lda, stA,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n,
                                                    (T* const*)nullptr, lda, stA,
                                                    (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, bc));

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
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            potf2_potrf_getError<STRIDED, POTRF, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA,
                                                    hARes, hInfo, hInfoRes, &max_error,
                                                    argus.singular);

        // collect performance data
        if(argus.timing)
            potf2_potrf_getPerfData<STRIDED, POTRF, T>(
                handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo, &gpu_time_used, &cpu_time_used,
                hot_calls, argus.profile, argus.profile_kernels, argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            potf2_potrf_getError<STRIDED, POTRF, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA,
                                                    hARes, hInfo, hInfoRes, &max_error,
                                                    argus.singular);

        // collect performance data
        if(argus.timing)
            potf2_potrf_getPerfData<STRIDED, POTRF, T>(
                handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo, &gpu_time_used, &cpu_time_used,
                hot_calls, argus.profile, argus.profile_kernels, argus.perf, argus.singular);
    }

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
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "batch_c");
                rocsolver_bench_output(uploC, n, lda, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideA", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
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

#define EXTERN_TESTING_POTF2_POTRF(...) \
    extern template void testing_potf2_potrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_POTF2_POTRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
