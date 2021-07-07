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

template <bool STRIDED, bool SYGST, typename T>
void sygsx_hegsx_checkBadArgs(const rocblas_handle handle,
                              const rocblas_eform itype,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              T dB,
                              const rocblas_int ldb,
                              const rocblas_stride stB,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, nullptr, itype, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, rocblas_eform(-1), uplo, n,
                                                dA, lda, stA, dB, ldb, stB, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, rocblas_fill_full, n,
                                                dA, lda, stA, dB, ldb, stB, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA, lda,
                                                    stA, dB, ldb, stB, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, (T) nullptr,
                                                lda, stA, dB, ldb, stB, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA, lda,
                                                stA, (T) nullptr, ldb, stB, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, 0, (T) nullptr,
                                                lda, stA, dB, ldb, stB, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, 0, dA, lda,
                                                stA, (T) nullptr, ldb, stB, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA, lda,
                                                    stA, dB, ldb, stB, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool SYGST, typename T>
void testing_sygsx_hegsx_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_int ldb = 1;
    rocblas_stride stB = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());

        // check bad arguments
        sygsx_hegsx_checkBadArgs<STRIDED, SYGST>(handle, itype, uplo, n, dA.data(), lda, stA,
                                                 dB.data(), ldb, stB, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());

        // check bad arguments
        sygsx_hegsx_checkBadArgs<STRIDED, SYGST>(handle, itype, uplo, n, dA.data(), lda, stA,
                                                 dB.data(), ldb, stB, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygsx_hegsx_initData(const rocblas_handle handle,
                          const rocblas_eform itype,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hATmp,
                          Th& hB,
                          Th& hBTmp)
{
    if(CPU)
    {
        rocblas_int info;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to ensure positive definiteness
            for(rocblas_int i = 0; i < n; i++)
            {
                hA[b][i + i * lda] = hA[b][i + i * lda] * sconj(hA[b][i + i * lda]) * 400;
                hB[b][i + i * ldb] = hB[b][i + i * ldb] * sconj(hB[b][i + i * ldb]) * 400;
            }

            // apply Cholesky factorization to B
            cblas_potrf(uplo, n, hB[b], ldb, &info);
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, bool SYGST, typename T, typename Td, typename Th>
void sygsx_hegsx_getError(const rocblas_handle handle,
                          const rocblas_eform itype,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Th& hB,
                          Th& hBRes,
                          double* max_err)
{
    // input data initialization
    sygsx_hegsx_initData<true, true, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc, hA,
                                        hARes, hB, hBRes);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA.data(),
                                              lda, stA, dB.data(), ldb, stB, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        SYGST ? cblas_sygst_hegst<T>(itype, uplo, n, hA[b], lda, hB[b], ldb)
              : cblas_sygs2_hegs2<T>(itype, uplo, n, hA[b], lda, hB[b], ldb);
    }

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    rocblas_int nn;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool SYGST, typename T, typename Td, typename Th>
void sygsx_hegsx_getPerfData(const rocblas_handle handle,
                             const rocblas_eform itype,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Td& dB,
                             const rocblas_int ldb,
                             const rocblas_stride stB,
                             const rocblas_int bc,
                             Th& hA,
                             Th& hATmp,
                             Th& hB,
                             Th& hBTmp,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool perf)
{
    if(!perf)
    {
        sygsx_hegsx_initData<true, false, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc,
                                             hA, hATmp, hB, hBTmp);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            SYGST ? cblas_sygst_hegst<T>(itype, uplo, n, hA[b], lda, hB[b], ldb)
                  : cblas_sygs2_hegs2<T>(itype, uplo, n, hA[b], lda, hB[b], ldb);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygsx_hegsx_initData<true, false, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc, hA,
                                         hATmp, hB, hBTmp);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygsx_hegsx_initData<false, true, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc,
                                             hA, hATmp, hB, hBTmp);

        CHECK_ROCBLAS_ERROR(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA.data(),
                                                  lda, stA, dB.data(), ldb, stB, bc));
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
        sygsx_hegsx_initData<false, true, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc,
                                             hA, hATmp, hB, hBTmp);

        start = get_time_us_sync(stream);
        rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n, dA.data(), lda, stA,
                              dB.data(), ldb, stB, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool SYGST, typename T>
void testing_sygsx_hegsx(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char itypeC = argus.get<char>("itype");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * n);

    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // hARes and hBRes should always be allocated (used in initData)
    size_t stARes = stA;
    size_t stBRes = stB;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        (T* const*)nullptr, lda, stA,
                                                        (T* const*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                        stB, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // hARes and hBRes should always be allocated (used in initData)
    size_t size_ARes = size_A;
    size_t size_BRes = size_B;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        (T* const*)nullptr, lda, stA,
                                                        (T* const*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                        stB, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                    (T* const*)nullptr, lda, stA,
                                                    (T* const*)nullptr, ldb, stB, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                    (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB, bc));

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
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        dA.data(), lda, stA, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygsx_hegsx_getError<STRIDED, SYGST, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb,
                                                    stB, bc, hA, hARes, hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing)
            sygsx_hegsx_getPerfData<STRIDED, SYGST, T>(
                handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc, hA, hARes, hB, hBRes,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygsx_hegsx(STRIDED, SYGST, handle, itype, uplo, n,
                                                        dA.data(), lda, stA, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygsx_hegsx_getError<STRIDED, SYGST, T>(handle, itype, uplo, n, dA, lda, stA, dB, ldb,
                                                    stB, bc, hA, hARes, hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing)
            sygsx_hegsx_getPerfData<STRIDED, SYGST, T>(
                handle, itype, uplo, n, dA, lda, stA, dB, ldb, stB, bc, hA, hARes, hB, hBRes,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.perf);
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
                rocsolver_bench_output("itype", "uplo", "n", "lda", "ldb", "batch_c");
                rocsolver_bench_output(itypeC, uploC, n, lda, ldb, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "uplo", "n", "lda", "strideA", "ldb", "strideB",
                                       "batch_c");
                rocsolver_bench_output(itypeC, uploC, n, lda, stA, ldb, stB, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "uplo", "n", "lda", "ldb");
                rocsolver_bench_output(itypeC, uploC, n, lda, ldb);
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
