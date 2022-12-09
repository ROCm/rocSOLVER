/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T>
void geblttrs_npvt_checkBadArgs(const rocblas_handle handle,
                                const rocblas_int nb,
                                const rocblas_int nblocks,
                                const rocblas_int nrhs,
                                T dA,
                                const rocblas_int lda,
                                const rocblas_stride stA,
                                T dB,
                                const rocblas_int ldb,
                                const rocblas_stride stB,
                                T dC,
                                const rocblas_int ldc,
                                const rocblas_stride stC,
                                T dX,
                                const rocblas_int ldx,
                                const rocblas_stride stX,
                                const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, nullptr, nb, nblocks, nrhs, dA, lda, stA,
                                                  dB, ldb, stB, dC, ldc, stC, dX, ldx, stX, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA, lda, stA,
                                                      dB, ldb, stB, dC, ldc, stC, dX, ldx, stX, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, (T) nullptr, lda,
                                                  stA, dB, ldb, stB, dC, ldc, stC, dX, ldx, stX, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA, lda, stA,
                                                  (T) nullptr, ldb, stB, dC, ldc, stC, dX, ldx, stX,
                                                  bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA, lda, stA, dB,
                                                  ldb, stB, (T) nullptr, ldc, stC, dX, ldx, stX, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA, lda, stA, dB,
                                                  ldb, stB, dC, ldc, stC, (T) nullptr, ldx, stX, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, 0, nblocks, nrhs, (T) nullptr,
                                                  lda, stA, (T) nullptr, ldb, stB, (T) nullptr, ldc,
                                                  stC, (T) nullptr, ldx, stX, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, 0, nrhs, (T) nullptr, lda,
                                                  stA, (T) nullptr, ldb, stB, (T) nullptr, ldc, stC,
                                                  (T) nullptr, ldx, stX, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, 0, dA, lda, stA, dB,
                                                  ldb, stB, dC, ldc, stC, (T) nullptr, ldx, stX, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA, lda, stA,
                                                      dB, ldb, stB, dC, ldc, stC, dX, ldx, stX, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geblttrs_npvt_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_int ldx = 1;
    rocblas_stride stA = 2;
    rocblas_stride stB = 2;
    rocblas_stride stC = 2;
    rocblas_stride stX = 2;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_batch_vector<T> dC(1, 1, 1);
        device_batch_vector<T> dX(1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());

        // check bad arguments
        geblttrs_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, nrhs, dA.data(), lda, stA, dB.data(),
                                            ldb, stB, dC.data(), ldc, stC, dX.data(), ldx, stX, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<T> dC(1, 1, 1, 1);
        device_strided_batch_vector<T> dX(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());

        // check bad arguments
        geblttrs_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, nrhs, dA.data(), lda, stA, dB.data(),
                                            ldb, stB, dC.data(), ldc, stC, dX.data(), ldx, stX, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrs_npvt_initData(const rocblas_handle handle,
                            const rocblas_int nb,
                            const rocblas_int nblocks,
                            const rocblas_int nrhs,
                            Td& dA,
                            const rocblas_int lda,
                            Td& dB,
                            const rocblas_int ldb,
                            Td& dC,
                            const rocblas_int ldc,
                            Td& dX,
                            const rocblas_int ldx,
                            const rocblas_int bc,
                            Th& hA,
                            Th& hB,
                            Th& hC,
                            Th& hX)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);
        rocblas_init<T>(hC, false);
        rocblas_init<T>(hX, false);

        rocblas_int n = nb * nblocks;

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to avoid singularities
            // leaving matrix as diagonal dominant so that pivoting is not required
            for(rocblas_int i = 0; i < nb; i++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int k = 0; k < nblocks; k++)
                    {
                        if(i == j)
                            hB[b][i + j * ldb + k * ldb * nb] += 400;
                        else
                            hB[b][i + j * ldb + k * ldb * nb] -= 4;
                    }

                    for(rocblas_int k = 0; k < nblocks - 1; k++)
                    {
                        hA[b][i + j * lda + k * lda * nb] -= 4;
                        hC[b][i + j * ldc + k * ldc * nb] -= 4;
                    }
                }
            }

            // TODO: Factorize the blocked matrix
        }
    }

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(dX.transfer_from(hX));
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void geblttrs_npvt_getError(const rocblas_handle handle,
                            const rocblas_int nb,
                            const rocblas_int nblocks,
                            const rocblas_int nrhs,
                            Td& dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            Td& dB,
                            const rocblas_int ldb,
                            const rocblas_stride stB,
                            Td& dC,
                            const rocblas_int ldc,
                            const rocblas_stride stC,
                            Td& dX,
                            const rocblas_int ldx,
                            const rocblas_stride stX,
                            const rocblas_int bc,
                            Th& hA,
                            Th& hB,
                            Th& hC,
                            Th& hX,
                            Th& hXRes,
                            double* max_err)
{
    // input data initialization
    geblttrs_npvt_initData<true, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX,
                                          ldx, bc, hA, hB, hC, hX);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA.data(), lda,
                                                stA, dB.data(), ldb, stB, dC.data(), ldc, stC,
                                                dX.data(), ldx, stX, bc));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));

    // // CPU lapack
    // for(rocblas_int b = 0; b < bc; ++b)
    // {
    //     cpu_getrs(trans, n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
    // }

    double err = 0;
    *max_err = 0;

    // error is ||hX - hXRes|| / ||hX||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // TODO: Complete the test error calculation

        err = norm_error('F', nb, nrhs * nblocks, ldx, hX[b], hXRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void geblttrs_npvt_getPerfData(const rocblas_handle handle,
                               const rocblas_int nb,
                               const rocblas_int nblocks,
                               const rocblas_int nrhs,
                               Td& dA,
                               const rocblas_int lda,
                               const rocblas_stride stA,
                               Td& dB,
                               const rocblas_int ldb,
                               const rocblas_stride stB,
                               Td& dC,
                               const rocblas_int ldc,
                               const rocblas_stride stC,
                               Td& dX,
                               const rocblas_int ldx,
                               const rocblas_stride stX,
                               const rocblas_int bc,
                               Th& hA,
                               Th& hB,
                               Th& hC,
                               Th& hX,
                               double* gpu_time_used,
                               double* cpu_time_used,
                               const rocblas_int hot_calls,
                               const int profile,
                               const bool profile_kernels,
                               const bool perf)
{
    if(!perf)
    {
        // geblttrs_npvt_initData<true, false, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc,
        //                                        dX, ldx, bc, hA, hB, hC, hX);

        // // cpu-lapack performance (only if not in perf mode)
        // *cpu_time_used = get_time_us_no_sync();
        // for(rocblas_int b = 0; b < bc; ++b)
        // {
        //     cpu_getrs(trans, n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
        // }
        // *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
        *cpu_time_used = nan("");
    }

    geblttrs_npvt_initData<true, false, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX,
                                           ldx, bc, hA, hB, hC, hX);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrs_npvt_initData<false, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc,
                                               dX, ldx, bc, hA, hB, hC, hX);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA.data(),
                                                    lda, stA, dB.data(), ldb, stB, dC.data(), ldc,
                                                    stC, dX.data(), ldx, stX, bc));
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
        geblttrs_npvt_initData<false, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc,
                                               dX, ldx, bc, hA, hB, hC, hX);

        start = get_time_us_sync(stream);
        rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, dA.data(), lda, stA, dB.data(),
                                ldb, stB, dC.data(), ldc, stC, dX.data(), ldx, stX, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geblttrs_npvt(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", nb);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * nb * nblocks);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nb * nblocks);
    rocblas_stride stC = argus.get<rocblas_stride>("strideC", ldc * nb * nblocks);
    rocblas_stride stX = argus.get<rocblas_stride>("strideX", ldx * nrhs * nblocks);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stXRes = (argus.unit_check || argus.norm_check) ? stX : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * nb * nblocks;
    size_t size_B = size_t(ldb) * nb * nblocks;
    size_t size_C = size_t(ldc) * nb * nblocks;
    size_t size_X = size_t(ldx) * nrhs * nblocks;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || nrhs < 0 || lda < nb || ldb < nb || ldc < nb
                         || ldx < nb || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, (T* const*)nullptr, lda,
                                        stA, (T* const*)nullptr, ldb, stB, (T* const*)nullptr, ldc,
                                        stC, (T* const*)nullptr, ldx, stX, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs,
                                                          (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                          stB, (T*)nullptr, ldc, stC, (T*)nullptr,
                                                          ldx, stX, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_geblttrs_npvt(
                STRIDED, handle, nb, nblocks, nrhs, (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                ldb, stB, (T* const*)nullptr, ldc, stC, (T* const*)nullptr, ldx, stX, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs, (T*)nullptr,
                                                      lda, stA, (T*)nullptr, ldb, stB, (T*)nullptr,
                                                      ldc, stC, (T*)nullptr, ldx, stX, bc));

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
        host_batch_vector<T> hC(size_C, 1, bc);
        host_batch_vector<T> hX(size_X, 1, bc);
        host_batch_vector<T> hXRes(size_XRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dC(size_C, 1, bc);
        device_batch_vector<T> dX(size_X, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());

        // check quick return
        if(nb == 0 || nblocks == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs,
                                                          dA.data(), lda, stA, dB.data(), ldb, stB,
                                                          dC.data(), ldc, stC, dX.data(), ldx, stX,
                                                          bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geblttrs_npvt_getError<STRIDED, T>(handle, nb, nblocks, nrhs, dA, lda, stA, dB, ldb,
                                               stB, dC, ldc, stC, dX, ldx, stX, bc, hA, hB, hC, hX,
                                               hXRes, &max_error);

        // collect performance data
        if(argus.timing)
            geblttrs_npvt_getPerfData<STRIDED, T>(handle, nb, nblocks, nrhs, dA, lda, stA, dB, ldb,
                                                  stB, dC, ldc, stC, dX, ldx, stX, bc, hA, hB, hC,
                                                  hX, &gpu_time_used, &cpu_time_used, hot_calls,
                                                  argus.profile, argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hC(size_C, 1, stC, bc);
        host_strided_batch_vector<T> hX(size_X, 1, stX, bc);
        host_strided_batch_vector<T> hXRes(size_XRes, 1, stXRes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dC(size_C, 1, stC, bc);
        device_strided_batch_vector<T> dX(size_X, 1, stX, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());

        // check quick return
        if(nb == 0 || nblocks == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt(STRIDED, handle, nb, nblocks, nrhs,
                                                          dA.data(), lda, stA, dB.data(), ldb, stB,
                                                          dC.data(), ldc, stC, dX.data(), ldx, stX,
                                                          bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geblttrs_npvt_getError<STRIDED, T>(handle, nb, nblocks, nrhs, dA, lda, stA, dB, ldb,
                                               stB, dC, ldc, stC, dX, ldx, stX, bc, hA, hB, hC, hX,
                                               hXRes, &max_error);

        // collect performance data
        if(argus.timing)
            geblttrs_npvt_getPerfData<STRIDED, T>(handle, nb, nblocks, nrhs, dA, lda, stA, dB, ldb,
                                                  stB, dC, ldc, stC, dX, ldx, stX, bc, hA, hB, hC,
                                                  hX, &gpu_time_used, &cpu_time_used, hot_calls,
                                                  argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using nb * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, nb);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("nb", "nblocks", "nrhs", "lda", "ldb", "ldc", "ldx",
                                       "batch_c");
                rocsolver_bench_output(nb, nblocks, nrhs, lda, ldb, ldc, ldx, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("nb", "nblocks", "nrhs", "lda", "strideA", "ldb", "strideB",
                                       "ldc", "strideC", "ldx", "strideX", "batch_c");
                rocsolver_bench_output(nb, nblocks, nrhs, lda, stA, ldb, stB, ldc, stC, ldx, stX, bc);
            }
            else
            {
                rocsolver_bench_output("nb", "nblocks", "nrhs", "lda", "ldb", "ldc", "ldx");
                rocsolver_bench_output(nb, nblocks, nrhs, lda, ldb, ldc, ldx);
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

#define EXTERN_TESTING_GEBLTTRS_NPVT(...) \
    extern template void testing_geblttrs_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRS_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
