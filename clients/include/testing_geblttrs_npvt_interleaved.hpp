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

template <typename T>
void geblttrs_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            const rocblas_int nrhs,
                                            T dA,
                                            const rocblas_int lda,
                                            T dB,
                                            const rocblas_int ldb,
                                            T dC,
                                            const rocblas_int ldc,
                                            T dX,
                                            const rocblas_int ldx,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(nullptr, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, (T) nullptr,
                                                              lda, dB, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              (T) nullptr, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                              ldb, (T) nullptr, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                              ldb, dC, ldc, (T) nullptr, ldx, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, 0, nblocks, nrhs, (T) nullptr,
                                                              lda, (T) nullptr, ldb, (T) nullptr,
                                                              ldc, (T) nullptr, ldx, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, 0, nrhs, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              (T) nullptr, ldx, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, 0, dA, lda, dB,
                                                              ldb, dC, ldc, (T) nullptr, ldx, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrs_npvt_interleaved_bad_arg()
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
    rocblas_int bc = 1;

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
    geblttrs_npvt_interleaved_checkBadArgs(handle, nb, nblocks, nrhs, dA.data(), lda, dB.data(),
                                           ldb, dC.data(), ldc, dX.data(), ldx, bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_initData(const rocblas_handle handle,
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
                            hB[k][b + i * bc + j * bc * ldb] += 400;
                        else
                            hB[k][b + i * bc + j * bc * ldb] -= 4;
                    }

                    for(rocblas_int k = 0; k < nblocks - 1; k++)
                    {
                        hA[k][b + i * bc + j * bc * lda] -= 4;
                        hC[k][b + i * bc + j * bc * ldc] -= 4;
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

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getError(const rocblas_handle handle,
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
                                        Th& hX,
                                        Th& hXRes,
                                        double* max_err)
{
    // input data initialization
    geblttrs_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb,
                                                      dC, ldc, dX, ldx, bc, hA, hB, hC, hX);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(),
                                                            lda, dB.data(), ldb, dC.data(), ldc,
                                                            dX.data(), ldx, bc));
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
    }
}

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getPerfData(const rocblas_handle handle,
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
        // geblttrs_npvt_interleaved_initData<true, false, T>(
        //     handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX, ldx, bc, hA, hB, hC, hX);

        // // cpu-lapack performance (only if not in perf mode)
        // *cpu_time_used = get_time_us_no_sync();
        // for(rocblas_int b = 0; b < bc; ++b)
        // {
        //    cpu_getrs(trans, n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
        // }
        // *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
        *cpu_time_used = nan("");
    }

    geblttrs_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb,
                                                       dC, ldc, dX, ldx, bc, hA, hB, hC, hX);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrs_npvt_interleaved_initData<false, true, T>(
            handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX, ldx, bc, hA, hB, hC, hX);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs,
                                                                dA.data(), lda, dB.data(), ldb,
                                                                dC.data(), ldc, dX.data(), ldx, bc));
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
        geblttrs_npvt_interleaved_initData<false, true, T>(
            handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX, ldx, bc, hA, hB, hC, hX);

        start = get_time_us_sync(stream);
        rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), lda, dB.data(),
                                            ldb, dC.data(), ldc, dX.data(), ldx, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrs_npvt_interleaved(Arguments& argus)
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

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * nb * bc;
    size_t size_B = size_t(ldb) * nb * bc;
    size_t size_C = size_t(ldc) * nb * bc;
    size_t size_X = size_t(ldx) * nrhs * bc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || nrhs < 0 || lda < nb || ldb < nb || ldc < nb
                         || ldx < nb || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                                  handle, nb, nblocks, nrhs, (T*)nullptr, lda, (T*)nullptr, ldb,
                                  (T*)nullptr, ldc, (T*)nullptr, ldx, bc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, (T*)nullptr,
                                                              lda, (T*)nullptr, ldb, (T*)nullptr,
                                                              ldc, (T*)nullptr, ldx, bc));

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
    host_strided_batch_vector<T> hA(size_A, 1, size_A, max(1, nblocks));
    host_strided_batch_vector<T> hB(size_B, 1, size_B, max(1, nblocks));
    host_strided_batch_vector<T> hC(size_C, 1, size_C, max(1, nblocks));
    host_strided_batch_vector<T> hX(size_X, 1, size_X, max(1, nblocks));
    host_strided_batch_vector<T> hXRes(size_XRes, 1, size_XRes, max(1, nblocks));
    device_strided_batch_vector<T> dA(size_A, 1, size_A, max(1, nblocks));
    device_strided_batch_vector<T> dB(size_B, 1, size_B, max(1, nblocks));
    device_strided_batch_vector<T> dC(size_C, 1, size_C, max(1, nblocks));
    device_strided_batch_vector<T> dX(size_X, 1, size_X, max(1, nblocks));
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
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), lda,
                                                dB.data(), ldb, dC.data(), ldc, dX.data(), ldx, bc),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrs_npvt_interleaved_getError<T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc,
                                              dX, ldx, bc, hA, hB, hC, hX, hXRes, &max_error);

    // collect performance data
    if(argus.timing)
        geblttrs_npvt_interleaved_getPerfData<T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC,
                                                 ldc, dX, ldx, bc, hA, hB, hC, hX, &gpu_time_used,
                                                 &cpu_time_used, hot_calls, argus.profile,
                                                 argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("nb", "nblocks", "nrhs", "lda", "ldb", "ldc", "ldx", "batch_c");
            rocsolver_bench_output(nb, nblocks, nrhs, lda, ldb, ldc, ldx, bc);
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

#define EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrs_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
