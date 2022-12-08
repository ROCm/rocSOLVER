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

template <typename T, typename U>
void geblttrf_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            T dA,
                                            const rocblas_int lda,
                                            T dB,
                                            const rocblas_int ldb,
                                            T dC,
                                            const rocblas_int ldc,
                                            U dInfo,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(nullptr, nb, nblocks, dA, lda, dB,
                                                              ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, dInfo, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T) nullptr, lda,
                                                              dB, ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda,
                                                              (T) nullptr, ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              (T) nullptr, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, 0, nblocks, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, 0, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, (U) nullptr, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrf_npvt_interleaved_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_int bc = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dB(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    geblttrf_npvt_interleaved_checkBadArgs(handle, nb, nblocks, dA.data(), lda, dB.data(), ldb,
                                           dC.data(), ldc, dInfo.data(), bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrf_npvt_interleaved_initData(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        const rocblas_int bc,
                                        Th& hA,
                                        Th& hB,
                                        Th& hC,
                                        const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);
        rocblas_init<T>(hC, false);

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

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes)

                rocblas_int jj = n / 4 + b;
                jj -= (jj / n) * n;
                rocblas_int j = jj % nb;
                rocblas_int k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[k][b + i * bc + j * bc * ldb] = 0;
                    if(k < nblocks - 1)
                        hA[k][b + i * bc + j * bc * lda] = 0;
                    if(k > 0)
                        hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                }

                jj = n / 2 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[k][b + i * bc + j * bc * ldb] = 0;
                    if(k < nblocks - 1)
                        hA[k][b + i * bc + j * bc * lda] = 0;
                    if(k > 0)
                        hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                }

                jj = n - 1 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[k][b + i * bc + j * bc * ldb] = 0;
                    if(k < nblocks - 1)
                        hA[k][b + i * bc + j * bc * lda] = 0;
                    if(k > 0)
                        hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                }
            }
        }
    }

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void geblttrf_npvt_interleaved_getError(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        Ud& dInfo,
                                        const rocblas_int bc,
                                        Th& hA,
                                        Th& hB,
                                        Th& hBRes,
                                        Th& hC,
                                        Th& hCRes,
                                        Uh& hInfo,
                                        Uh& hInfoRes,
                                        double* max_err,
                                        const bool singular)
{
    // input data initialization
    geblttrf_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC,
                                                      ldc, bc, hA, hB, hC, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
        handle, nb, nblocks, dA.data(), lda, dB.data(), ldb, dC.data(), ldc, dInfo.data(), bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // // CPU lapack
    // for(rocblas_int b = 0; b < bc; ++b)
    // {
    //     cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    // }

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
        {
            if(hInfoRes[b][0] <= 0)
                err++;
        }
        else
        {
            if(hInfoRes[b][0] != 0)
                err++;
        }
    }
    *max_err += err;

    // error is ||hB - hBRes|| / ||hB|| or ||hC - hCRes|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            // TODO: Complete the test error calculation
        }
    }
}

template <typename T, typename Td, typename Ud, typename Th>
void geblttrf_npvt_interleaved_getPerfData(const rocblas_handle handle,
                                           const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           Td& dA,
                                           const rocblas_int lda,
                                           Td& dB,
                                           const rocblas_int ldb,
                                           Td& dC,
                                           const rocblas_int ldc,
                                           Ud& dInfo,
                                           const rocblas_int bc,
                                           Th& hA,
                                           Th& hB,
                                           Th& hC,
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
        // geblttrf_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
        //                                     hB, hC, singular);

        // // cpu-lapack performance (only if not in perf mode)
        // *cpu_time_used = get_time_us_no_sync();
        // for(rocblas_int b = 0; b < bc; ++b)
        // {
        //     cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        // }
        // *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
        *cpu_time_used = nan("");
    }

    geblttrf_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC,
                                                       ldc, bc, hA, hB, hC, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb,
                                                           dC, ldc, bc, hA, hB, hC, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
            handle, nb, nblocks, dA.data(), lda, dB.data(), ldb, dC.data(), ldc, dInfo.data(), bc));
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
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb,
                                                           dC, ldc, bc, hA, hB, hC, singular);

        start = get_time_us_sync(stream);
        rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA.data(), lda, dB.data(), ldb,
                                            dC.data(), ldc, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrf_npvt_interleaved(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * nb * bc;
    size_t size_B = size_t(ldb) * nb * bc;
    size_t size_C = size_t(ldc) * nb * bc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T*)nullptr,
                                                                  lda, (T*)nullptr, ldb, (T*)nullptr,
                                                                  ldc, (rocblas_int*)nullptr, bc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T*)nullptr, lda,
                                                              (T*)nullptr, ldb, (T*)nullptr, ldc,
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

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, max(1, nblocks));
    host_strided_batch_vector<T> hB(size_B, 1, size_B, max(1, nblocks));
    host_strided_batch_vector<T> hC(size_C, 1, size_C, max(1, nblocks));
    host_strided_batch_vector<T> hBRes(size_BRes, 1, size_BRes, max(1, nblocks));
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, max(1, nblocks));
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, max(1, nblocks));
    device_strided_batch_vector<T> dB(size_B, 1, size_B, max(1, nblocks));
    device_strided_batch_vector<T> dC(size_C, 1, size_C, max(1, nblocks));
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_B)
        CHECK_HIP_ERROR(dB.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(nb == 0 || nblocks == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA.data(),
                                                                  lda, dB.data(), ldb, dC.data(),
                                                                  ldc, dInfo.data(), bc),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrf_npvt_interleaved_getError<T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, dInfo,
                                              bc, hA, hB, hBRes, hC, hCRes, hInfo, hInfoRes,
                                              &max_error, argus.singular);

    // collect performance data
    if(argus.timing)
        geblttrf_npvt_interleaved_getPerfData<T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc,
                                                 dInfo, bc, hA, hB, hC, &gpu_time_used,
                                                 &cpu_time_used, hot_calls, argus.profile,
                                                 argus.profile_kernels, argus.perf, argus.singular);

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
            rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc", "batch_c");
            rocsolver_bench_output(nb, nblocks, lda, ldb, ldc, bc);
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

#define EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrf_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
