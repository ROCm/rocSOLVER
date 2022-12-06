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

template <bool STRIDED, typename T, typename U>
void bttrf_npvt_checkBadArgs(const rocblas_handle handle,
                             const rocblas_int nb,
                             const rocblas_int nblocks,
                             T dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             T dB,
                             const rocblas_int ldb,
                             const rocblas_stride stB,
                             T dC,
                             const rocblas_int ldc,
                             const rocblas_stride stC,
                             U dInfo,
                             const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, nullptr, nb, nblocks, dA, lda, stA, dB, ldb,
                                               stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB,
                                                   ldb, stB, dC, ldc, stC, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, (T) nullptr, lda, stA,
                                               dB, ldb, stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA,
                                               (T) nullptr, ldb, stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB, ldb,
                                               stB, (T) nullptr, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB, ldb,
                                               stB, dC, ldc, stC, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, 0, nblocks, (T) nullptr, lda, stA,
                                               (T) nullptr, ldb, stB, (T) nullptr, ldc, stC, dInfo,
                                               bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, 0, (T) nullptr, lda, stA,
                                               (T) nullptr, ldb, stB, (T) nullptr, ldc, stC, dInfo,
                                               bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB,
                                                   ldb, stB, dC, ldc, stC, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_bttrf_npvt_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_stride stA = 2;
    rocblas_stride stB = 2;
    rocblas_stride stC = 2;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_batch_vector<T> dC(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        bttrf_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, dA.data(), lda, stA, dB.data(), ldb,
                                         stB, dC.data(), ldc, stC, dInfo.data(), bc);
    }
    else
    {
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
        bttrf_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, dA.data(), lda, stA, dB.data(), ldb,
                                         stB, dC.data(), ldc, stC, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void bttrf_npvt_initData(const rocblas_handle handle,
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

            // TODO: add singularities to the matrix
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

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void bttrf_npvt_getError(const rocblas_handle handle,
                         const rocblas_int nb,
                         const rocblas_int nblocks,
                         Td& dA,
                         const rocblas_int lda,
                         const rocblas_stride stA,
                         Td& dB,
                         const rocblas_int ldb,
                         const rocblas_stride stB,
                         Td& dC,
                         const rocblas_int ldc,
                         const rocblas_stride stC,
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
    bttrf_npvt_initData<true, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA, hB,
                                       hC, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda, stA,
                                             dB.data(), ldb, stB, dC.data(), ldc, stC, dInfo.data(),
                                             bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        //cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] != 0)
            err++;
    }
    *max_err += err;

    // error is ||hB - hBRes|| / ||hB|| or ||hC - hCRes|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // TODO: Complete the test error calculation

        err = norm_error('F', nb, nb * nblocks, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;

        err = norm_error('F', nb, nb * (nblocks - 1), ldc, hC[b], hCRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th>
void bttrf_npvt_getPerfData(const rocblas_handle handle,
                            const rocblas_int nb,
                            const rocblas_int nblocks,
                            Td& dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            Td& dB,
                            const rocblas_int ldb,
                            const rocblas_stride stB,
                            Td& dC,
                            const rocblas_int ldc,
                            const rocblas_stride stC,
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
        // bttrf_npvt_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
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

    bttrf_npvt_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA, hB,
                                        hC, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        bttrf_npvt_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
                                            hB, hC, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda, stA,
                                                 dB.data(), ldb, stB, dC.data(), ldc, stC,
                                                 dInfo.data(), bc));
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
        bttrf_npvt_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
                                            hB, hC, singular);

        start = get_time_us_sync(stream);
        rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda, stA, dB.data(), ldb, stB,
                             dC.data(), ldc, stC, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_bttrf_npvt(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * nb * nblocks);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nb * nblocks);
    rocblas_stride stC = argus.get<rocblas_stride>("strideC", ldc * nb * nblocks);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    rocblas_stride stCRes = (argus.unit_check || argus.norm_check) ? stC : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * nb * nblocks;
    size_t size_B = size_t(ldb) * nb * nblocks;
    size_t size_C = size_t(ldc) * nb * nblocks;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, (T* const*)nullptr, lda, stA,
                                     (T* const*)nullptr, ldb, stB, (T* const*)nullptr, ldc, stC,
                                     (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, (T*)nullptr,
                                                       lda, stA, (T*)nullptr, ldb, stB, (T*)nullptr,
                                                       ldc, stC, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_bttrf_npvt(
                STRIDED, handle, nb, nblocks, (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb,
                stB, (T* const*)nullptr, ldc, stC, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, (T*)nullptr, lda,
                                                   stA, (T*)nullptr, ldb, stB, (T*)nullptr, ldc,
                                                   stC, (rocblas_int*)nullptr, bc));

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
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_batch_vector<T> hCRes(size_CRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dC(size_C, 1, bc);
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
            EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda,
                                                       stA, dB.data(), ldb, stB, dC.data(), ldc,
                                                       stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            bttrf_npvt_getError<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                            ldc, stC, dInfo, bc, hA, hB, hBRes, hC, hCRes, hInfo,
                                            hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            bttrf_npvt_getPerfData<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                               ldc, stC, dInfo, bc, hA, hB, hC, &gpu_time_used,
                                               &cpu_time_used, hot_calls, argus.profile,
                                               argus.profile_kernels, argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hC(size_C, 1, stC, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<T> hCRes(size_CRes, 1, stCRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dC(size_C, 1, stC, bc);
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
            EXPECT_ROCBLAS_STATUS(rocsolver_bttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda,
                                                       stA, dB.data(), ldb, stB, dC.data(), ldc,
                                                       stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            bttrf_npvt_getError<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                            ldc, stC, dInfo, bc, hA, hB, hBRes, hC, hCRes, hInfo,
                                            hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            bttrf_npvt_getPerfData<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                               ldc, stC, dInfo, bc, hA, hB, hC, &gpu_time_used,
                                               &cpu_time_used, hot_calls, argus.profile,
                                               argus.profile_kernels, argus.perf, argus.singular);
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
                rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc", "batch_c");
                rocsolver_bench_output(nb, nblocks, lda, ldb, ldc, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("nb", "nblocks", "lda", "strideA", "ldb", "strideB", "ldc",
                                       "strideC", "batch_c");
                rocsolver_bench_output(nb, nblocks, lda, stA, ldb, stB, ldc, stC, bc);
            }
            else
            {
                rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc");
                rocsolver_bench_output(nb, nblocks, lda, ldb, ldc);
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

#define EXTERN_TESTING_BTTRF_NPVT(...) \
    extern template void testing_bttrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_BTTRF_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
