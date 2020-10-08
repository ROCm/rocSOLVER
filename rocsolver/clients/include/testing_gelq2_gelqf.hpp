/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, bool GELQF, typename T, typename U>
void gelq2_gelqf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, nullptr, m, n, dA, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, dIpiv, stP, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T) nullptr, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, (U) nullptr, stP, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, 0, n, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, 0, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, dIpiv, stP, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GELQF, typename T>
void testing_gelq2_gelqf_bad_arg()
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
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gelq2_gelqf_checkBadArgs<STRIDED, GELQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gelq2_gelqf_checkBadArgs<STRIDED, GELQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
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
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GELQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_getError(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          double* max_err)
{
    std::vector<T> hW(m);

    // input data initialization
    gelq2_gelqf_initData<true, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GELQF ? cblas_gelqf<T>(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
              : cblas_gelq2<T>(m, n, hA[b], lda, hIpiv[b], hW.data());
    }

    // error is ||hA - hARes|| / ||hA|| (ideally ||QR - Qres Rres|| / ||QR||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool GELQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const bool perf)
{
    std::vector<T> hW(m);

    if(!perf)
    {
        gelq2_gelqf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GELQF ? cblas_gelqf<T>(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
                  : cblas_gelq2<T>(m, n, hA[b], lda, hIpiv[b], hW.data());
        }
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    gelq2_gelqf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gelq2_gelqf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA,
                                                  dIpiv.data(), stP, bc));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        gelq2_gelqf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        start = get_time_us();
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP,
                              bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GELQF, typename T>
void testing_gelq2_gelqf(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stP = argus.bsp;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T*)nullptr,
                                                        lda, stA, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gelq2_gelqf_getError<STRIDED, GELQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                                    hARes, hIpiv, &max_error);

        // collect performance data
        if(argus.timing)
            gelq2_gelqf_getPerfData<STRIDED, GELQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc,
                                                       hA, hIpiv, &gpu_time_used, &cpu_time_used,
                                                       hot_calls, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gelq2_gelqf_getError<STRIDED, GELQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                                    hARes, hIpiv, &max_error);

        // collect performance data
        if(argus.timing)
            gelq2_gelqf_getPerfData<STRIDED, GELQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc,
                                                       hA, hIpiv, &gpu_time_used, &cpu_time_used,
                                                       hot_calls, argus.perf);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    // (for possibly singular of ill-conditioned matrices we could use n*min(m,n))
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
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
