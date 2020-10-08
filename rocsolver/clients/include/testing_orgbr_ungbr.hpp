/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T>
void orgbr_ungbr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_storev storev,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(nullptr, storev, m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, rocblas_storev(-1), m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, storev, m, n, k, (T) nullptr, lda, dIpiv),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA, lda, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_orgbr_ungbr(handle, rocblas_row_wise, 0, n, 0, (T) nullptr, lda, (T) nullptr),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_orgbr_ungbr(handle, rocblas_column_wise, m, 0, 0, (T) nullptr, lda, (T) nullptr),
        rocblas_status_success);
}

template <typename T>
void testing_orgbr_ungbr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());

    // check bad arguments
    orgbr_ungbr_checkBadArgs(handle, storev, m, n, k, dA.data(), lda, dIpiv.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgbr_ungbr_initData(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Th& hA,
                          Th& hIpiv,
                          std::vector<T>& hW,
                          size_t size_W)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        size_t s = max(hIpiv.n(), 2);
        std::vector<S> E(s - 1);
        std::vector<S> D(s);
        std::vector<T> P(s);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        // and compute gebrd
        if(storev == rocblas_column_wise)
        {
            for(int i = 0; i < m; ++i)
            {
                for(int j = 0; j < k; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cblas_gebrd<S, T>(m, k, hA[0], lda, D.data(), E.data(), hIpiv[0], P.data(), hW.data(),
                              size_W);
        }
        else
        {
            for(int i = 0; i < k; ++i)
            {
                for(int j = 0; j < n; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cblas_gebrd<S, T>(k, n, hA[0], lda, D.data(), E.data(), P.data(), hIpiv[0], hW.data(),
                              size_W);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <typename T, typename Td, typename Th>
void orgbr_ungbr_getError(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Th& hA,
                          Th& hAr,
                          Th& hIpiv,
                          double* max_err)
{
    size_t size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    orgbr_ungbr_initData<true, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                        size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_orgbr_ungbr<T>(storev, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void orgbr_ungbr_getPerfData(const rocblas_handle handle,
                             const rocblas_storev storev,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int k,
                             Td& dA,
                             const rocblas_int lda,
                             Td& dIpiv,
                             Th& hA,
                             Th& hIpiv,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const bool perf)
{
    size_t size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgbr_ungbr_initData<true, false, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_orgbr_ungbr<T>(storev, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    orgbr_ungbr_initData<true, false, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                         size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        CHECK_ROCBLAS_ERROR(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        start = get_time_us();
        rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data());
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_orgbr_ungbr(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int k = argus.K;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int hot_calls = argus.iters;
    char storevC = argus.storev;
    rocblas_storev storev = char2rocblas_storev(storevC);

    // check non-supported values
    // N/A

    // determine sizes
    // size_P could be zero in test cases that are not quick-return or invalid
    // cases setting it to one to avoid possible memory access errors in the rest
    // of the unit test
    bool row = (storev == rocblas_row_wise);
    size_t size_A = row ? size_t(lda) * n : size_t(lda) * max(n, k);
    size_t size_P = row ? max(size_t(min(n, k)), 1) : max(size_t(min(m, k)), 1);

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = ((m < 0 || n < 0 || k < 0 || lda < m) || (row && (m > n || m < min(n, k)))
                         || (!row && (n > m || n < min(m, k))));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, (T*)nullptr, lda, (T*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dIpiv(size_P, 1, size_P, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()),
            rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgbr_ungbr_getError<T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hAr, hIpiv, &max_error);

    // collect performance data
    if(argus.timing)
        orgbr_ungbr_getPerfData<T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv,
                                   &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    rocblas_int s = row ? n : m;
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("storev", "m", "n", "k", "lda");
            rocsolver_bench_output(storevC, m, n, k, lda);

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
