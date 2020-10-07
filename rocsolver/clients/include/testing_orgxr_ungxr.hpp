/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool GQR, typename T>
void orgxr_ungxr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxr_ungxr(GQR, nullptr, m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxr_ungxr(GQR, handle, m, n, k, (T) nullptr, lda, dIpiv),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxr_ungxr(GQR, handle, m, n, k, dA, lda, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxr_ungxr(GQR, handle, 0, 0, 0, (T) nullptr, lda, (T) nullptr),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxr_ungxr(GQR, handle, m, 0, 0, (T) nullptr, lda, (T) nullptr),
                          rocblas_status_success);
}

template <typename T, bool GQR>
void testing_orgxr_ungxr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
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
    orgxr_ungxr_checkBadArgs<GQR>(handle, m, n, k, dA.data(), lda, dIpiv.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgxr_ungxr_initData(const rocblas_handle handle,
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
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
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

        // compute QR factorization
        cblas_geqrf<T>(m, n, hA[0], lda, hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool GQR, typename T, typename Td, typename Th>
void orgxr_ungxr_getError(const rocblas_handle handle,
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
    size_t size_W = size_t(n);
    std::vector<T> hW(size_W);

    // initialize data
    orgxr_ungxr_initData<true, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_orgxr_ungxr(GQR, handle, m, n, k, dA.data(), lda, dIpiv.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    GQR ? cblas_orgqr_ungqr<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W)
        : cblas_org2r_ung2r<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data());

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <bool GQR, typename T, typename Td, typename Th>
void orgxr_ungxr_getPerfData(const rocblas_handle handle,
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
    size_t size_W = size_t(n);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgxr_ungxr_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        GQR ? cblas_orgqr_ungqr<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W)
            : cblas_org2r_ung2r<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data());
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    orgxr_ungxr_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgxr_ungxr_initData<false, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_orgxr_ungxr(GQR, handle, m, n, k, dA.data(), lda, dIpiv.data()));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgxr_ungxr_initData<false, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us();
        rocsolver_orgxr_ungxr(GQR, handle, m, n, k, dA.data(), lda, dIpiv.data());
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool GQR>
void testing_orgxr_ungxr(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int k = argus.K;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 0 || lda < m || n > m || k > n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgxr_ungxr(GQR, handle, m, n, k, (T*)nullptr, lda, (T*)nullptr),
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
            rocsolver_orgxr_ungxr(GQR, handle, m, n, k, dA.data(), lda, dIpiv.data()),
            rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgxr_ungxr_getError<GQR, T>(handle, m, n, k, dA, lda, dIpiv, hA, hAr, hIpiv, &max_error);

    // collect performance data
    if(argus.timing)
        orgxr_ungxr_getPerfData<GQR, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, &gpu_time_used,
                                        &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, m);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("m", "n", "k", "lda");
            rocsolver_bench_output(m, n, k, lda);

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
