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
void larf_checkBadArgs(const rocblas_handle handle,
                       const rocblas_side side,
                       const rocblas_int m,
                       const rocblas_int n,
                       T dx,
                       const rocblas_int inc,
                       T dt,
                       T dA,
                       const rocblas_int lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(nullptr, side, m, n, dx, inc, dt, dA, lda),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_both, m, n, dx, inc, dt, dA, lda),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, (T) nullptr, inc, dt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, dx, inc, (T) nullptr, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, dx, inc, dt, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_left, 0, n, (T) nullptr, inc,
                                         (T) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_right, m, 0, (T) nullptr, inc,
                                         (T) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
}

template <typename T>
void testing_larf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int inc = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dx(1, 1, 1, 1);
    device_strided_batch_vector<T> dt(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dt.memcheck());

    // check bad arguments
    larf_checkBadArgs(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void larf_initData(const rocblas_handle handle,
                   const rocblas_side side,
                   const rocblas_int m,
                   const rocblas_int n,
                   Td& dx,
                   const rocblas_int inc,
                   Td& dt,
                   Td& dA,
                   const rocblas_int lda,
                   Th& xx,
                   Th& hx,
                   Th& ht,
                   Th& hA)
{
    if(CPU)
    {
        rocblas_int order = xx.n();

        rocblas_init<T>(hA, true);
        rocblas_init<T>(xx, true);

        // compute householder reflector
        cblas_larfg<T>(order, xx[0], xx[0] + abs(inc), abs(inc), ht[0]);
        xx[0][0] = 1;
        for(rocblas_int i = 0; i < order; i++)
        {
            if(inc < 0)
                hx[0][i * abs(inc)] = xx[0][(order - 1 - i) * abs(inc)];
            else
                hx[0][i * inc] = xx[0][i * inc];
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dt.transfer_from(ht));
    }
}

template <typename T, typename Td, typename Th>
void larf_getError(const rocblas_handle handle,
                   const rocblas_side side,
                   const rocblas_int m,
                   const rocblas_int n,
                   Td& dx,
                   const rocblas_int inc,
                   Td& dt,
                   Td& dA,
                   const rocblas_int lda,
                   Th& xx,
                   Th& hx,
                   Th& ht,
                   Th& hA,
                   Th& hAr,
                   double* max_err)
{
    size_t size_w = (side == rocblas_side_left) ? size_t(n) : size_t(m);
    std::vector<T> hw(size_w);

    // initialize data
    larf_initData<true, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_larf<T>(side, m, n, hx[0], inc, ht[0], hA[0], lda, hw.data());

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void larf_getPerfData(const rocblas_handle handle,
                      const rocblas_side side,
                      const rocblas_int m,
                      const rocblas_int n,
                      Td& dx,
                      const rocblas_int inc,
                      Td& dt,
                      Td& dA,
                      const rocblas_int lda,
                      Th& xx,
                      Th& hx,
                      Th& ht,
                      Th& hA,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const bool perf)
{
    size_t size_w = (side == rocblas_side_left) ? size_t(n) : size_t(m);
    std::vector<T> hw(size_w);

    if(!perf)
    {
        larf_initData<true, false, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_larf<T>(side, m, n, hx[0], inc, ht[0], hA[0], lda, hw.data());
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    larf_initData<true, false, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larf_initData<false, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        CHECK_ROCBLAS_ERROR(
            rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        larf_initData<false, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        start = get_time_us();
        rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_larf(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int inc = argus.incx;
    rocblas_int lda = argus.lda;
    rocblas_int hot_calls = argus.iters;
    char sideC = argus.side_option;
    rocblas_side side = char2rocblas_side(sideC);

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, (T*)nullptr, inc, (T*)nullptr, (T*)nullptr, lda),
            rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    size_t size_A = size_t(lda) * n;
    size_t size_x = left ? size_t(m) : size_t(n);
    size_t stx = size_x * abs(inc);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || !inc || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, (T*)nullptr, inc, (T*)nullptr, (T*)nullptr, lda),
            rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hx(size_x, abs(inc), stx, 1);
    host_strided_batch_vector<T> xx(size_x, abs(inc), stx, 1);
    host_strided_batch_vector<T> ht(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dx(size_x, abs(inc), stx, 1);
    device_strided_batch_vector<T> dt(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_x)
        CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dt.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda),
            rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larf_getError<T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        larf_getPerfData<T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA,
                            &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using size_x * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, size_x);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("side", "m", "n", "inc", "lda");
            rocsolver_bench_output(side, m, n, inc, lda);

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
