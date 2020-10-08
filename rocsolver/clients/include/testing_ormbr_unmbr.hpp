/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool COMPLEX, typename T>
void ormbr_unmbr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_storev storev,
                              const rocblas_side side,
                              const rocblas_operation trans,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv,
                              T dC,
                              const rocblas_int ldc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormbr_unmbr(nullptr, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side(-1), trans, m, n, k,
                                                dA, lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, rocblas_storev(-1), side, trans, m, n, k,
                                                dA, lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, rocblas_operation(-1), m, n,
                                                k, dA, lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, rocblas_operation_transpose,
                                                    m, n, k, dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side,
                                                    rocblas_operation_conjugate_transpose, m, n, k,
                                                    dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, (T) nullptr,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA, lda, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA, lda,
                                                dIpiv, (T) nullptr, ldc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_left, trans, 0, n, k,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_right, trans, m, 0, k,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_left, trans, m, n, 0,
                                                (T) nullptr, lda, (T) nullptr, dC, ldc),
                          rocblas_status_success);
}

template <typename T, bool COMPLEX = is_complex<T>>
void testing_ormbr_unmbr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldc = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormbr_unmbr_checkBadArgs<COMPLEX>(handle, storev, side, trans, m, n, k, dA.data(), lda,
                                      dIpiv.data(), dC.data(), ldc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormbr_unmbr_initData(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Td& dC,
                          const rocblas_int ldc,
                          Th& hA,
                          Th& hIpiv,
                          Th& hC,
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
        rocblas_int nq = (side == rocblas_side_left) ? m : n;

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        // and compute gebrd
        if(storev == rocblas_column_wise)
        {
            for(int i = 0; i < nq; ++i)
            {
                for(int j = 0; j < s; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cblas_gebrd<S, T>(nq, s, hA[0], lda, D.data(), E.data(), hIpiv[0], P.data(), hW.data(),
                              size_W);
        }
        else
        {
            for(int i = 0; i < s; ++i)
            {
                for(int j = 0; j < nq; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cblas_gebrd<S, T>(s, nq, hA[0], lda, D.data(), E.data(), P.data(), hIpiv[0], hW.data(),
                              size_W);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Td, typename Th>
void ormbr_unmbr_getError(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Td& dC,
                          const rocblas_int ldc,
                          Th& hA,
                          Th& hIpiv,
                          Th& hC,
                          Th& hCr,
                          double* max_err)
{
    size_t size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    ormbr_unmbr_initData<true, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC,
                                        ldc, hA, hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(), lda,
                                              dIpiv.data(), dC.data(), ldc));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    cblas_ormbr_unmbr<T>(storev, side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(),
                         size_W);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <typename T, typename Td, typename Th>
void ormbr_unmbr_getPerfData(const rocblas_handle handle,
                             const rocblas_storev storev,
                             const rocblas_side side,
                             const rocblas_operation trans,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int k,
                             Td& dA,
                             const rocblas_int lda,
                             Td& dIpiv,
                             Td& dC,
                             const rocblas_int ldc,
                             Th& hA,
                             Th& hIpiv,
                             Th& hC,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const bool perf)
{
    size_t size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        ormbr_unmbr_initData<true, false, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_ormbr_unmbr<T>(storev, side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc,
                             hW.data(), size_W);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    ormbr_unmbr_initData<true, false, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC,
                                         ldc, hA, hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormbr_unmbr_initData<false, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(),
                                                  lda, dIpiv.data(), dC.data(), ldc));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        ormbr_unmbr_initData<false, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        start = get_time_us();
        rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(), lda, dIpiv.data(),
                              dC.data(), ldc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool COMPLEX = is_complex<T>>
void testing_ormbr_unmbr(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int k = argus.K;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldc = argus.ldc;
    rocblas_int hot_calls = argus.iters;
    char storevC = argus.storev;
    char sideC = argus.side_option;
    char transC = argus.transA_option;
    rocblas_side side = char2rocblas_side(sideC);
    rocblas_storev storev = char2rocblas_storev(storevC);
    rocblas_operation trans = char2rocblas_operation(transC);

    // check non-supported values
    bool invalid_value
        = (side == rocblas_side_both || (COMPLEX && trans == rocblas_operation_transpose)
           || (!COMPLEX && trans == rocblas_operation_conjugate_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k,
                                                    (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    rocblas_int nq = left ? m : n;
    size_t size_P = size_t(min(nq, k));
    size_t size_C = size_t(ldc) * n;

    bool row = (storev == rocblas_row_wise);
    size_t size_A = row ? size_t(lda) * nq : size_t(lda) * size_P;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size
        = ((m < 0 || n < 0 || k < 0 || ldc < m) || (row && lda < min(nq, k)) || (!row && lda < nq));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k,
                                                    (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCr(size_Cr, 1, size_Cr, 1);
    host_strided_batch_vector<T> hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T> dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());

    // check quick return
    if(n == 0 || m == 0 || k == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(),
                                                    lda, dIpiv.data(), dC.data(), ldc),
                              rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormbr_unmbr_getError<T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                hIpiv, hC, hCr, &max_error);

    // collect performance data
    if(argus.timing)
        ormbr_unmbr_getPerfData<T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                   hIpiv, hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    rocblas_int s = left ? m : n;
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
            rocsolver_bench_output("storev", "side", "trans", "m", "n", "k", "lda", "ldc");
            rocsolver_bench_output(storevC, sideC, transC, m, n, k, lda, ldc);

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
