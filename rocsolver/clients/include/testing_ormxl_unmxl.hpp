/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool MQL, bool COMPLEX, typename T>
void ormxl_unmxl_checkBadArgs(const rocblas_handle handle,
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
        rocsolver_ormxl_unmxl(MQL, nullptr, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, rocblas_side(-1), trans, m, n, k, dA,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side, rocblas_operation(-1), m, n, k,
                                                dA, lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side, rocblas_operation_transpose,
                                                    m, n, k, dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side,
                                                    rocblas_operation_conjugate_transpose, m, n, k,
                                                    dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, (T) nullptr, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA, lda, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA, lda, dIpiv, (T) nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, rocblas_side_right, trans, 0, n, k, dA,
                                                lda, dIpiv, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, rocblas_side_left, trans, m, 0, k, dA,
                                                lda, dIpiv, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, rocblas_side_left, trans, m, n, 0,
                                                (T) nullptr, lda, (T) nullptr, dC, ldc),
                          rocblas_status_success);
}

template <typename T, bool MQL, bool COMPLEX = is_complex<T>>
void testing_ormxl_unmxl_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
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
    ormxl_unmxl_checkBadArgs<MQL, COMPLEX>(handle, side, trans, m, n, k, dA.data(), lda,
                                           dIpiv.data(), dC.data(), ldc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormxl_unmxl_initData(const rocblas_handle handle,
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
        rocblas_int nq = (side == rocblas_side_left) ? m : n;

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        for(int i = 0; i < nq; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                if(m - i == n - j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute QL factorization
        cblas_geqlf<T>(nq, k, hA[0], lda, hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool MQL, typename T, typename Td, typename Th>
void ormxl_unmxl_getError(const rocblas_handle handle,
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
    ormxl_unmxl_initData<true, true, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                        hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA.data(), lda,
                                              dIpiv.data(), dC.data(), ldc));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    MQL ? cblas_ormql_unmql<T>(side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(),
                               size_W)
        : cblas_orm2l_unm2l<T>(side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data());

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <bool MQL, typename T, typename Td, typename Th>
void ormxl_unmxl_getPerfData(const rocblas_handle handle,
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
        ormxl_unmxl_initData<true, false, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc,
                                             hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        MQL ? cblas_ormql_unmql<T>(side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc,
                                   hW.data(), size_W)
            : cblas_orm2l_unm2l<T>(side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data());
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    ormxl_unmxl_initData<true, false, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                         hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormxl_unmxl_initData<false, true, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc,
                                             hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA.data(), lda,
                                                  dIpiv.data(), dC.data(), ldc));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        ormxl_unmxl_initData<false, true, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc,
                                             hA, hIpiv, hC, hW, size_W);

        start = get_time_us();
        rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA.data(), lda, dIpiv.data(),
                              dC.data(), ldc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool MQL, bool COMPLEX = is_complex<T>>
void testing_ormxl_unmxl(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int k = argus.K;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldc = argus.ldc;
    rocblas_int hot_calls = argus.iters;
    char sideC = argus.side_option;
    char transC = argus.transA_option;
    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);

    // check non-supported values
    bool invalid_value
        = (side == rocblas_side_both || (COMPLEX && trans == rocblas_operation_transpose)
           || (!COMPLEX && trans == rocblas_operation_conjugate_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, (T*)nullptr,
                                                    lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    size_t size_A = size_t(lda) * k;
    size_t size_P = size_t(k);
    size_t size_C = size_t(ldc) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = ((m < 0 || n < 0 || k < 0 || ldc < m) || (left && lda < m)
                         || (!left && lda < n) || (left && k > m) || (!left && k > n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, (T*)nullptr,
                                                    lda, (T*)nullptr, (T*)nullptr, ldc),
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
        EXPECT_ROCBLAS_STATUS(rocsolver_ormxl_unmxl(MQL, handle, side, trans, m, n, k, dA.data(),
                                                    lda, dIpiv.data(), dC.data(), ldc),
                              rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormxl_unmxl_getError<MQL, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                     hIpiv, hC, hCr, &max_error);

    // collect performance data
    if(argus.timing)
        ormxl_unmxl_getPerfData<MQL, T>(handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                        hIpiv, hC, &gpu_time_used, &cpu_time_used, hot_calls,
                                        argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
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
            rocsolver_bench_output("side", "trans", "m", "n", "k", "lda", "ldc");
            rocsolver_bench_output(sideC, transC, m, n, k, lda, ldc);

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
