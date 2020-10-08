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
void larfb_checkBadArgs(const rocblas_handle handle,
                        const rocblas_side side,
                        const rocblas_operation trans,
                        const rocblas_direct direct,
                        const rocblas_storev storev,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int k,
                        T dV,
                        const rocblas_int ldv,
                        T dT,
                        const rocblas_int ldt,
                        T dA,
                        const rocblas_int lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_larfb(nullptr, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA, lda),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side(-1), trans, direct, storev, m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, rocblas_operation(-1), direct, storev, m, n,
                                          k, dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, rocblas_direct(-1), storev, m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, rocblas_storev(-1), m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, (T) nullptr,
                                          ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV, ldv,
                                          (T) nullptr, ldt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                          ldt, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side_left, trans, direct, storev, 0, n, k,
                                          (T) nullptr, ldv, dT, ldt, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side_right, trans, direct, storev, m, 0,
                                          k, (T) nullptr, ldv, dT, ldt, (T) nullptr, lda),
                          rocblas_status_success);
}

template <typename T>
void testing_larfb_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_direct direct = rocblas_forward_direction;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int ldv = 1;
    rocblas_int ldt = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dT(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dT.memcheck());
    CHECK_HIP_ERROR(dA.memcheck());

    // check bad arguments
    larfb_checkBadArgs(handle, side, trans, direct, storev, m, n, k, dV.data(), ldv, dT.data(), ldt,
                       dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void larfb_initData(const rocblas_handle handle,
                    const rocblas_side side,
                    const rocblas_operation trans,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hV,
                    Th& hT,
                    Th& hA,
                    std::vector<T>& hW,
                    size_t sizeW)
{
    if(CPU)
    {
        bool left = (side == rocblas_side_left);
        bool forward = (direct == rocblas_forward_direction);
        bool column = (storev == rocblas_column_wise);
        std::vector<T> htau(k);

        rocblas_init<T>(hV, true);
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hT, true);

        // scale to avoid singularities
        // create householder reflectors and triangular factor
        if(left)
        {
            if(column)
            {
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < k; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cblas_geqrf<T>(m, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cblas_geqlf<T>(m, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }
            else
            {
                for(int i = 0; i < k; ++i)
                {
                    for(int j = 0; j < m; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cblas_gelqf<T>(k, m, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cblas_gerqf<T>(k, m, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }

            cblas_larft<T>(direct, storev, m, k, hV[0], ldv, htau.data(), hT[0], ldt);
        }
        else
        {
            if(column)
            {
                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < k; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cblas_geqrf<T>(n, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cblas_geqlf<T>(n, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }
            else
            {
                for(int i = 0; i < k; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cblas_gelqf<T>(k, n, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cblas_gerqf<T>(k, n, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }

            cblas_larft<T>(direct, storev, n, k, hV[0], ldv, htau.data(), hT[0], ldt);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dT.transfer_from(hT));
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Th>
void larfb_getError(const rocblas_handle handle,
                    const rocblas_side side,
                    const rocblas_operation trans,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hV,
                    Th& hT,
                    Th& hA,
                    Th& hAr,
                    double* max_err)
{
    bool left = (side == rocblas_side_left);
    rocblas_int ldw = left ? n : m;
    size_t sizeW = size_t(ldw) * k;
    std::vector<T> hW(sizeW);

    // initialize data
    larfb_initData<true, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt,
                                  dA, lda, hV, hT, hA, hW, sizeW);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(),
                                        ldv, dT.data(), ldt, dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_larfb<T>(side, trans, direct, storev, m, n, k, hV[0], ldv, hT[0], ldt, hA[0], lda,
                   hW.data(), ldw);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void larfb_getPerfData(const rocblas_handle handle,
                       const rocblas_side side,
                       const rocblas_operation trans,
                       const rocblas_direct direct,
                       const rocblas_storev storev,
                       const rocblas_int m,
                       const rocblas_int n,
                       const rocblas_int k,
                       Td& dV,
                       const rocblas_int ldv,
                       Td& dT,
                       const rocblas_int ldt,
                       Td& dA,
                       const rocblas_int lda,
                       Th& hV,
                       Th& hT,
                       Th& hA,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    bool left = (side == rocblas_side_left);
    rocblas_int ldw = left ? n : m;
    size_t sizeW = size_t(ldw) * k;
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        larfb_initData<true, false, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_larfb<T>(side, trans, direct, storev, m, n, k, hV[0], ldv, hT[0], ldt, hA[0], lda,
                       hW.data(), ldw);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    larfb_initData<true, false, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt,
                                   dA, lda, hV, hT, hA, hW, sizeW);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larfb_initData<false, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        CHECK_ROCBLAS_ERROR(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(),
                                            ldv, dT.data(), ldt, dA.data(), lda));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        larfb_initData<false, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        start = get_time_us();
        rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(), ldv, dT.data(),
                        ldt, dA.data(), lda);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_larfb(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int k = argus.K;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int ldv = argus.ldv;
    rocblas_int lda = argus.lda;
    rocblas_int ldt = argus.ldt;
    rocblas_int hot_calls = argus.iters;
    char sideC = argus.side_option;
    char transC = argus.transH_option;
    char directC = argus.direct_option;
    char storevC = argus.storev;
    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_direct direct = char2rocblas_direct(directC);
    rocblas_storev storev = char2rocblas_storev(storevC);

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              (T*)nullptr, ldv, (T*)nullptr, ldt, (T*)nullptr, lda),
                              rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    bool row = (storev == rocblas_row_wise);
    bool left = (side == rocblas_side_left);

    size_t size_V = size_t(ldv) * k;
    if(row)
        size_V = left ? size_t(ldv) * m : size_t(ldv) * n;

    size_t size_T = size_t(ldt) * k;
    size_t size_A = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 1 || ldt < k || lda < m || (row && ldv < k)
                         || (!row && !left && ldv < n) || (!row && left && ldv < m));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              (T*)nullptr, ldv, (T*)nullptr, ldt, (T*)nullptr, lda),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hT(size_T, 1, size_T, 1);
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dT(size_T, 1, size_T, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_T)
        CHECK_HIP_ERROR(dT.memcheck());
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              dV.data(), ldv, dT.data(), ldt, dA.data(), lda),
                              rocblas_status_success);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larfb_getError<T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA, lda,
                          hV, hT, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        larfb_getPerfData<T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA,
                             lda, hV, hT, hA, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

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
            rocsolver_bench_output("side", "trans", "direct", "storev", "m", "n", "k", "ldv", "ldt",
                                   "lda");
            rocsolver_bench_output(sideC, transC, directC, storevC, m, n, k, ldv, ldt, lda);

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
