/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename S, typename T, typename U>
void steqr_checkBadArgs(const rocblas_handle handle,
                        const rocblas_evect compc,
                        const rocblas_int n,
                        S dD,
                        S dE,
                        T dC,
                        const rocblas_int ldc,
                        U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(nullptr, compc, n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, rocblas_evect(-1), n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, compc, n, (S) nullptr, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, compc, n, dD, (S) nullptr, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, compc, n, dD, dE, (T) nullptr, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, compc, n, dD, dE, dC, ldc, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_steqr(handle, compc, 0, (S) nullptr, (S) nullptr, (T) nullptr, ldc, dInfo),
        rocblas_status_success);
}

template <typename S, typename T>
void testing_steqr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect compc = rocblas_evect_original;
    rocblas_int n = 1;
    rocblas_int ldc = 1;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    steqr_checkBadArgs(handle, compc, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
}

template <bool CPU, bool GPU, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_initData(const rocblas_handle handle,
                    const rocblas_evect compc,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Th& hA,
                    Sh& hD,
                    Sh& hE,
                    Th& hC,
                    Uh& hInfo,
                    std::vector<T>& hW,
                    size_t size_W)
{
    if(CPU)
    {
        rocblas_int lda = n;
        std::vector<T> ipiv(n - 1);

        rocblas_init<T>(hA, true);

        if(compc == rocblas_evect_none)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }

            // compute sytrd/hetrd on A
            cblas_sytrd_hetrd<S, T>(rocblas_fill_upper, n, hA[0], lda, hD[0], hE[0], ipiv.data(),
                                    hW.data(), size_W);

            // C is ignored
        }
        else
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hC[0][i + j * ldc] = hA[0][i + j * lda] += 400;
                    else
                        hC[0][i + j * ldc] = hA[0][i + j * lda] -= 4;
                }
            }

            // compute sytrd/hetrd on C
            cblas_sytrd_hetrd<S, T>(rocblas_fill_upper, n, hC[0], ldc, hD[0], hE[0], ipiv.data(),
                                    hW.data(), size_W);

            if(compc == rocblas_evect_original)
            {
                // A is the original matrix
                cblas_orgtr_ungtr<T>(rocblas_fill_upper, n, hC[0], ldc, ipiv.data(), hW.data(),
                                     size_W);
            }
            else
            {
                // A is the tridiagonal matrix
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        if(i + 1 >= j)
                            hA[0][i + j * lda] = hC[0][i + j * ldc];
                        else
                            hA[0][i + j * lda] = 0;
                    }
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(compc == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_getError(const rocblas_handle handle,
                    const rocblas_evect compc,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Th& hA,
                    Sh& hD,
                    Sh& hDRes,
                    Sh& hE,
                    Sh& hERes,
                    Th& hC,
                    Th& hCRes,
                    Uh& hInfo,
                    double* max_err)
{
    size_t size_W = n * 32;
    std::vector<T> hW(size_W);

    // input data initialization
    steqr_initData<true, true, S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC,
                                     hInfo, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_steqr(handle, compc, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    if(compc != rocblas_evect_none)
        CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // CPU lapack
    cblas_sterf<S>(n, hD[0], hE[0]);

    // error is ||hD - hDRes|| / ||hD||
    // using frobenius norm
    double err;
    *max_err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
    if(compc != rocblas_evect_none)
    {
        rocblas_int lda = n;
        T alpha;
        T beta = 0;
        for(int j = 0; j < n; j++)
        {
            alpha = T(1) / hD[0][j];
            cblas_symv_hemv(rocblas_fill_upper, n, alpha, hA[0], lda, hCRes[0] + j * ldc, 1, beta,
                            hC[0] + j * ldc, 1);
        }
        err = norm_error('F', n, n, ldc, hC[0], hCRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_getPerfData(const rocblas_handle handle,
                       const rocblas_evect compc,
                       const rocblas_int n,
                       Sd& dD,
                       Sd& dE,
                       Td& dC,
                       const rocblas_int ldc,
                       Ud& dInfo,
                       Th& hA,
                       Sh& hD,
                       Sh& hE,
                       Th& hC,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    size_t size_W = n * 32;
    size_t size_W2 = (compc == rocblas_evect_none ? 0 : 2 * n - 2);
    std::vector<T> hW(size_W);
    std::vector<S> hW2(size_W2);

    if(!perf)
    {
        steqr_initData<true, false, S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC,
                                          hInfo, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_steqr<S, T>(compc, n, hD[0], hE[0], hC[0], ldc, hW2.data());
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    steqr_initData<true, false, S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC,
                                      hInfo, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        steqr_initData<false, true, S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC,
                                          hInfo, hW, size_W);

        CHECK_ROCBLAS_ERROR(
            rocsolver_steqr(handle, compc, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        steqr_initData<false, true, S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC,
                                          hInfo, hW, size_W);

        start = get_time_us();
        rocsolver_steqr(handle, compc, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename S, typename T>
void testing_steqr(Arguments argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.N;
    rocblas_int ldc = argus.ldc;
    rocblas_int hot_calls = argus.iters;
    char compcC = argus.evect;
    rocblas_evect compc = char2rocblas_evect(compcC);

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = n;
    size_t size_E = n;
    size_t size_C = ldc * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || (compc != rocblas_evect_none && ldc < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, compc, n, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldc, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<S> hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_steqr(handle, compc, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()),
            rocblas_status_success);
        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        steqr_getError<S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hDRes, hE, hERes, hC,
                             hCRes, hInfo, &max_error);

    // collect performance data
    if(argus.timing)
        steqr_getPerfData<S, T>(handle, compc, n, dD, dE, dC, ldc, dInfo, hA, hD, hE, hC, hInfo,
                                &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
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
            rocsolver_bench_output("compc", "n", "ldc");
            rocsolver_bench_output(compcC, n, ldc);

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
