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
void labrd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int nb,
                        T dA,
                        const rocblas_int lda,
                        S dD,
                        S dE,
                        U dTauq,
                        U dTaup,
                        T dX,
                        const rocblas_int ldx,
                        T dY,
                        const rocblas_int ldy)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(nullptr, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, (T) nullptr, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, (S) nullptr, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, (S) nullptr, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, (U) nullptr, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, (U) nullptr, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, (T) nullptr, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, (T) nullptr, ldy),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, 0, n, 0, (T) nullptr, lda, dD, dE, dTauq, dTaup,
                                          (T) nullptr, ldx, dY, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, 0, 0, (T) nullptr, lda, dD, dE, dTauq, dTaup,
                                          dX, ldx, (T) nullptr, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, 0, dA, lda, (S) nullptr, (S) nullptr,
                                          (U) nullptr, (U) nullptr, (T) nullptr, ldx, (T) nullptr,
                                          ldy),
                          rocblas_status_success);
}

template <typename T>
void testing_labrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int nb = 1;
    rocblas_int lda = 1;
    rocblas_int ldx = 1;
    rocblas_int ldy = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
    device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
    device_strided_batch_vector<T> dX(1, 1, 1, 1);
    device_strided_batch_vector<T> dY(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dTauq.memcheck());
    CHECK_HIP_ERROR(dTaup.memcheck());
    CHECK_HIP_ERROR(dX.memcheck());
    CHECK_HIP_ERROR(dY.memcheck());

    // check bad arguments
    labrd_checkBadArgs(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(), dTauq.data(),
                       dTaup.data(), dX.data(), ldx, dY.data(), ldy);
}

template <bool CPU, bool GPU, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_initData(const rocblas_handle handle,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dD,
                    Sd& dE,
                    Ud& dTauq,
                    Ud& dTaup,
                    Td& dX,
                    const rocblas_int ldx,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Sh& hD,
                    Sh& hE,
                    Uh& hTauq,
                    Uh& hTaup,
                    Th& hX,
                    Th& hY)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < m; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // zero X and Y
        memset(hX[0], 0, ldx * nb * sizeof(T));
        memset(hY[0], 0, ldy * nb * sizeof(T));
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dX.transfer_from(hX));
        CHECK_HIP_ERROR(dY.transfer_from(hY));
    }
}

template <typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_getError(const rocblas_handle handle,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dD,
                    Sd& dE,
                    Ud& dTauq,
                    Ud& dTaup,
                    Td& dX,
                    const rocblas_int ldx,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Th& hARes,
                    Sh& hD,
                    Sh& hE,
                    Uh& hTauq,
                    Uh& hTaup,
                    Th& hX,
                    Th& hXRes,
                    Th& hY,
                    Th& hYRes,
                    double* max_err)
{
    // input data initialization
    labrd_initData<true, true, S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                     ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(),
                                        dTauq.data(), dTaup.data(), dX.data(), ldx, dY.data(), ldy));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));
    CHECK_HIP_ERROR(hYRes.transfer_from(dY));

    // CPU lapack
    cblas_labrd<S, T>(m, n, nb, hA[0], lda, hD[0], hE[0], hTauq[0], hTaup[0], hX[0], ldx, hY[0], ldy);

    // error is max(||hA - hARes|| / ||hA||, ||hX - hXRes|| / ||hX||, ||hY -
    // hYRes|| / ||hY||) (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F', m, n, lda, hA[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', m, nb, ldx, hX[0], hXRes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', n, nb, ldy, hY[0], hYRes[0]);
    *max_err = err > *max_err ? err : *max_err;
}

template <typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_getPerfData(const rocblas_handle handle,
                       const rocblas_int m,
                       const rocblas_int n,
                       const rocblas_int nb,
                       Td& dA,
                       const rocblas_int lda,
                       Sd& dD,
                       Sd& dE,
                       Ud& dTauq,
                       Ud& dTaup,
                       Td& dX,
                       const rocblas_int ldx,
                       Td& dY,
                       const rocblas_int ldy,
                       Th& hA,
                       Sh& hD,
                       Sh& hE,
                       Uh& hTauq,
                       Uh& hTaup,
                       Th& hX,
                       Th& hY,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    if(!perf)
    {
        labrd_initData<true, false, S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx,
                                          dY, ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        // cpu-lapack performance
        *cpu_time_used = get_time_us();
        cblas_labrd<S, T>(m, n, nb, hA[0], lda, hD[0], hE[0], hTauq[0], hTaup[0], hX[0], ldx, hY[0],
                          ldy);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    labrd_initData<true, false, S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                      ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        labrd_initData<false, true, S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx,
                                          dY, ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(),
                                            dTauq.data(), dTaup.data(), dX.data(), ldx, dY.data(),
                                            ldy));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        labrd_initData<false, true, S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx,
                                          dY, ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        start = get_time_us();
        rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(), dTauq.data(),
                        dTaup.data(), dX.data(), ldx, dY.data(), ldy);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_labrd(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int nb = argus.K;
    rocblas_int lda = argus.lda;
    rocblas_int ldx = argus.ldb;
    rocblas_int ldy = argus.ldc;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = nb;
    size_t size_E = nb;
    size_t size_Q = nb;
    size_t size_P = nb;
    size_t size_X = ldx * nb;
    size_t size_Y = ldy * nb;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;
    size_t size_YRes = (argus.unit_check || argus.norm_check) ? size_Y : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || nb < 0 || nb > min(m, n) || lda < m || ldx < m || ldy < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, (S*)nullptr,
                                              (S*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr,
                                              ldx, (T*)nullptr, ldy),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hTauq(size_Q, 1, size_Q, 1);
    host_strided_batch_vector<T> hTaup(size_P, 1, size_P, 1);
    host_strided_batch_vector<T> hX(size_X, 1, size_X, 1);
    host_strided_batch_vector<T> hXRes(size_XRes, 1, size_XRes, 1);
    host_strided_batch_vector<T> hY(size_Y, 1, size_Y, 1);
    host_strided_batch_vector<T> hYRes(size_YRes, 1, size_YRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dTauq(size_Q, 1, size_Q, 1);
    device_strided_batch_vector<T> dTaup(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dX(size_X, 1, size_X, 1);
    device_strided_batch_vector<T> dY(size_Y, 1, size_Y, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_Q)
        CHECK_HIP_ERROR(dTauq.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dTaup.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dX.memcheck());
    if(size_Y)
        CHECK_HIP_ERROR(dY.memcheck());

    // check quick return
    if(m == 0 || n == 0 || nb == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(),
                                              dE.data(), dTauq.data(), dTaup.data(), dX.data(), ldx,
                                              dY.data(), ldy),
                              rocblas_status_success);
        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        labrd_getError<S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy, hA,
                             hARes, hD, hE, hTauq, hTaup, hX, hXRes, hY, hYRes, &max_error);

    // collect performance data
    if(argus.timing)
        labrd_getPerfData<S, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy,
                                hA, hD, hE, hTauq, hTaup, hX, hY, &gpu_time_used, &cpu_time_used,
                                hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using nb * max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, nb * max(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("m", "n", "nb", "lda", "ldx", "ldy");
            rocsolver_bench_output(m, n, nb, lda, ldx, ldy);
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
