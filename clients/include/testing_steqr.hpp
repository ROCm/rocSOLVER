/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename S, typename T, typename U>
void steqr_checkBadArgs(const rocblas_handle handle,
                        const rocblas_evect evect,
                        const rocblas_int n,
                        S dD,
                        S dE,
                        T dC,
                        const rocblas_int ldc,
                        U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(nullptr, evect, n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, rocblas_evect(-1), n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, evect, n, (S) nullptr, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, evect, n, dD, (S) nullptr, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, evect, n, dD, dE, (T) nullptr, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, evect, n, dD, dE, dC, ldc, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_steqr(handle, evect, 0, (S) nullptr, (S) nullptr, (T) nullptr, ldc, dInfo),
        rocblas_status_success);
}

template <typename T>
void testing_steqr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_original;
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
    steqr_checkBadArgs(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
}

template <bool CPU, bool GPU, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_initData(const rocblas_handle handle,
                    const rocblas_evect evect,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Th& hC,
                    Uh& hInfo)
{
    if(CPU)
    {
        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, true);

        // scale matrix and add random splits
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 400;
            hE[0][i] -= 5;
        }

        // add fixed splits in the matrix to test split handling
        rocblas_int k = n / 2;
        hE[0][k] = 0;
        hE[0][k - 1] = 0;

        // initialize C to the identity matrix
        if(evect == rocblas_evect_original)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hC[0][i + j * ldc] = 1;
                    else
                        hC[0][i + j * ldc] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_getError(const rocblas_handle handle,
                    const rocblas_evect evect,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hDRes,
                    Sh& hE,
                    Sh& hERes,
                    Th& hC,
                    Th& hCRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err)
{
    using S = decltype(std::real(T{}));

    size_t lwork = (evect == rocblas_evect_none ? 0 : 2 * n - 2);
    std::vector<S> work(lwork);

    // input data initialization
    steqr_initData<true, true, S, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_steqr(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != rocblas_evect_none)
        CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // if eigenvectors were required, prepare matrix A (upper triangular) for implicit tests
    rocblas_int lda = n;
    size_t size_A = lda * n;
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    if(evect != rocblas_evect_none)
    {
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = i; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hD[0][i];
                else if(i + 1 == j)
                    hA[0][i + j * lda] = hE[0][i];
                else
                    hA[0][i + j * lda] = 0;
            }
        }
    }

    // CPU lapack
    cblas_steqr<S, T>(evect, n, hD[0], hE[0], hC[0], ldc, work.data(), hInfo[0]);

    // check info
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    double err;

    if(hInfo[0][0] == 0)
    {
        // check that eigenvalues are correct and in order
        // error is ||hD - hDRes|| / ||hD||
        // using frobenius norm
        err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
        *max_err = err > *max_err ? err : *max_err;

        // check eigenvectors if required
        if(evect != rocblas_evect_none)
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling

            // multiply A with each of the n eigenvectors and divide by corresponding
            // eigenvalues
            T alpha;
            T beta = 0;
            for(int j = 0; j < n; j++)
            {
                alpha = T(1) / hDRes[0][j];
                cblas_symv_hemv(rocblas_fill_upper, n, alpha, hA[0], lda, hCRes[0] + j * ldc, 1,
                                beta, hC[0] + j * ldc, 1);
            }

            // error is ||hC - hCRes|| / ||hC||
            // using frobenius norm
            err = norm_error('F', n, n, ldc, hC[0], hCRes[0]);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void steqr_getPerfData(const rocblas_handle handle,
                       const rocblas_evect evect,
                       const rocblas_int n,
                       Sd& dD,
                       Sd& dE,
                       Td& dC,
                       const rocblas_int ldc,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Th& hC,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    using S = decltype(std::real(T{}));

    size_t lwork = (evect == rocblas_evect_none ? 0 : 2 * n - 2);
    std::vector<S> work(lwork);

    if(!perf)
    {
        steqr_initData<true, false, S, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                          hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_steqr<S, T>(evect, n, hD[0], hE[0], hC[0], ldc, work.data(), hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    steqr_initData<true, false, S, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        steqr_initData<false, true, S, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                          hInfo);

        CHECK_ROCBLAS_ERROR(
            rocsolver_steqr(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        steqr_initData<false, true, S, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                          hInfo);

        start = get_time_us_sync(stream);
        rocsolver_steqr(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_steqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldc = argus.get<rocblas_int>("ldc", n);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_C = ldc * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || (evect != rocblas_evect_none && ldc < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_steqr(handle, evect, n, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldc, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory size query is necessary
    if(!USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_steqr(handle, evect, n, (S*)nullptr, (S*)nullptr, (T*)nullptr,
                                          ldc, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<S> hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
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
            rocsolver_steqr(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()),
            rocblas_status_success);
        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        steqr_getError<T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hDRes, hE, hERes, hC, hCRes,
                          hInfo, hInfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        steqr_getPerfData<T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Arguments:\n";
            rocsolver_cout << "============================================\n";
            rocsolver_bench_output("evect", "n", "ldc");
            rocsolver_bench_output(evectC, n, ldc);

            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Results:\n";
            rocsolver_cout << "============================================\n";
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
            rocsolver_cout << std::endl;
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
