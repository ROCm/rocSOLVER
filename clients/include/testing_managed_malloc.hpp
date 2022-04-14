/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

/*
 * ===========================================================================
 *    testing_managed_malloc is a modified version of testing_labrd that tests
 *    the unified memory model/HMM. checkBadArgs has been removed as the memory
 *    model has no impact on the bad arg check.
 * ===========================================================================
 */

template <bool CPU, bool GPU, typename T>
void managed_malloc_initData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int nb,
                             T* dA,
                             T* dARes,
                             const rocblas_int lda)
{
    if(CPU)
    {
        rocblas_init<T>(dA, m, n, lda);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < m; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                    dA[i + j * lda] += 400;
                else
                    dA[i + j * lda] -= 4;
            }
        }
    }

    if(GPU)
    {
        // copy A
        for(rocblas_int i = 0; i < m; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                dARes[i + j * lda] = dA[i + j * lda];
            }
        }
    }
}

template <typename T, typename S>
void managed_malloc_getError(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int nb,
                             T* dA,
                             T* dARes,
                             const rocblas_int lda,
                             S* dD,
                             S* dE,
                             T* dTauq,
                             T* dTaup,
                             T* dX,
                             T* dXRes,
                             const rocblas_int ldx,
                             T* dY,
                             T* dYRes,
                             const rocblas_int ldy,
                             double* max_err)
{
    // input data initialization
    managed_malloc_initData<true, true, T>(handle, m, n, nb, dA, dARes, lda);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dARes, lda, dD, dE, dTauq, dTaup, dXRes,
                                        ldx, dYRes, ldy));
    hipDeviceSynchronize();

    // CPU lapack
    cblas_labrd<T>(m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy);

    // error is max(||hA - hARes|| / ||hA||, ||hX - hXRes|| / ||hX||, ||hY -
    // hYRes|| / ||hY||) (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F', m, n, lda, dA, dARes);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', m - nb, nb, ldx, dX + nb, dXRes + nb);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', n - nb, nb, ldy, dY + nb, dYRes + nb);
    *max_err = err > *max_err ? err : *max_err;
}

template <typename T, typename S>
void managed_malloc_getPerfData(const rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int nb,
                                T* dA,
                                T* dARes,
                                const rocblas_int lda,
                                S* dD,
                                S* dE,
                                T* dTauq,
                                T* dTaup,
                                T* dX,
                                T* dXRes,
                                const rocblas_int ldx,
                                T* dY,
                                T* dYRes,
                                const rocblas_int ldy,
                                double* gpu_time_used,
                                double* cpu_time_used,
                                const rocblas_int hot_calls,
                                const int profile,
                                const bool profile_kernels,
                                const bool perf)
{
    if(!perf)
    {
        managed_malloc_initData<true, false, T>(handle, m, n, nb, dA, dARes, lda);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        cblas_labrd<T>(m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    managed_malloc_initData<true, false, T>(handle, m, n, nb, dA, dARes, lda);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        managed_malloc_initData<false, true, T>(handle, m, n, nb, dA, dARes, lda);

        CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dARes, lda, dD, dE, dTauq, dTaup,
                                            dXRes, ldx, dYRes, ldy));
        hipDeviceSynchronize();
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        managed_malloc_initData<false, true, T>(handle, m, n, nb, dA, dARes, lda);

        start = get_time_us_sync(stream);
        rocsolver_labrd(handle, m, n, nb, dARes, lda, dD, dE, dTauq, dTaup, dXRes, ldx, dYRes, ldy);
        hipDeviceSynchronize();
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_managed_malloc(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int nb = argus.get<rocblas_int>("k", min(m, n));
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", m);
    rocblas_int ldy = argus.get<rocblas_int>("ldy", n);

    rocblas_int hot_calls = argus.iters;

    // check managed memory enablement
    int deviceID, hmm_enabled;
    hipGetDevice(&deviceID);
    hipDeviceGetAttribute(&hmm_enabled, hipDeviceAttributeManagedMemory, deviceID);
    if(!hmm_enabled)
    {
        std::puts("Managed memory not enabled on device. Skipping test...");
        std::fflush(stdout);
        return;
    }

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

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || nb < 0 || nb > min(m, n) || lda < m || ldx < m || ldy < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, (S*)nullptr,
                                              (S*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr,
                                              ldx, (T*)nullptr, ldy),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, (S*)nullptr,
                                          (S*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr, ldx,
                                          (T*)nullptr, ldy));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    S *dD, *dE;
    T *dA, *dARes, *dTauq, *dTaup, *dX, *dXRes, *dY, *dYRes;
    hipMallocManaged(&dA, sizeof(T) * size_A);
    hipMallocManaged(&dARes, sizeof(T) * size_A);
    hipMallocManaged(&dD, sizeof(S) * size_D);
    hipMallocManaged(&dE, sizeof(S) * size_E);
    hipMallocManaged(&dTauq, sizeof(T) * size_Q);
    hipMallocManaged(&dTaup, sizeof(T) * size_P);
    hipMallocManaged(&dX, sizeof(T) * size_X);
    hipMallocManaged(&dXRes, sizeof(T) * size_X);
    hipMallocManaged(&dY, sizeof(T) * size_Y);
    hipMallocManaged(&dYRes, sizeof(T) * size_Y);

    // check quick return
    if(m == 0 || n == 0 || nb == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, dARes, lda, dD, dE, dTauq, dTaup,
                                              dXRes, ldx, dYRes, ldy),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        managed_malloc_getError<T>(handle, m, n, nb, dA, dARes, lda, dD, dE, dTauq, dTaup, dX,
                                   dXRes, ldx, dY, dYRes, ldy, &max_error);

    // collect performance data
    if(argus.timing)
        managed_malloc_getPerfData<T>(handle, m, n, nb, dA, dARes, lda, dD, dE, dTauq, dTaup, dX,
                                      dXRes, ldx, dY, dYRes, ldy, &gpu_time_used, &cpu_time_used,
                                      hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // free memory
    hipFree(dA);
    hipFree(dARes);
    hipFree(dD);
    hipFree(dE);
    hipFree(dTauq);
    hipFree(dTaup);
    hipFree(dX);
    hipFree(dXRes);
    hipFree(dY);
    hipFree(dYRes);

    // validate results for rocsolver-test
    // using nb * max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, nb * max(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("m", "n", "nb", "lda", "ldx", "ldy");
            rocsolver_bench_output(m, n, nb, lda, ldx, ldy);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
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
