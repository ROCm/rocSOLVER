/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename S, typename T>
void managed_malloc_checkBadArgs(const rocblas_handle handle,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 const rocblas_int nb,
                                 T* dA,
                                 const rocblas_int lda,
                                 S* dD,
                                 S* dE,
                                 T* dTauq,
                                 T* dTaup,
                                 T* dX,
                                 const rocblas_int ldx,
                                 T* dY,
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
        rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, (S*)nullptr, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, (S*)nullptr, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, (T*)nullptr, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, (T*)nullptr, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, (T*)nullptr, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, (T*)nullptr, ldy),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, 0, n, 0, (T*)nullptr, lda, dD, dE, dTauq, dTaup,
                                          (T*)nullptr, ldx, dY, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, 0, 0, (T*)nullptr, lda, dD, dE, dTauq, dTaup,
                                          dX, ldx, (T*)nullptr, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, 0, dA, lda, (S*)nullptr, (S*)nullptr,
                                          (T*)nullptr, (T*)nullptr, (T*)nullptr, ldx, (T*)nullptr,
                                          ldy),
                          rocblas_status_success);
}

template <typename T>
void testing_managed_malloc_bad_arg()
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
    S *dD, *dE;
    T *dA, *dTauq, *dTaup, *dX, *dY;
    hipMallocManaged(&dA, sizeof(T));
    hipMallocManaged(&dD, sizeof(S));
    hipMallocManaged(&dE, sizeof(S));
    hipMallocManaged(&dTauq, sizeof(T));
    hipMallocManaged(&dTaup, sizeof(T));
    hipMallocManaged(&dX, sizeof(T));
    hipMallocManaged(&dY, sizeof(T));

    // check bad arguments
    managed_malloc_checkBadArgs(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy);

    // free memory
    hipFree(dA);
    hipFree(dD);
    hipFree(dE);
    hipFree(dTauq);
    hipFree(dTaup);
    hipFree(dX);
    hipFree(dY);
}

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

template <typename S, typename T>
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
    // CHECK_HIP_ERROR(hARes.transfer_from(dA));
    // CHECK_HIP_ERROR(hXRes.transfer_from(dX));
    // CHECK_HIP_ERROR(hYRes.transfer_from(dY));

    // CPU lapack
    cblas_labrd<S, T>(m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy);

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

template <typename S, typename T>
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
                                const bool perf)
{
    if(!perf)
    {
        managed_malloc_initData<true, false, T>(handle, m, n, nb, dA, dARes, lda);

        // cpu-lapack performance
        *cpu_time_used = get_time_us();
        // memset(hX[0], 0, ldx * nb * sizeof(T));
        // memset(hY[0], 0, ldy * nb * sizeof(T));
        cblas_labrd<S, T>(m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy);
        *cpu_time_used = get_time_us() - *cpu_time_used;
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
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        managed_malloc_initData<false, true, T>(handle, m, n, nb, dA, dARes, lda);

        start = get_time_us();
        rocsolver_labrd(handle, m, n, nb, dARes, lda, dD, dE, dTauq, dTaup, dXRes, ldx, dYRes, ldy);
        hipDeviceSynchronize();
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_managed_malloc(Arguments argus)
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
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        managed_malloc_getError<S, T>(handle, m, n, nb, dA, dARes, lda, dD, dE, dTauq, dTaup, dX,
                                      dXRes, ldx, dY, dYRes, ldy, &max_error);

    // collect performance data
    if(argus.timing)
        managed_malloc_getPerfData<S, T>(handle, m, n, nb, dA, dARes, lda, dD, dE, dTauq, dTaup, dX,
                                         dXRes, ldx, dY, dYRes, ldy, &gpu_time_used, &cpu_time_used,
                                         hot_calls, argus.perf);

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
