/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool BATCHED, bool STRIDED, typename U>
void gels_checkBadArgs(const rocblas_handle handle,
                       const rocblas_operation trans,
                       const rocblas_int m,
                       const rocblas_int n,
                       const rocblas_int nrhs,
                       U dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       U dC,
                       const rocblas_int ldc,
                       const rocblas_stride stC,
                       rocblas_int* info,
                       const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gels(STRIDED, nullptr, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC, info, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, rocblas_operation(-1), m, n, nrhs, dA,
                                         lda, stA, dC, ldc, stC, info, bc),
                          rocblas_status_invalid_value)
        << "Must report error when operation is invalid";

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA, dC,
                                             ldc, stC, info, -1),
                              rocblas_status_invalid_size)
            << "Must report error when batch size is negative";

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, (U) nullptr, lda, stA,
                                         dC, ldc, stC, info, bc),
                          rocblas_status_invalid_pointer)
        << "Should normally report error when A is null";
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA,
                                         (U) nullptr, ldc, stC, info, bc),
                          rocblas_status_invalid_pointer)
        << "Should normally report error when C is null";
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC, nullptr, bc),
        rocblas_status_invalid_pointer)
        << "Should normally report error when info is null";

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, 0, n, nrhs, (U) nullptr, lda, stA,
                                         dC, ldc, stC, info, bc),
                          rocblas_status_not_implemented) // TODO: replace with success
        << "Matrix A may be null when m is 0 (empty matrix)";
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, 0, nrhs, (U) nullptr, lda, stA,
                                         dC, ldc, stC, info, bc),
                          rocblas_status_success)
        << "Matrix A may be null when n is 0 (empty matrix)";
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, 0, dA, lda, stA, (U) nullptr,
                                         ldc, stC, info, bc),
                          rocblas_status_success)
        << "Matrix C may be null when nhrs is 0 (empty matrix)";
    EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, 0, 0, nrhs, (U) nullptr, lda, stA,
                                         (U) nullptr, ldc, stC, info, bc),
                          rocblas_status_success)
        << "Matrices A and C may be null when m and n are 0 (empty matrix)";
    if(BATCHED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA, dC,
                                             ldc, stC, nullptr, 0),
                              rocblas_status_success)
            << "Info may be null when batch size is 0";

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC, info, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gels_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldc = 1;
    rocblas_stride stA = 1;
    rocblas_stride stC = 1;
    rocblas_int bc = 1;
    rocblas_operation trans = rocblas_operation_none;
    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dC(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        gels_checkBadArgs<BATCHED, STRIDED>(handle, trans, m, n, nrhs, dA.data(), lda, stA,
                                            dC.data(), ldc, stC, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dC(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        gels_checkBadArgs<BATCHED, STRIDED>(handle, trans, m, n, nrhs, dA.data(), lda, stA,
                                            dC.data(), ldc, stC, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_initData(const rocblas_handle handle,
                   const rocblas_operation trans,
                   const rocblas_int m,
                   const rocblas_int n,
                   const rocblas_int nrhs,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Td& dC,
                   const rocblas_int ldc,
                   const rocblas_stride stC,
                   Ud& dInfo,
                   const rocblas_int bc,
                   Th& hA,
                   Th& hC,
                   Uh& hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hC, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < m; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_getError(const rocblas_handle handle,
                   const rocblas_operation trans,
                   const rocblas_int m,
                   const rocblas_int n,
                   const rocblas_int nrhs,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Td& dC,
                   const rocblas_int ldc,
                   const rocblas_stride stC,
                   Ud& dInfo,
                   const rocblas_int bc,
                   Th& hA,
                   Th& hC,
                   Th& hCRes,
                   Uh& hInfo,
                   double* max_err)
{
    rocblas_int sizeW = max(1, min(m, n) + max(min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    // input data initialization
    gels_initData<true, true, T>(handle, trans, n, m, nrhs, dA, lda, stA, dC, ldc, stC, dInfo, bc,
                                 hA, hC, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda, stA,
                                       dC.data(), ldc, stC, dInfo.data(), bc));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cblas_gels<T>(trans, m, n, nrhs, hA[b], lda, hC[b], ldc, hW.data(), sizeW);
    }

    // error is ||hC - hCRes|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        const rocblas_int rowsC = (trans == rocblas_operation_none) ? m : n;
        err = norm_error('I', rowsC, nrhs, ldc, hC[b], hCRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_getPerfData(const rocblas_handle handle,
                      const rocblas_operation trans,
                      const rocblas_int m,
                      const rocblas_int n,
                      const rocblas_int nrhs,
                      Td& dA,
                      const rocblas_int lda,
                      const rocblas_stride stA,
                      Td& dC,
                      const rocblas_int ldc,
                      const rocblas_stride stC,
                      Ud& dInfo,
                      const rocblas_int bc,
                      Th& hA,
                      Th& hC,
                      Uh& hInfo,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const bool perf)
{
    rocblas_int sizeW = max(1, min(m, n) + max(min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        gels_initData<true, false, T>(handle, trans, n, m, nrhs, dA, lda, stA, dC, ldc, stC, dInfo,
                                      bc, hA, hC, hInfo);
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_gels<T>(trans, m, n, nrhs, hA[b], lda, hC[b], ldc, hW.data(), sizeW);
        }
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }
    gels_initData<true, false, T>(handle, trans, n, m, nrhs, dA, lda, stA, dC, ldc, stC, dInfo, bc,
                                  hA, hC, hInfo);
    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gels_initData<false, true, T>(handle, trans, n, m, nrhs, dA, lda, stA, dC, ldc, stC, dInfo,
                                      bc, hA, hC, hInfo);
        CHECK_ROCBLAS_ERROR(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda, stA,
                                           dC.data(), ldc, stC, dInfo.data(), bc));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        gels_initData<false, true, T>(handle, trans, n, m, nrhs, dA, lda, stA, dC, ldc, stC, dInfo,
                                      bc, hA, hC, hInfo);

        start = get_time_us();
        rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda, stA, dC.data(), ldc, stC,
                       dInfo.data(), bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gels(Arguments argus)
{
    rocblas_local_handle handle;
    // Set handle memory size to a large enough value for all tests to pass.
    //(TODO: Investigate why rocblas is not automatically increasing the size of
    //the memory stack in rocblas_handle)
    rocblas_set_device_memory_size(handle, 80000000);

    // get arguments
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int nrhs = argus.K;
    rocblas_int lda = argus.lda;
    rocblas_int ldc = argus.ldc;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stC = argus.bsc;
    rocblas_int bc = argus.batch_count;
    char transC = argus.transA_option;
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stCRes = (argus.unit_check || argus.norm_check) ? stC : 0;

    // check non-supported values
    if(m < n || trans == rocblas_operation_transpose || trans == rocblas_operation_conjugate_transpose)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, (T* const*)nullptr,
                                             lda, stA, (T* const*)nullptr, ldc, stC,
                                             (rocblas_int*)nullptr, bc),
                              rocblas_status_not_implemented);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_C = size_t(ldc) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || nrhs < 0 || lda < m || ldc < m || ldc < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs,
                                                 (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                                                 ldc, stC, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, (T*)nullptr,
                                                 lda, stA, (T*)nullptr, ldc, stC,
                                                 (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hC(size_C, 1, bc);
        host_batch_vector<T> hCRes(size_CRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dC(size_C, 1, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        if(bc)
            CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda,
                                                 stA, dC.data(), ldc, stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gels_getError<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC, dInfo,
                                      bc, hA, hC, hCRes, hInfo, &max_error);

        // collect performance data
        if(argus.timing)
            gels_getPerfData<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC,
                                         dInfo, bc, hA, hC, hInfo, &gpu_time_used, &cpu_time_used,
                                         hot_calls, argus.perf);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hC(size_C, 1, stC, bc);
        host_strided_batch_vector<T> hCRes(size_CRes, 1, stCRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dC(size_C, 1, stC, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        if(bc)
            CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gels(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda,
                                                 stA, dC.data(), ldc, stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gels_getError<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC, dInfo,
                                      bc, hA, hC, hCRes, hInfo, &max_error);

        // collect performance data
        if(argus.timing)
            gels_getPerfData<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dC, ldc, stC,
                                         dInfo, bc, hA, hC, hInfo, &gpu_time_used, &cpu_time_used,
                                         hot_calls, argus.perf);
    }
    // validate results for rocsolver-test
    // using max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, max(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldc", "batch_c");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldc, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldc", "strideA",
                                       "strideC", "batch_c");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldc, stA, stC, bc);
            }
            else
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldc");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldc);
            }
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
