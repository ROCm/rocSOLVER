/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "norm.hpp"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


template <typename T, typename U>
void trtri_checkBadArgs(const rocblas_handle handle,
                         const rocblas_fill uplo,
                         const rocblas_diagonal diag,
                         const rocblas_int n,
                         T dA,
                         const rocblas_int lda,
                         U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(nullptr,uplo,diag,n,dA,lda,dInfo), 
                          rocblas_status_invalid_handle);
    
    // values
    rocblas_diagonal bad_diag = static_cast<rocblas_diagonal>(-1);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,rocblas_fill_full,diag,n,dA,lda,dInfo), 
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,uplo,bad_diag,n,dA,lda,dInfo), 
                          rocblas_status_invalid_value);
        
    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,uplo,diag,n,(T)nullptr,lda,dInfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,uplo,diag,n,dA,lda,(U)nullptr), 
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,uplo,diag,0,(T)nullptr,lda,dInfo), 
                          rocblas_status_success);
}


template <typename T>
void testing_trtri_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_int n = 1;
    rocblas_int lda = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1,1,1,1);
    device_strided_batch_vector<rocblas_int> dInfo(1,1,1,1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    trtri_checkBadArgs(handle,uplo,diag,n,dA.data(),lda,dInfo.data());
}


template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void trtri_initData(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_diagonal diag,
                        const rocblas_int n,
                        Td &dA,
                        const rocblas_int lda,
                        Ud &dInfo,
                        Th &hA,
                        Uh &hInfo)
{
    if (CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities 
        if (diag == rocblas_diagonal_non_unit)
        {
            for (rocblas_int i = 0; i < n; i++)
            {
                for (rocblas_int j = 0; j < n; j++)
                {
                    if (i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
        }
        else
        {
            for (rocblas_int i = 0; i < n; i++)
            {
                for (rocblas_int j = 0; j < n; j++)
                {
                    if (i != j)
                        hA[0][i + j * lda] = (hA[0][i + j * lda] - 4) / 400;
                }
            }
        }

        // (TODO: add some singular matrices)
    }
    
    // now copy data to the GPU
    if (GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}


template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void trtri_getError(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_diagonal diag,
                        const rocblas_int n,
                        Td &dA,
                        const rocblas_int lda,
                        Ud &dInfo,
                        Th &hA,
                        Th &hARes,
                        Uh &hInfo,
                        double *max_err)
{
    // input data initialization 
    trtri_initData<true,true,T>(handle, uplo, diag, n, dA, lda, dInfo,
                                      hA, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_trtri(handle, uplo, diag, n, dA.data(), lda, dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    cblas_trtri<T>(uplo, diag, n, hA[0], lda);
   
    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES. 
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F',n,n,lda,hA[0],hARes[0]);
    *max_err = err > *max_err ? err : *max_err;
}


template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void trtri_getPerfData(const rocblas_handle handle, 
                        const rocblas_fill uplo,
                        const rocblas_diagonal diag,
                        const rocblas_int n,
                        Td &dA,
                        const rocblas_int lda,
                        Ud &dInfo,
                        Th &hA,
                        Uh &hInfo,
                        double *gpu_time_used,
                        double *cpu_time_used,
                        const rocblas_int hot_calls,
                        const bool perf)
{
    if (!perf)
    {
        trtri_initData<true,false,T>(handle, uplo, diag, n, dA, lda, dInfo,
                                        hA, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_trtri<T>(uplo, diag, n, hA[0], lda);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    trtri_initData<true,false,T>(handle, uplo, diag, n, dA, lda, dInfo,
                                      hA, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trtri_initData<false,true,T>(handle, uplo, diag, n, dA, lda, dInfo,
                                        hA, hInfo);

        CHECK_ROCBLAS_ERROR(rocsolver_trtri(handle, uplo, diag, n, dA.data(), lda, dInfo.data()));
    }
        
    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        trtri_initData<false,true,T>(handle, uplo, diag, n, dA, lda, dInfo,
                                        hA, hInfo);
        
        start = get_time_us();
        rocsolver_trtri(handle, uplo, diag, n, dA.data(), lda, dInfo.data());
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}


template <typename T>
void testing_trtri(Arguments argus) 
{
    // get arguments 
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    char uploC = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(uploC);
    char diagC = argus.diag_option;
    rocblas_diagonal diag = char2rocblas_diagonal(diagC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if ((uplo != rocblas_fill_upper && uplo != rocblas_fill_lower) ||
        (diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)) {
        EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle,uplo,diag,n,(T*)nullptr,lda,(rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if (invalid_size) {
        EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle, uplo, diag, n, (T *)nullptr, lda, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A,1,size_A,1);
    host_strided_batch_vector<T> hARes(size_ARes,1,size_ARes,1);
    host_strided_batch_vector<rocblas_int> hInfo(1,1,1,1);
    device_strided_batch_vector<T> dA(size_A,1,size_A,1);
    device_strided_batch_vector<rocblas_int> dInfo(1,1,1,1);
    if (size_A) CHECK_HIP_ERROR(dA.memcheck());

    // check quick return
    if (n == 0) {
        EXPECT_ROCBLAS_STATUS(rocsolver_trtri(handle, uplo, diag, n, dA.data(), lda, dInfo.data()),
                              rocblas_status_success);
        if (argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if (argus.unit_check || argus.norm_check) 
        trtri_getError<T>(handle, uplo, diag, n, dA, lda, dInfo,
                          hA, hARes, hInfo, &max_error);

    // collect performance data
    if (argus.timing) 
        trtri_getPerfData<T>(handle, uplo, diag, n, dA, lda, dInfo,
                             hA, hInfo, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if (argus.unit_check) 
        rocsolver_test_check<T>(max_error,n);     

    // output results for rocsolver-bench
    if (argus.timing) {
        if (!argus.perf) {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("n", "lda");
            rocsolver_bench_output(n, lda);
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Results:\n";
            rocblas_cout << "============================================\n";
            if (argus.norm_check) {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocblas_cout << std::endl;
        }
        else {
            if (argus.norm_check) rocsolver_bench_output(gpu_time_used,max_error);
            else rocsolver_bench_output(gpu_time_used);
        }
    }
}
