/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "norm.hpp"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


template <bool STRIDED, bool POTRF, typename T, typename U>
void potf2_potrf_checkBadArgs(const rocblas_handle handle, 
                         const rocblas_fill uplo, 
                         const rocblas_int n, 
                         T dA, 
                         const rocblas_int lda, 
                         const rocblas_stride stA,
                         U dinfo,
                         const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,nullptr,uplo,n,dA,lda,stA,dinfo,bc), 
                          rocblas_status_invalid_handle);
    
    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,rocblas_fill_full,n,dA,lda,stA,dinfo,bc), 
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,n,dA,lda,stA,dinfo,-1), 
                              rocblas_status_invalid_size);
        
    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,n,(T)nullptr,lda,stA,dinfo,bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,n,dA,lda,stA,(U)nullptr,bc), 
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,0,(T)nullptr,lda,stA,dinfo,bc), 
                          rocblas_status_success);
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,n,dA,lda,stA,(U)nullptr,0),
                              rocblas_status_success);
    
    // quick return with zero batch_count if applicable
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED,POTRF,handle,uplo,n,dA,lda,stA,dinfo,0),
                              rocblas_status_success);
}


template <bool BATCHED, bool STRIDED, bool POTRF, typename T>
void testing_potf2_potrf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_int bc = 1;

    if (BATCHED) {
        // memory allocations
        device_batch_vector<T> dA(1,1,1);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());
        
        // check bad arguments
        potf2_potrf_checkBadArgs<STRIDED,POTRF>(handle,uplo,n,dA.data(),lda,stA,dinfo.data(),bc);

    } else {
        // memory allocations
        device_strided_batch_vector<T> dA(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        potf2_potrf_checkBadArgs<STRIDED,POTRF>(handle,uplo,n,dA.data(),lda,stA,dinfo.data(),bc);
    }
}


template <bool STRIDED, bool POTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potf2_potrf_getError(const rocblas_handle handle, 
                        const rocblas_fill uplo, 
                        const rocblas_int n, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dinfo,
                        const rocblas_int bc,
                        Th &hA, 
                        Th &hARes, 
                        Uh &hinfo,
                        double *max_err)
{
    // input data initialization 
    rocblas_init<T>(hARes, true);

    // make A hermitian and scale to ensure positive definiteness  
    for (rocblas_int b = 0; b < bc; ++b) {
        cblas_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, n, n, n,
                   (T)1.0, hARes[b], lda, hARes[b], lda, (T)0.0, hA[b], lda);
        
        for (rocblas_int i = 0; i < n; i++) 
                    hA[b][i + i * lda] += 400;
    }
    
    // now copy data to the GPU
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_potf2_potrf(STRIDED,POTRF,handle, uplo, n, dA.data(), lda, stA, dinfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for (rocblas_int b = 0; b < bc; ++b) {
        POTRF ?
            cblas_potrf<T>(uplo, n, hA[b], lda, hinfo[b]):
            cblas_potf2<T>(uplo, n, hA[b], lda, hinfo[b]);
    }
   
    // error is ||hA - hARes|| / ||hA|| (ideally ||LL' - Lres Lres'|| / ||LL'||) 
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES. 
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for (rocblas_int b = 0; b < bc; ++b) {
        err = norm_error('F',n,n,lda,hA[b],hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}


template <bool STRIDED, bool POTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potf2_potrf_getPerfData(const rocblas_handle handle, 
                        const rocblas_fill uplo, 
                        const rocblas_int n, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dinfo,
                        const rocblas_int bc,
                        Th &hA, 
                        Uh &hinfo,
                        double *gpu_time_used,
                        double *cpu_time_used,
                        const rocblas_int hot_calls)
{
    // cpu-lapack performance
    *cpu_time_used = get_time_us();
    for (rocblas_int b = 0; b < bc; ++b) {
        POTRF ?
            cblas_potrf<T>(uplo, n, hA[b], lda, hinfo[b]):
            cblas_potf2<T>(uplo, n, hA[b], lda, hinfo[b]);
    }
    *cpu_time_used = get_time_us() - *cpu_time_used;

    // cold calls
    for(int iter = 0; iter < 2; iter++)
        CHECK_ROCBLAS_ERROR(rocsolver_potf2_potrf(STRIDED,POTRF,handle, uplo, n, dA.data(), lda, stA, dinfo.data(), bc));
        
    // gpu-lapack performance
    *gpu_time_used = get_time_us(); 
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
        rocsolver_potf2_potrf(STRIDED,POTRF,handle, uplo, n, dA.data(), lda, stA, dinfo.data(), bc);
    *gpu_time_used = (get_time_us() - *gpu_time_used) / hot_calls;
}


template <bool BATCHED, bool STRIDED, bool POTRF, typename T> 
void testing_potf2_potrf(Arguments argus) 
{
    // get arguments 
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    char uploC = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(uploC);

    // check non-supported values 
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower) {
        if (BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T *const *)nullptr, lda, stA, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T *)nullptr, lda, stA, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_value);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes 
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if (invalid_size) {
        if (BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T *const *)nullptr, lda, stA, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, (T *)nullptr, lda, stA, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if (BATCHED) {
        // memory allocations
        host_batch_vector<T> hA(size_A,1,bc);
        host_batch_vector<T> hARes(size_A,1,bc);
        host_strided_batch_vector<rocblas_int> hinfo(1,1,1,bc);
        device_batch_vector<T> dA(size_A,1,bc);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,bc);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check quick return
        if (n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(), lda, stA, dinfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            potf2_potrf_getError<STRIDED,POTRF,T>(handle, uplo, n, dA, lda, stA, dinfo, bc, 
                                          hA, hARes, hinfo, &max_error);

        // collect performance data
        if (argus.timing) 
            potf2_potrf_getPerfData<STRIDED,POTRF,T>(handle, uplo, n, dA, lda, stA, dinfo, bc, 
                                              hA, hinfo, &gpu_time_used, &cpu_time_used, hot_calls);
    } 

    else {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A,1,stA,bc);
        host_strided_batch_vector<T> hARes(size_A,1,stA,bc);
        host_strided_batch_vector<rocblas_int> hinfo(1,1,1,bc);
        device_strided_batch_vector<T> dA(size_A,1,stA,bc);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,bc);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check quick return
        if (n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_potf2_potrf(STRIDED, POTRF, handle, uplo, n, dA.data(), lda, stA, dinfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            potf2_potrf_getError<STRIDED,POTRF,T>(handle, uplo, n, dA, lda, stA, dinfo, bc, 
                                          hA, hARes, hinfo, &max_error);

        // collect performance data
        if (argus.timing) 
            potf2_potrf_getPerfData<STRIDED,POTRF,T>(handle, uplo, n, dA, lda, stA, dinfo, bc, 
                                              hA, hinfo, &gpu_time_used, &cpu_time_used, hot_calls);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if (argus.unit_check) 
        rocsolver_test_check<T>(max_error,n);     

    // output results for rocsolver-bench
    if (argus.timing) {
        rocblas_cout << "\n============================================\n";
        rocblas_cout << "Arguments:\n";
        rocblas_cout << "============================================\n";
        if (BATCHED) {
            rocsolver_bench_output("uplo", "n", "lda", "batch_c");
            rocsolver_bench_output(uploC, n, lda, bc);
        }
        else if (STRIDED) {
            rocsolver_bench_output("uplo", "n", "lda", "strideA", "batch_c");
            rocsolver_bench_output(uploC, n, lda, stA, bc);
        }
        else {
            rocsolver_bench_output("uplo", "n", "lda");
            rocsolver_bench_output(uploC, n, lda);
        }
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
}
  

#undef POTRF_ERROR_EPS_MULTIPLIER
