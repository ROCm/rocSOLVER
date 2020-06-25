/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "norm.hpp"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


template <bool STRIDED, typename T, typename U>
void getri_checkBadArgs(const rocblas_handle handle, 
                         const rocblas_int n, 
                         T dA1, 
                         T dA, 
                         const rocblas_int lda, 
                         const rocblas_stride stA,
                         U dIpiv, 
                         const rocblas_stride stP,
                         U dInfo, 
                         const rocblas_int bc)
{
    // NOTE: dA1 is only used for getri_outofplace_batched
    // It is ignored in bad arg checks

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,nullptr,n,dA1,dA,lda,stA,dIpiv,stP,dInfo,bc), 
                          rocblas_status_invalid_handle);
    
    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,n,dA1,dA,lda,stA,dIpiv,stP,dInfo,-1), 
                              rocblas_status_invalid_size);
        
    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,n,dA1,(T)nullptr,lda,stA,dIpiv,stP,dInfo,bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,n,dA1,dA,lda,stA,(U)nullptr,stP,dInfo,bc), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,n,dA1,dA,lda,stA,dIpiv,stP,(U)nullptr,bc), 
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,0,dA1,(T)nullptr,lda,stA,(U)nullptr,stP,dInfo,bc), 
                          rocblas_status_success);
    
    // quick return with zero batch_count if applicable
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED,handle,n,dA1,dA,lda,stA,dIpiv,stP,(U)nullptr,0),
                              rocblas_status_success);
}


template <bool BATCHED, bool STRIDED, typename T>
void testing_getri_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if (BATCHED) {
        // memory allocations
        device_batch_vector<T> dA1(1,1,1);
        device_batch_vector<T> dA(1,1,1);
        device_strided_batch_vector<rocblas_int> dIpiv(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dInfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        
        // check bad arguments
        getri_checkBadArgs<STRIDED>(handle,n,dA1.data(),dA.data(),lda,stA,dIpiv.data(),stP,dInfo.data(),bc);

    } else {
        // memory allocations
        device_strided_batch_vector<T> dA1(1,1,1,1);
        device_strided_batch_vector<T> dA(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dIpiv(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dInfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        getri_checkBadArgs<STRIDED>(handle,n,dA1.data(),dA.data(),lda,stA,dIpiv.data(),stP,dInfo.data(),bc);
    }
}


template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getri_initData(const rocblas_handle handle, 
                        const rocblas_int n, 
                        Td &dA1, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dInfo, 
                        const rocblas_int bc,
                        Th &hA1, 
                        Th &hA, 
                        Uh &hIpiv, 
                        Uh &hInfo)
{
    if (CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities 
        for (rocblas_int b = 0; b < bc; ++b) {
            for (rocblas_int i = 0; i < n; i++) {
                for (rocblas_int j = 0; j < n; j++) {
                    if (i == j)
                        hA[b][i + j * lda] += 400;
                    else    
                        hA[b][i + j * lda] -= 4;
                }
            }
        }

        // do the LU decomposition of matrix A w/ the reference LAPACK routine
        for (rocblas_int b = 0; b < bc; ++b) {
            cblas_getrf<T>(n, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
    }
    
    // now copy data to the GPU
    if (GPU)
    {
        if (dA1.n() > 0)
            CHECK_HIP_ERROR(dA1.transfer_from(hA));
        else
            CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}


template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getri_getError(const rocblas_handle handle, 
                        const rocblas_int n, 
                        Td &dA1, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dInfo, 
                        const rocblas_int bc,
                        Th &hA1, 
                        Th &hA, 
                        Th &hARes, 
                        Uh &hIpiv, 
                        Uh &hInfo, 
                        double *max_err)
{
    rocblas_int sizeW = n;
    std::vector<T> hW(sizeW);

    // input data initialization 
    getri_initData<true,true,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                      hA1, hA, hIpiv, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getri(STRIDED,handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for (rocblas_int b = 0; b < bc; ++b) {
        if (hInfo[b][0] != 0)
            return;  // error encountered - unlucky pick of random numbers? no use to continue
        
        cblas_getri<T>(n, hA[b], lda, hIpiv[b], hW.data(), &sizeW);
    }
   
    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA||
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


template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getri_getPerfData(const rocblas_handle handle, 
                        const rocblas_int n, 
                        Td &dA1, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dInfo, 
                        const rocblas_int bc,
                        Th &hA1, 
                        Th &hA, 
                        Uh &hIpiv, 
                        Uh &hInfo, 
                        double *gpu_time_used,
                        double *cpu_time_used,
                        const rocblas_int hot_calls)
{
    rocblas_int sizeW = n;
    std::vector<T> hW(sizeW);

    // cpu-lapack performance
    getri_initData<true,false,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                      hA1, hA, hIpiv, hInfo);
    *cpu_time_used = get_time_us();
    for (rocblas_int b = 0; b < bc; ++b) {
        cblas_getri<T>(n, hA[b], lda, hIpiv[b], hW.data(), &sizeW);
    }
    *cpu_time_used = get_time_us() - *cpu_time_used;
    getri_initData<true,false,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                      hA1, hA, hIpiv, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getri_initData<false,true,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                          hA1, hA, hIpiv, hInfo);

        CHECK_ROCBLAS_ERROR(rocsolver_getri(STRIDED,handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc));
    }
        
    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        getri_initData<false,true,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                          hA1, hA, hIpiv, hInfo);
        
        start = get_time_us();
        rocsolver_getri(STRIDED,handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}


template <bool BATCHED, bool STRIDED, typename T> 
void testing_getri(Arguments argus) 
{
    // get arguments 
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stP = argus.bsp;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = argus.unit_check || argus.norm_check ? stA : 0;

    // check non-supported values 
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = argus.unit_check || argus.norm_check ? size_A : 0;

    // check invalid sizes 
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if (invalid_size) {
        if (BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED, handle, n, (T *const *)nullptr, (T *const *)nullptr, lda, stA, (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED, handle, n, (T *)nullptr, (T *)nullptr, lda, stA, (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if (BATCHED && STRIDED) {
        // memory allocations
        host_batch_vector<T> hA1(0,1,bc);
        host_batch_vector<T> hA(size_A,1,bc);
        host_batch_vector<T> hARes(size_ARes,1,bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,stP,bc);
        host_strided_batch_vector<rocblas_int> hInfo(1,1,1,bc);
        device_batch_vector<T> dA1(0,1,bc);
        device_batch_vector<T> dA(size_A,1,bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,stP,bc);
        device_strided_batch_vector<rocblas_int> dInfo(1,1,1,bc);
        if (size_A) CHECK_HIP_ERROR(dA.memcheck());
        if (size_P) CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if (n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED, handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            getri_getError<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                          hA1, hA, hARes, hIpiv, hInfo, &max_error);

        // collect performance data
        if (argus.timing) 
            getri_getPerfData<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                              hA1, hA, hIpiv, hInfo, &gpu_time_used, &cpu_time_used, hot_calls);
    } 
    
    else if (BATCHED) {
        // memory allocations
        host_batch_vector<T> hA1(size_A,1,bc);
        host_batch_vector<T> hA(size_A,1,bc);
        host_batch_vector<T> hARes(size_ARes,1,bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,stP,bc);
        host_strided_batch_vector<rocblas_int> hInfo(1,1,1,bc);
        device_batch_vector<T> dA1(size_A,1,bc);
        device_batch_vector<T> dA(size_A,1,bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,stP,bc);
        device_strided_batch_vector<rocblas_int> dInfo(1,1,1,bc);
        if (size_A) CHECK_HIP_ERROR(dA1.memcheck());
        if (size_A) CHECK_HIP_ERROR(dA.memcheck());
        if (size_P) CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if (n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED, handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            getri_getError<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                          hA1, hA, hARes, hIpiv, hInfo, &max_error);

        // collect performance data
        if (argus.timing) 
            getri_getPerfData<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                              hA1, hA, hIpiv, hInfo, &gpu_time_used, &cpu_time_used, hot_calls);
    } 

    else {
        // memory allocations
        host_strided_batch_vector<T> hA1(0,1,0,bc);
        host_strided_batch_vector<T> hA(size_A,1,stA,bc);
        host_strided_batch_vector<T> hARes(size_ARes,1,stARes,bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,stP,bc);
        host_strided_batch_vector<rocblas_int> hInfo(1,1,1,bc);
        device_strided_batch_vector<T> dA1(0,1,0,bc);
        device_strided_batch_vector<T> dA(size_A,1,stA,bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,stP,bc);
        device_strided_batch_vector<rocblas_int> dInfo(1,1,1,bc);
        if (size_A) CHECK_HIP_ERROR(dA.memcheck());
        if (size_P) CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if (n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_getri(STRIDED, handle, n, dA1.data(), dA.data(), lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            getri_getError<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                          hA1, hA, hARes, hIpiv, hInfo, &max_error);

        // collect performance data
        if (argus.timing) 
            getri_getPerfData<STRIDED,T>(handle, n, dA1, dA, lda, stA, dIpiv, stP, dInfo, bc, 
                                              hA1, hA, hIpiv, hInfo, &gpu_time_used, &cpu_time_used, hot_calls);
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
            rocsolver_bench_output("n", "lda", "strideP", "batch_c");
            rocsolver_bench_output(n, lda, stP, bc);
        }
        else if (STRIDED) {
            rocsolver_bench_output("n", "lda", "strideA", "strideP", "batch_c");
            rocsolver_bench_output(n, lda, stA, stP, bc);
        }
        else {
            rocsolver_bench_output("n", "lda");
            rocsolver_bench_output(n, lda);
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
  

#undef GETRF_ERROR_EPS_MULTIPLIER
