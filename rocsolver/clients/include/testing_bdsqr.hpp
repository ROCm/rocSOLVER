/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "norm.hpp"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


template <typename S, typename T>
void bdsqr_checkBadArgs(const rocblas_handle handle, 
                         const rocblas_fill uplo,
                         const rocblas_int n, 
                         const rocblas_int nv, 
                         const rocblas_int nu,
                         const rocblas_int nc,
                         S dD, 
                         S dE,
                         T dV,
                         const rocblas_int ldv,
                         T dU,
                         const rocblas_int ldu,
                         T dC,
                         const rocblas_int ldc,
                         rocblas_int* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(nullptr,uplo,n,nv,nu,nc,dD,dE,dV,ldv,dU,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_handle);
    
    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,rocblas_fill_full,n,nv,nu,nc,dD,dE,dV,ldv,dU,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_value);
       
    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,(S)nullptr,dE,dV,ldv,dU,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,(S)nullptr,dV,ldv,dU,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,dE,(T)nullptr,ldv,dU,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,dE,dV,ldv,(T)nullptr,ldu,dC,ldc,dinfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,dE,dV,ldv,dU,ldu,(T)nullptr,ldc,dinfo), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,dE,dV,ldv,dU,ldu,dC,ldc,(rocblas_int*)nullptr), 
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,0,nv,nu,nc,(S)nullptr,(S)nullptr,(T)nullptr,ldv,(T)nullptr,ldu,(T)nullptr,ldc,dinfo), 
                          rocblas_status_success);
}


template <typename T>
void testing_bdsqr_bad_arg()
{
    typedef typename std::conditional<!is_complex<T>, T, decltype(std::real(T{}))>::type S;

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 2;
    rocblas_int nv = 2;
    rocblas_int nu = 2;
    rocblas_int nc = 2;
    rocblas_int ldv = 2;
    rocblas_int ldu = 2;
    rocblas_int ldc = 2;

    // memory allocations
    device_strided_batch_vector<S> dD(1,1,1,1);
    device_strided_batch_vector<S> dE(1,1,1,1);
    device_strided_batch_vector<T> dV(1,1,1,1);
    device_strided_batch_vector<T> dU(1,1,1,1);
    device_strided_batch_vector<T> dC(1,1,1,1);
    device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());
    
    // check bad arguments
    bdsqr_checkBadArgs(handle,uplo,n,nv,nu,nc,dD.data(),dE.data(),dV.data(),ldv,dU.data(),ldu,dC.data(),ldc,dinfo.data());
}


template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getError(const rocblas_handle handle, 
                        const rocblas_fill uplo,
                        const rocblas_int n, 
                        const rocblas_int nv, 
                        const rocblas_int nu, 
                        const rocblas_int nc, 
                        Sd &dD,
                        Sd &dE,
                        Td &dV,
                        const rocblas_int ldv,
                        Td &dU, 
                        const rocblas_int ldu,
                        Td &dC, 
                        const rocblas_int ldc,
                        Ud &dinfo,
                        Sh &hD, 
                        Sh &hDres, 
                        Sh &hE, 
                        Sh &hEres, 
                        Th &hV,
                        Th &hVres,
                        Th &hU,
                        Th &hUres,
                        Th &hC,
                        Th &hCres,
                        Uh &hinfo,
                        double *max_err)
{
    typedef typename std::conditional<!is_complex<T>, T, decltype(std::real(T{}))>::type S;
    std::vector<S> hW(4*n);
    
    // input data initialization 
    rocblas_init<S>(hD, true);
    rocblas_init<S>(hE, true);

    // make V,U and C identities so that results are actually singular vectors of B
    if (nv > 0) {
        memset(hV[0], 0, ldv * nv * sizeof(T));
        for (rocblas_int i = 0; i < min(n,nv); ++i) 
            hV[0][i + i*ldv] = T(1.0);
    }            
    if (nu > 0) {
        memset(hU[0], 0, ldu * n * sizeof(T));
        for (rocblas_int i = 0; i < min(n,nu); ++i) 
            hU[0][i + i*ldu] = T(1.0);
    }            
    if (nc > 0) {
        memset(hC[0], 0, ldc * nc * sizeof(T));
        for (rocblas_int i = 0; i < min(n,nc); ++i) 
            hC[0][i + i*ldc] = T(1.0);
    }            
    
    // now copy to the GPU
    CHECK_HIP_ERROR(dD.transfer_from(hD));
    CHECK_HIP_ERROR(dE.transfer_from(hE));
    if (nv > 0) CHECK_HIP_ERROR(dV.transfer_from(hV));
    if (nu > 0) CHECK_HIP_ERROR(dU.transfer_from(hU));
    if (nc > 0) CHECK_HIP_ERROR(dC.transfer_from(hC));

rocblas_cout<<std::endl;
for (int i=0;i<n;++i) 
    rocblas_cout << hD[0][i] << " ";
rocblas_cout<<std::endl;
for (int i=0;i<n-1;++i) 
    rocblas_cout << hE[0][i] << " ";

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv, dU.data(), ldu, dC.data(), ldc, dinfo.data()));
    CHECK_HIP_ERROR(hDres.transfer_from(dD));
    CHECK_HIP_ERROR(hEres.transfer_from(dE));
    if (nv > 0) CHECK_HIP_ERROR(hVres.transfer_from(dV));
    if (nu > 0) CHECK_HIP_ERROR(hUres.transfer_from(dU));
    if (nc > 0) CHECK_HIP_ERROR(hCres.transfer_from(dC));

rocblas_cout<<std::endl;
rocblas_cout<<std::endl;
for (int i=0;i<n;++i) 
    rocblas_cout << hDres[0][i] << " ";
rocblas_cout<<std::endl;
for (int i=0;i<n-1;++i) 
    rocblas_cout << hEres[0][i] << " ";


    // CPU lapack
    cblas_bdsqr<T>(uplo,n,nv,nu,nc,hD[0],hE[0],hV[0],ldv,hU[0],ldu,hC[0],ldc,hW.data(),hinfo[0]);
   
    // error is 
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES. 
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F',1,n,1,hD[0],hDres[0]);
    *max_err = err > *max_err ? err : *max_err;
    if (nv > 0) {err = norm_error('F',n,nv,ldv,hV[0],hVres[0]);
    *max_err = err > *max_err ? err : *max_err;}
    if (nu > 0) {err = norm_error('F',nu,n,ldu,hU[0],hUres[0]);
    *max_err = err > *max_err ? err : *max_err;}
    if (nc > 0) {err = norm_error('F',n,nc,ldc,hC[0],hCres[0]);
    *max_err = err > *max_err ? err : *max_err;}
    if (hinfo[0][0] >  0) {err = norm_error('F',1,n-1,1,hE[0],hEres[0]);
    *max_err = err > *max_err ? err : *max_err;}    
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getPerfData(const rocblas_handle handle, 
                        const rocblas_fill uplo,
                        const rocblas_int n, 
                        const rocblas_int nv, 
                        const rocblas_int nu, 
                        const rocblas_int nc, 
                        Sd &dD,
                        Sd &dE,
                        Td &dV,
                        const rocblas_int ldv,
                        Td &dU, 
                        const rocblas_int ldu,
                        Td &dC, 
                        const rocblas_int ldc,
                        Ud &dinfo,
                        Sh &hD, 
                        Sh &hE, 
                        Th &hV,
                        Th &hU,
                        Th &hC,
                        Uh &hinfo,
                        double *gpu_time_used,
                        double *cpu_time_used,
                        const rocblas_int hot_calls)
{
    typedef typename std::conditional<!is_complex<T>, T, decltype(std::real(T{}))>::type S;
    std::vector<S> hW(4*n);
    
    // cpu-lapack performance
    *cpu_time_used = get_time_us();
    cblas_bdsqr<T>(uplo,n,nv,nu,nc,hD[0],hE[0],hV[0],ldv,hU[0],ldu,hC[0],ldc,hW.data(),hinfo[0]);
    *cpu_time_used = get_time_us() - *cpu_time_used;

    // cold calls
    for(int iter = 0; iter < 2; iter++)
        CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv, dU.data(), ldu, dC.data(), ldc, dinfo.data()));
        
    // gpu-lapack performance
    *gpu_time_used = get_time_us(); 
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
        rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv, dU.data(), ldu, dC.data(), ldc, dinfo.data());
    *gpu_time_used = (get_time_us() - *gpu_time_used) / hot_calls;
}


template <typename T> 
void testing_bdsqr(Arguments argus) 
{
    typedef typename std::conditional<!is_complex<T>, T, decltype(std::real(T{}))>::type S;
    
    // get arguments 
    rocblas_local_handle handle;
    rocblas_int n = argus.M;
    rocblas_int nv = argus.N;
    rocblas_int nu = argus.K;
    rocblas_int nc = argus.S4;
    rocblas_int ldv = argus.lda;
    rocblas_int ldu = argus.ldb;
    rocblas_int ldc = argus.ldc;
    char uploC = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;
    
    // check non-supported values 
    if (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower) {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,(S*)nullptr,(S*)nullptr,(T*)nullptr,ldv,(T*)nullptr,ldu,(T*)nullptr,ldc,(rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_D = n;
    size_t size_E = n - 1;
    size_t size_V = size_t(ldv)*nv;
    size_t size_U = size_t(ldu)*n;
    size_t size_C = size_t(ldc)*nc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0 ;

    // check invalid sizes 
    bool invalid_size = (n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1) ||
                        (nv > 0 && ldv < n) || (nc > 0 && ldc < n);
    if (invalid_size) {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,(S*)nullptr,(S*)nullptr,(T*)nullptr,ldv,(T*)nullptr,ldu,(T*)nullptr,ldc,(rocblas_int*)nullptr), 
                              rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D,1,size_D,1);
    host_strided_batch_vector<S> hE(size_E,1,size_E,1);
    host_strided_batch_vector<T> hV(size_V,1,size_V,1);
    host_strided_batch_vector<T> hU(size_U,1,size_U,1);
    host_strided_batch_vector<T> hC(size_C,1,size_C,1);
    host_strided_batch_vector<rocblas_int> hinfo(1,1,1,1);
    device_strided_batch_vector<S> dD(size_D,1,size_D,1);
    device_strided_batch_vector<S> dE(size_E,1,size_E,1);
    device_strided_batch_vector<T> dV(size_V,1,size_V,1);
    device_strided_batch_vector<T> dU(size_U,1,size_U,1);
    device_strided_batch_vector<T> dC(size_C,1,size_C,1);
    device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
    if (size_D) CHECK_HIP_ERROR(dD.memcheck());
    if (size_E) CHECK_HIP_ERROR(dE.memcheck());
    if (size_V) CHECK_HIP_ERROR(dV.memcheck());
    if (size_U) CHECK_HIP_ERROR(dU.memcheck());
    if (size_C) CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if (n == 0) {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle,uplo,n,nv,nu,nc,dD,dE,dV,ldv,dU,ldu,dC,ldc,dinfo), 
                              rocblas_status_success);
        if (argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if (argus.unit_check || argus.norm_check) {
        host_strided_batch_vector<S> hDres(size_D,1,size_D,1);
        host_strided_batch_vector<S> hEres(size_E,1,size_E,1);
        host_strided_batch_vector<T> hVres(size_V,1,size_V,1);
        host_strided_batch_vector<T> hUres(size_U,1,size_U,1);
        host_strided_batch_vector<T> hCres(size_C,1,size_C,1);
        bdsqr_getError<T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dinfo,
                          hD, hDres, hE, hEres, hV, hVres, hU, hUres, hC, hCres, hinfo, &max_error);
    }

    // collect performance data
    if (argus.timing) 
        bdsqr_getPerfData<T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dinfo,
                            hD, hE, hV, hU, hC, hinfo, &gpu_time_used, &cpu_time_used, hot_calls);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if (argus.unit_check) 
        rocsolver_test_check<T>(max_error,n);     

    // output results for rocsolver-bench
    if (argus.timing) {
        rocblas_cout << "\n============================================\n";
        rocblas_cout << "Arguments:\n";
        rocblas_cout << "============================================\n";
        rocsolver_bench_output("uplo", "n", "nv", "nu", "nc",  "ldv", "ldu", "ldc");
        rocsolver_bench_output(uploC, n, nv, nu, nc, ldv, ldu, ldc);
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
