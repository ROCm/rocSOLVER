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
    EXPECT_ROCBLAS_STATUS(
        rocsolver_bdsqr(nullptr, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dinfo),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, rocblas_fill_full, n, nv, nu, nc, dD, dE, dV, ldv,
                                          dU, ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S) nullptr, dE, dV, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, (S) nullptr, dV, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, (T) nullptr, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, (T) nullptr,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu,
                                          (T) nullptr, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                          ldc, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, 0, nv, nu, nc, (S) nullptr, (S) nullptr,
                                          (T) nullptr, ldv, (T) nullptr, ldu, (T) nullptr, ldc, dinfo),
                          rocblas_status_success);
}

template <typename T>
void testing_bdsqr_bad_arg()
{
    using S = decltype(std::real(T{}));

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
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dU(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    bdsqr_checkBadArgs(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv, dU.data(),
                       ldu, dC.data(), ldc, dinfo.data());
}

template <bool CPU, bool GPU, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nv,
                    const rocblas_int nu,
                    const rocblas_int nc,
                    Sd& dD,
                    Sd& dE,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dU,
                    const rocblas_int ldu,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Th& hV,
                    Th& hU,
                    Th& hC,
                    Uh& hInfo,
                    std::vector<S>& D,
                    std::vector<S>& E)
{
    if(CPU)
    {
        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, false);

        // Adding possible gaps to fully test the algorithm.
        for(rocblas_int i = 0; i < n - 1; ++i)
        {
            hE[0][i] -= 5;
            hD[0][i] -= 4;
        }
        hD[0][n - 1] -= 4;

        // (Forcing non-convergence expecting lapack and rocsolver to give
        // the same orthogonal equivalent matrix is not possible. Testing
        // implicitly the equivalent matrix is very complicated and it boils
        // down to essentially run the algorithm again and until convergence is achieved).

        // make copy of original data to test vectors if required
        if(nv || nu || nc)
        {
            for(rocblas_int i = 0; i < nv - 1; ++i)
            {
                E[i] = hE[0][i];
                D[i] = hD[0][i];
            }
            D[nv - 1] = hD[0][nv - 1];
        }

        // make V,U and C identities so that results are actually singular vectors
        // of B
        if(nv > 0)
        {
            memset(hV[0], 0, ldv * nv * sizeof(T));
            for(rocblas_int i = 0; i < min(n, nv); ++i)
                hV[0][i + i * ldv] = T(1.0);
        }
        if(nu > 0)
        {
            memset(hU[0], 0, ldu * n * sizeof(T));
            for(rocblas_int i = 0; i < min(n, nu); ++i)
                hU[0][i + i * ldu] = T(1.0);
        }
        if(nc > 0)
        {
            memset(hC[0], 0, ldc * nc * sizeof(T));
            for(rocblas_int i = 0; i < min(n, nc); ++i)
                hC[0][i + i * ldc] = T(1.0);
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        if(nv > 0)
            CHECK_HIP_ERROR(dV.transfer_from(hV));
        if(nu > 0)
            CHECK_HIP_ERROR(dU.transfer_from(hU));
        if(nc > 0)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nv,
                    const rocblas_int nu,
                    const rocblas_int nc,
                    Sd& dD,
                    Sd& dE,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dU,
                    const rocblas_int ldu,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hDRes,
                    Sh& hE,
                    Sh& hERes,
                    Th& hV,
                    Th& hU,
                    Th& hC,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err,
                    double* max_errv)
{
    using S = decltype(std::real(T{}));
    std::vector<S> hW(4 * n);
    std::vector<S> D(nv);
    std::vector<S> E(nv);

    // input data initialization
    bdsqr_initData<true, true, S, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc,
                                     dInfo, hD, hE, hV, hU, hC, hInfo, D, E);

    // execute computations
    // CPU lapack
    cblas_bdsqr<T>(uplo, n, nv, nu, nc, hD[0], hE[0], hV[0], ldv, hU[0], ldu, hC[0], ldc, hW.data(),
                   hInfo[0]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(),
                                        ldv, dU.data(), ldu, dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(nv > 0)
        CHECK_HIP_ERROR(hV.transfer_from(dV));
    if(nu > 0)
        CHECK_HIP_ERROR(hU.transfer_from(dU));
    if(nc > 0)
        CHECK_HIP_ERROR(hC.transfer_from(dC));

    // Check info for non-covergence
    *max_err = 0;
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;

    // (We expect the used input matrices to always converge. Testing
    // implicitely the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    // error is ||hD - hDRes||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    double err;
    T tmp;
    *max_errv = 0;
    err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // Check the singular vectors if required
    if(hInfo[0][0] == 0 && (nv || nu || nc))
    {
        err = 0;

        if(uplo == rocblas_fill_upper)
        {
            // check singular vectors implicitely (A'*u_i = s_i*v_i)
            for(rocblas_int i = 0; i < nv; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    if(i > 0)
                        tmp = D[i] * hU[0][i + j * ldu] + E[i - 1] * hU[0][(i - 1) + j * ldu]
                            - hDRes[0][j] * hV[0][j + i * ldv];
                    else
                        tmp = D[i] * hU[0][i + j * ldu] - hDRes[0][j] * hV[0][j + i * ldv];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
        }
        else
        {
            // check singular vectors implicitely (A*v_i = s_i*u_i)
            for(rocblas_int i = 0; i < nv; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    if(i > 0)
                        tmp = D[i] * hV[0][j + i * ldv] + E[i - 1] * hV[0][j + (i - 1) * ldv]
                            - hDRes[0][j] * hU[0][i + j * ldu];
                    else
                        tmp = D[i] * hV[0][j + i * ldv] - hDRes[0][j] * hU[0][i + j * ldu];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
        }

        double normD = double(snorm('F', 1, n, D.data(), 1));
        double normE = double(snorm('F', 1, n - 1, E.data(), 1));
        err = std::sqrt(err) / std::sqrt(normD * normD + normE * normE);
        *max_errv = err > *max_errv ? err : *max_errv;

        // C should be the transpose of U
        if(nc)
        {
            err = 0;
            for(rocblas_int i = 0; i < nv; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    tmp = hC[0][j + i * ldc] - hU[0][i + j * ldu];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err = std::sqrt(err);
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int nv,
                       const rocblas_int nu,
                       const rocblas_int nc,
                       Sd& dD,
                       Sd& dE,
                       Td& dV,
                       const rocblas_int ldv,
                       Td& dU,
                       const rocblas_int ldu,
                       Td& dC,
                       const rocblas_int ldc,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Th& hV,
                       Th& hU,
                       Th& hC,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    using S = decltype(std::real(T{}));
    std::vector<S> hW(4 * n);
    std::vector<S> D(nv);
    std::vector<S> E(nv);

    if(!perf)
    {
        bdsqr_initData<true, false, S, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                          ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        cblas_bdsqr<T>(uplo, n, nv, nu, nc, hD[0], hE[0], hV[0], ldv, hU[0], ldu, hC[0], ldc,
                       hW.data(), hInfo[0]);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    bdsqr_initData<true, false, S, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                      ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        bdsqr_initData<false, true, S, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                          ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E);

        CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(),
                                            dV.data(), ldv, dU.data(), ldu, dC.data(), ldc,
                                            dInfo.data()));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        bdsqr_initData<false, true, S, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                          ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E);

        start = get_time_us();
        rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv,
                        dU.data(), ldu, dC.data(), ldc, dInfo.data());
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_bdsqr(Arguments argus)
{
    using S = decltype(std::real(T{}));

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

    // size for testing singular vectors
    rocblas_int nT, nvT = 0, nuT = 0, ncT = 0, lduT = 1, ldcT = 1, ldvT = 1;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                              (rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    // (TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY (NOT EXPLICITLY COMPARING
    // WITH LAPACK)
    //  SO, WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF THE RIGHT AND
    //  LEFT VECTORS)
    if(nc)
    {
        nT = min(n, max(nc, max(nu, nv)));
        nuT = nT;
        nvT = nT;
        ncT = nT;
        ldvT = n;
        ldcT = n;
        lduT = nT;
    }
    else if(nv || nu)
    {
        nT = min(n, max(nv, nu));
        nuT = nT;
        nvT = nT;
        lduT = nT;
        ldvT = n;
    }
    // E, V, U, and C could have size zero in cases that are not quick-return or
    // invalid cases setting the size to one to avoid possible memory-access
    // errors in the rest of the unit test
    size_t size_D = size_t(n);
    size_t size_E = n > 1 ? size_t(n - 1) : 1;
    size_t size_V = max(size_t(ldv) * nv, 1);
    size_t size_U = max(size_t(ldu) * n, 1);
    size_t size_C = max(size_t(ldc) * nc, 1);
    size_t size_VT = max(size_t(ldvT) * nvT, 1);
    size_t size_UT = max(size_t(lduT) * n, 1);
    size_t size_CT = max(size_t(ldcT) * ncT, 1);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1)
        || (nv > 0 && ldv < n) || (nc > 0 && ldc < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                              (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(),
                                              (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                              dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(0);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        host_strided_batch_vector<S> hDRes(size_D, 1, size_D, 1);
        host_strided_batch_vector<S> hERes(size_E, 1, size_E, 1);
        host_strided_batch_vector<T> hV(size_VT, 1, size_VT, 1);
        host_strided_batch_vector<T> hU(size_UT, 1, size_UT, 1);
        host_strided_batch_vector<T> hC(size_CT, 1, size_CT, 1);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(size_VT, 1, size_VT, 1);
        device_strided_batch_vector<T> dU(size_UT, 1, size_UT, 1);
        device_strided_batch_vector<T> dC(size_CT, 1, size_CT, 1);
        if(size_VT)
            CHECK_HIP_ERROR(dV.memcheck());
        if(size_UT)
            CHECK_HIP_ERROR(dU.memcheck());
        if(size_CT)
            CHECK_HIP_ERROR(dC.memcheck());

        bdsqr_getError<T>(handle, uplo, n, nvT, nuT, ncT, dD, dE, dV, ldvT, dU, lduT, dC, ldcT,
                          dInfo, hD, hDRes, hE, hERes, hV, hU, hC, hInfo, hInfoRes, &max_error,
                          &max_errorv);
    }

    // collect performance data
    if(argus.timing)
    {
        host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
        host_strided_batch_vector<T> hU(size_U, 1, size_U, 1);
        host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
        device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
        device_strided_batch_vector<T> dU(size_U, 1, size_U, 1);
        device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
        if(size_V)
            CHECK_HIP_ERROR(dV.memcheck());
        if(size_U)
            CHECK_HIP_ERROR(dU.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());

        bdsqr_getPerfData<T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dInfo,
                             hD, hE, hV, hU, hC, hInfo, &gpu_time_used, &cpu_time_used, hot_calls,
                             argus.perf);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
    {
        rocsolver_test_check<T>(max_error, n);
        if(nv || nu || nc)
            rocsolver_test_check<T>(max_errorv, n);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            if(nv || nu || nc)
                max_error = (max_error >= max_errorv) ? max_error : max_errorv;
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("uplo", "n", "nv", "nu", "nc", "ldv", "ldu", "ldc");
            rocsolver_bench_output(uploC, n, nv, nu, nc, ldv, ldu, ldc);
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
