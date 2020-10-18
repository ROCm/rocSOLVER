/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename TT, typename W, typename U>
void gesvd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_svect left_svect,
                        const rocblas_svect right_svect,
                        const rocblas_int m,
                        const rocblas_int n,
                        W dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        TT dS,
                        const rocblas_stride stS,
                        T dU,
                        const rocblas_int ldu,
                        const rocblas_stride stU,
                        T dV,
                        const rocblas_int ldv,
                        const rocblas_stride stV,
                        TT dE,
                        const rocblas_stride stE,
                        const rocblas_workmode fa,
                        U dinfo,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, nullptr, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE, stE, fa,
                                          dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, rocblas_svect(-1), right_svect, m, n, dA,
                                          lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE, stE,
                                          fa, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, rocblas_svect(-1), m, n, dA,
                                          lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE, stE,
                                          fa, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, rocblas_svect_overwrite,
                                          rocblas_svect_overwrite, m, n, dA, lda, stA, dS, stS, dU,
                                          ldu, stU, dV, ldv, stV, dE, stE, fa, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA,
                                              lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE,
                                              stE, fa, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n,
                                          (W) nullptr, lda, stA, dS, stS, dU, ldu, stU, dV, ldv,
                                          stV, dE, stE, fa, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, (TT) nullptr, stS, dU, ldu, stU, dV, ldv, stV, dE,
                                          stE, fa, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, (T) nullptr, ldu, stU, dV, ldv, stV, dE,
                                          stE, fa, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, (T) nullptr, ldv, stV, dE,
                                          stE, fa, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, dV, ldv, stV, (TT) nullptr,
                                          stE, fa, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE, stE, fa,
                                          (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, 0, n,
                                          (W) nullptr, lda, stA, (TT) nullptr, stS, (T) nullptr,
                                          ldu, stU, dV, ldv, stV, (TT) nullptr, stE, fa, dinfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, 0,
                                          (W) nullptr, lda, stA, (TT) nullptr, stS, dU, ldu, stU,
                                          (T) nullptr, ldv, stV, (TT) nullptr, stE, fa, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA,
                                              lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dE,
                                              stE, fa, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_svect left_svect = rocblas_svect_all;
    rocblas_svect right_svect = rocblas_svect_all;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_int ldu = 2;
    rocblas_int ldv = 2;
    rocblas_stride stA = 2;
    rocblas_stride stS = 2;
    rocblas_stride stU = 2;
    rocblas_stride stV = 2;
    rocblas_stride stE = 2;
    rocblas_int bc = 1;
    rocblas_workmode fa = rocblas_outofplace;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesvd_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                                    dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                    dE.data(), stE, fa, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesvd_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                                    dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                    dE.data(), stE, fa, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesvd_initData(const rocblas_handle handle,
                    const rocblas_svect left_svect,
                    const rocblas_svect right_svect,
                    const rocblas_int m,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_int bc,
                    Th& hA,
                    std::vector<T>& A,
                    bool test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test && (left_svect != rocblas_svect_none || right_svect != rocblas_svect_none))
            {
                for(rocblas_int i = 0; i < m; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                        A[b * lda * n + i + j * lda] = hA[b][i + j * lda];
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, typename T, typename Wd, typename Td, typename Ud, typename Id, typename Wh, typename Th, typename Uh, typename Ih>
void gesvd_getError(const rocblas_handle handle,
                    const rocblas_svect left_svect,
                    const rocblas_svect right_svect,
                    const rocblas_int m,
                    const rocblas_int n,
                    Wd& dA,
                    const rocblas_int lda,
                    const rocblas_stride stA,
                    Td& dS,
                    const rocblas_stride stS,
                    Ud& dU,
                    const rocblas_int ldu,
                    const rocblas_stride stU,
                    Ud& dV,
                    const rocblas_int ldv,
                    const rocblas_stride stV,
                    Td& dE,
                    const rocblas_stride stE,
                    const rocblas_workmode fa,
                    Id& dinfo,
                    const rocblas_int bc,
                    const rocblas_svect left_svectT,
                    const rocblas_svect right_svectT,
                    const rocblas_int mT,
                    const rocblas_int nT,
                    Ud& dUT,
                    const rocblas_int lduT,
                    const rocblas_stride stUT,
                    Ud& dVT,
                    const rocblas_int ldvT,
                    const rocblas_stride stVT,
                    Wh& hA,
                    Th& hS,
                    Th& hSres,
                    Uh& hU,
                    Uh& Ures,
                    const rocblas_int ldures,
                    Uh& hV,
                    Uh& Vres,
                    const rocblas_int ldvres,
                    Th& hE,
                    Th& hEres,
                    Ih& hinfo,
                    Ih& hinfoRes,
                    double* max_err,
                    double* max_errv)
{
    rocblas_int lwork = 5 * max(m, n);
    std::vector<T> hWork(lwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    gesvd_initData<true, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // execute computations
    // complementary execution (to compute all singular vectors if needed)
    CHECK_ROCBLAS_ERROR(rocsolver_gesvd(
        STRIDED, handle, left_svectT, right_svectT, mT, nT, dA.data(), lda, stA, dS.data(), stS,
        dUT.data(), lduT, stUT, dVT.data(), ldvT, stVT, dE.data(), stE, fa, dinfo.data(), bc));

    if(left_svect == rocblas_svect_none && right_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Ures.transfer_from(dUT));
    if(right_svect == rocblas_svect_none && left_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Vres.transfer_from(dVT));

    gesvd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cblas_gesvd<T>(left_svect, right_svect, m, n, hA[b], lda, hS[b], hU[b], ldu, hV[b], ldv,
                       hWork.data(), lwork, hE[b], hinfo[b]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA.data(),
                                        lda, stA, dS.data(), stS, dU.data(), ldu, stU, dV.data(),
                                        ldv, stV, dE.data(), stE, fa, dinfo.data(), bc));

    CHECK_HIP_ERROR(hSres.transfer_from(dS));
    CHECK_HIP_ERROR(hEres.transfer_from(dE));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    if(left_svect == rocblas_svect_singular || left_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
    if(right_svect == rocblas_svect_singular || right_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Vres.transfer_from(dV));

    if(left_svect == rocblas_svect_overwrite)
    {
        CHECK_HIP_ERROR(hA.transfer_from(dA));
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < min(m, n); j++)
                    Ures[b][i + j * ldures] = hA[b][i + j * lda];
            }
        }
    }
    if(right_svect == rocblas_svect_overwrite)
    {
        CHECK_HIP_ERROR(hA.transfer_from(dA));
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < min(m, n); i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                    Vres[b][i + j * ldvres] = hA[b][i + j * lda];
            }
        }
    }

    // Check info for non-covergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;

    // (We expect the used input matrices to always converge. Testing
    // implicitely the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err;
    T tmp;
    *max_errv = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        // error is ||hS - hSres||
        err = norm_error('F', 1, min(m, n), 1, hS[b], hSres[b]);
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        if(hinfo[b][0] == 0 && (left_svect != rocblas_svect_none || right_svect != rocblas_svect_none))
        {
            err = 0;
            // check singular vectors implicitely (A*v_k = s_k*u_k)
            for(rocblas_int k = 0; k < min(m, n); ++k)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    tmp = 0;
                    for(rocblas_int j = 0; j < n; ++j)
                        tmp += A[b * lda * n + i + j * lda] * sconj(Vres[b][k + j * ldvres]);
                    tmp -= hSres[b][k] * Ures[b][i + k * ldures];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err = std::sqrt(err) / double(snorm('F', m, n, A.data() + b * lda * n, lda));
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <bool STRIDED, typename T, typename Wd, typename Td, typename Ud, typename Id, typename Wh, typename Th, typename Uh, typename Ih>
void gesvd_getPerfData(const rocblas_handle handle,
                       const rocblas_svect left_svect,
                       const rocblas_svect right_svect,
                       const rocblas_int m,
                       const rocblas_int n,
                       Wd& dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       Td& dS,
                       const rocblas_stride stS,
                       Ud& dU,
                       const rocblas_int ldu,
                       const rocblas_stride stU,
                       Ud& dV,
                       const rocblas_int ldv,
                       const rocblas_stride stV,
                       Td& dE,
                       const rocblas_stride stE,
                       const rocblas_workmode fa,
                       Id& dinfo,
                       const rocblas_int bc,
                       Wh& hA,
                       Th& hS,
                       Uh& hU,
                       Uh& hV,
                       Th& hE,
                       Ih& hinfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const bool perf)
{
    rocblas_int lwork = 5 * max(m, n);
    std::vector<T> hWork(lwork);
    std::vector<T> A;

    if(!perf)
    {
        gesvd_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(rocblas_int b = 0; b < bc; ++b)
            cblas_gesvd<T>(left_svect, right_svect, m, n, hA[b], lda, hS[b], hU[b], ldu, hV[b], ldv,
                           hWork.data(), lwork, hE[b], hinfo[b]);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    gesvd_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesvd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_gesvd(
            STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA, dS.data(), stS,
            dU.data(), ldu, stU, dV.data(), ldv, stV, dE.data(), stE, fa, dinfo.data(), bc));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        gesvd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        start = get_time_us();
        rocsolver_gesvd(STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                        dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV, dE.data(), stE,
                        fa, dinfo.data(), bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvd(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldu = argus.ldb;
    rocblas_int ldv = argus.ldv;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stS = argus.bsb;
    rocblas_stride stU = argus.bsc;
    rocblas_stride stV = argus.bsp;
    rocblas_stride stE = argus.bs5;
    rocblas_int bc = argus.batch_count;

    char faC = argus.workmode;
    char leftvC = argus.left_svect;
    char rightvC = argus.right_svect;
    rocblas_svect leftv = char2rocblas_svect(leftvC);
    rocblas_svect rightv = char2rocblas_svect(rightvC);
    rocblas_workmode fa = char2rocblas_workmode(faC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(rightv == rocblas_svect_overwrite && leftv == rocblas_svect_overwrite)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n,
                                                  (T* const*)nullptr, lda, stA, (S*)nullptr, stS,
                                                  (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                                  (S*)nullptr, stE, fa, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr,
                                                  lda, stA, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                                  (T*)nullptr, ldv, stV, (S*)nullptr, stE, fa,
                                                  (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // (TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY, NOT EXPLICITLY COMPARING
    // WITH LAPACK.
    // SO, WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF THE RIGHT AND
    // LEFT VECTORS)
    rocblas_svect leftvT = rocblas_svect_none;
    rocblas_svect rightvT = rocblas_svect_none;
    rocblas_int ldvT = 1;
    rocblas_int lduT = 1;
    rocblas_int mT = 0;
    rocblas_int nT = 0;
    bool svects = (leftv != rocblas_svect_none || rightv != rocblas_svect_none);
    if(svects)
    {
        if(leftv == rocblas_svect_none)
        {
            leftvT = rocblas_svect_all;
            lduT = m;
            mT = m;
            nT = n;
        }
        if(rightv == rocblas_svect_none)
        {
            rightvT = rocblas_svect_all;
            ldvT = n;
            mT = m;
            nT = n;
        }
    }

    // determine sizes
    rocblas_int ldures = 1;
    rocblas_int ldvres = 1;
    size_t size_Sres = 0;
    size_t size_Eres = 0;
    size_t size_Ures = 0;
    size_t size_Vres = 0;
    size_t size_UT = 0;
    size_t size_VT = 0;
    size_t size_A = size_t(lda) * n;
    size_t size_S = size_t(min(m, n));
    size_t size_E = size_S;
    size_t size_V = size_t(ldv) * n;
    size_t size_U = size_t(ldu) * m;
    if(argus.unit_check || argus.norm_check)
    {
        size_VT = size_t(ldvT) * nT;
        size_UT = size_t(lduT) * mT;
        size_Sres = size_S;
        size_Eres = size_E;
        if(svects)
        {
            if(leftv == rocblas_svect_none)
            {
                size_Ures = size_UT;
                ldures = lduT;
            }
            else if(leftv == rocblas_svect_singular || leftv == rocblas_svect_all)
            {
                size_Ures = size_U;
                ldures = ldu;
            }
            else
            {
                size_Ures = m * m;
                ldures = m;
            }

            if(rightv == rocblas_svect_none)
            {
                size_Vres = size_VT;
                ldvres = ldvT;
            }
            else if(rightv == rocblas_svect_singular || rightv == rocblas_svect_all)
            {
                size_Vres = size_V;
                ldvres = ldv;
            }
            else
            {
                size_Vres = n * n;
                ldvres = n;
            }
        }
    }
    rocblas_stride stUT = size_UT;
    rocblas_stride stVT = size_VT;
    rocblas_stride stUres = size_Ures;
    rocblas_stride stVres = size_Vres;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || bc < 0)
        || ((leftv == rocblas_svect_all || leftv == rocblas_svect_singular) && ldu < m)
        || ((rightv == rocblas_svect_all && ldv < n)
            || (rightv == rocblas_svect_singular && ldv < min(m, n)));

    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n,
                                                  (T* const*)nullptr, lda, stA, (S*)nullptr, stS,
                                                  (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                                  (S*)nullptr, stE, fa, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr,
                                                  lda, stA, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                                  (T*)nullptr, ldv, stV, (S*)nullptr, stE, fa,
                                                  (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hE(5 * max(m, n), 1, 5 * max(m, n), bc);
    host_strided_batch_vector<S> hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T> hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T> hU(size_U, 1, stU, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hSres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<S> hEres(size_Eres, 1, stE, bc);
    host_strided_batch_vector<T> Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T> Ures(size_Ures, 1, stUres, bc);
    // device
    device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
    device_strided_batch_vector<S> dS(size_S, 1, stS, bc);
    device_strided_batch_vector<T> dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T> dU(size_U, 1, stU, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T> dVT(size_VT, 1, stVT, bc);
    device_strided_batch_vector<T> dUT(size_UT, 1, stUT, bc);
    if(size_VT)
        CHECK_HIP_ERROR(dVT.memcheck());
    if(size_UT)
        CHECK_HIP_ERROR(dUT.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_U)
        CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n, dA.data(),
                                                  lda, stA, dS.data(), stS, dU.data(), ldu, stU,
                                                  dV.data(), ldv, stV, dE.data(), stE, fa,
                                                  dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvd_getError<STRIDED, T>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU, ldu,
                                       stU, dV, ldv, stV, dE, stE, fa, dinfo, bc, leftvT, rightvT,
                                       mT, nT, dUT, lduT, stUT, dVT, ldvT, stVT, hA, hS, hSres, hU,
                                       Ures, ldures, hV, Vres, ldvres, hE, hEres, hinfo, hinfoRes,
                                       &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvd_getPerfData<STRIDED, T>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU,
                                          ldu, stU, dV, ldv, stV, dE, stE, fa, dinfo, bc, hA, hS,
                                          hU, hV, hE, hinfo, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvd(STRIDED, handle, leftv, rightv, m, n, dA.data(),
                                                  lda, stA, dS.data(), stS, dU.data(), ldu, stU,
                                                  dV.data(), ldv, stV, dE.data(), stE, fa,
                                                  dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvd_getError<STRIDED, T>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU, ldu,
                                       stU, dV, ldv, stV, dE, stE, fa, dinfo, bc, leftvT, rightvT,
                                       mT, nT, dUT, lduT, stUT, dVT, ldvT, stVT, hA, hS, hSres, hU,
                                       Ures, ldures, hV, Vres, ldvres, hE, hEres, hinfo, hinfoRes,
                                       &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvd_getPerfData<STRIDED, T>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU,
                                          ldu, stU, dV, ldv, stV, dE, stE, fa, dinfo, bc, hA, hS,
                                          hU, hV, hE, hinfo, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        rocsolver_test_check<T>(max_error, min(m, n));
        if(svects)
            rocsolver_test_check<T>(max_errorv, min(m, n));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            if(svects)
                max_error = (max_error >= max_errorv) ? max_error : max_errorv;
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "strideS",
                                       "ldu", "strideU", "ldv", "strideV", "strideE", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, stS, ldu, stU, ldv, stV, stE, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "strideA",
                                       "strideS", "ldu", "strideU", "ldv", "strideV", "strideE",
                                       "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, stA, stS, ldu, stU, ldv, stV,
                                       stE, bc);
            }
            else
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "ldu", "ldv");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, ldu, ldv);
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
