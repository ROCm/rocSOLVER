/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename W>
void gesvdx_checkBadArgs(const rocblas_handle handle,
                         const rocblas_svect left_svect,
                         const rocblas_svect right_svect,
                         const rocblas_srange srange,
                         const rocblas_int m,
                         const rocblas_int n,
                         W dA,
                         const rocblas_int lda,
                         const rocblas_stride stA,
                         const S vl,
                         const S vu,
                         const rocblas_int il,
                         const rocblas_int iu,
                         rocblas_int* dNsv,
                         S* dS,
                         const rocblas_stride stS,
                         T dU,
                         const rocblas_int ldu,
                         const rocblas_stride stU,
                         T dV,
                         const rocblas_int ldv,
                         const rocblas_stride stV,
                         rocblas_int* difail,
                         const rocblas_stride stF,
                         rocblas_int* dinfo,
                         const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, nullptr, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU, ldu,
                                           stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, rocblas_svect_all, right_svect, srange,
                                           m, n, dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU,
                                           ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, rocblas_svect_all, srange,
                                           m, n, dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU,
                                           ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect,
                                           rocblas_srange(-1), m, n, dA, lda, stA, vl, vu, il, iu,
                                           dNsv, dS, stS, dU, ldu, stU, dV, ldv, stV, difail, stF,
                                           dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m,
                                               n, dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU,
                                               ldu, stU, dV, ldv, stV, difail, stF, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           (W) nullptr, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU,
                                           ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, (rocblas_int*)nullptr, dS,
                                           stS, dU, ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, (S*)nullptr, stS, dU,
                                           ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, (T) nullptr,
                                           ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU, ldu,
                                           stU, (T) nullptr, ldv, stV, difail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU, ldu,
                                           stU, dV, ldv, stV, (rocblas_int*)nullptr, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                           dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS, dU, ldu, stU,
                                           dV, ldv, stV, difail, stF, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, 0, n,
                                           (W) nullptr, lda, stA, vl, vu, il, iu, dNsv, (S*)nullptr,
                                           stS, (T) nullptr, ldu, stU, dV, ldv, stV,
                                           (rocblas_int*)nullptr, stF, dinfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, 0,
                                           (W) nullptr, lda, stA, vl, vu, il, iu, dNsv, (S*)nullptr,
                                           stS, (T) nullptr, ldu, stU, (T) nullptr, ldv, stV,
                                           (rocblas_int*)nullptr, stF, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m,
                                               n, dA, lda, stA, vl, vu, il, iu,
                                               (rocblas_int*)nullptr, dS, stS, dU, ldu, stU, dV,
                                               ldv, stV, difail, stF, (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvdx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_svect left_svect = rocblas_svect_singular;
    rocblas_svect right_svect = rocblas_svect_singular;
    rocblas_srange srange = rocblas_srange_all;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_int ldu = 2;
    rocblas_int ldv = 2;
    rocblas_stride stA = 2;
    rocblas_stride stS = 2;
    rocblas_stride stU = 2;
    rocblas_stride stV = 2;
    rocblas_stride stF = 2;
    rocblas_int bc = 1;
    S vl = 0;
    S vu = 0;
    rocblas_int il = 0;
    rocblas_int iu = 0;

    // memory allocations (all cases)
    device_strided_batch_vector<S> dS(1, 1, 1, 1);
    device_strided_batch_vector<T> dU(1, 1, 1, 1);
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dNsv(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> difail(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dNsv.memcheck());
    CHECK_HIP_ERROR(difail.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());

        // check bad arguments
        gesvdx_checkBadArgs<STRIDED>(handle, left_svect, right_svect, srange, m, n, dA.data(), lda,
                                     stA, vl, vu, il, iu, dNsv, dS.data(), stS, dU.data(), ldu, stU,
                                     dV.data(), ldv, stV, difail.data(), stF, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());

        // check bad arguments
        gesvdx_checkBadArgs<STRIDED>(handle, left_svect, right_svect, srange, m, n, dA.data(), lda,
                                     stA, vl, vu, il, iu, dNsv, dS.data(), stS, dU.data(), ldu, stU,
                                     dV.data(), ldv, stV, difail.data(), stF, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesvdx_initData(const rocblas_handle handle,
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
        rocblas_int nn = std::min(m, n);

        // construct well conditioned matrix A such that all singular values are in (0, 20)
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 10;
                    else
                    {
                        if(m >= n)
                        {
                            if(j == i + 1)
                                hA[b][i + j * lda] = (hA[b][i + j * lda] - 5) / 10;
                            else
                                hA[b][i + j * lda] = 0;
                        }
                        else
                        {
                            if(i == j + 1)
                                hA[b][i + j * lda] = (hA[b][i + j * lda] - 5) / 10;
                            else
                                hA[b][i + j * lda] = 0;
                        }
                    }
                }
                if(i == nn / 4 || i == nn / 2 || i == nn - 1 || i == nn / 7 || i == nn / 5
                   || i == nn / 3)
                    hA[b][i + i * lda] *= -1;
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

template <bool STRIDED,
          typename T,
          typename S,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvdx_getError(const rocblas_handle handle,
                     const rocblas_svect left_svect,
                     const rocblas_svect right_svect,
                     const rocblas_srange srange,
                     const rocblas_int m,
                     const rocblas_int n,
                     Wd& dA,
                     const rocblas_int lda,
                     const rocblas_stride stA,
                     const S vl,
                     const S vu,
                     const rocblas_int il,
                     const rocblas_int iu,
                     Id& dNsv,
                     Ud& dS,
                     const rocblas_stride stS,
                     Td& dU,
                     const rocblas_int ldu,
                     const rocblas_stride stU,
                     Td& dV,
                     const rocblas_int ldv,
                     const rocblas_stride stV,
                     Id& difail,
                     const rocblas_stride stF,
                     Id& dinfo,
                     const rocblas_int bc,
                     const rocblas_svect left_svectT,
                     const rocblas_svect right_svectT,
                     const rocblas_int mT,
                     const rocblas_int nT,
                     Td& dUT,
                     const rocblas_int lduT,
                     const rocblas_stride stUT,
                     Td& dVT,
                     const rocblas_int ldvT,
                     const rocblas_stride stVT,
                     Wh& hA,
                     Ih& hNsv,
                     Ih& NsvRes,
                     Uh& hS,
                     Uh& Sres,
                     Th& hU,
                     Th& Ures,
                     const rocblas_int ldures,
                     Th& hV,
                     Th& Vres,
                     const rocblas_int ldvres,
                     Ih& hifail,
                     Ih& ifailRes,
                     Ih& hinfo,
                     Ih& infoRes,
                     double* max_err,
                     double* max_errv)
{
    /** As per lapack's documentation, the following workspace size should work:
    rocblas_int minn = std::min(m,n);
    rocblas_int maxn = std::max(m,n);
    rocblas_int lwork = minn * minn + 6 * minn + maxn;
    rocblas_int lrwork = 17 * minn * minn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    ** but gesvdx_ gives ilegal value for lwork.

    ** Making the memory query to get the correct workspace dimension:
    std::vector<T> query(1);
    cblas_gesvdx<T>(left_svect, right_svect, srange, m, n, hA[0], lda, vl, vu, il, iu, hNsv[0], hS[0], hU[0], ldu, hV[0], ldv,
                       query.data(), -1, rwork.data(), hifail[0], hinfo[0]);
    rocblas_int lwork = int(std::real(query[0]));
    std::vector<T> work(lwork);
    ** and now gesvdx_ fails with seg fault. **/

    // (TODO: Need to confirm problem with gesvdx_ and report it)
    // WORKAROUND: for now, we will call gesvd_ to get all the singular values and
    // offset the result array according to srange, vl, vu, il, and iu. ifail cannot be verified for now.
    rocblas_int offset[bc];
    rocblas_int lwork = 5 * max(m, n);
    std::vector<T> work(lwork);
    std::vector<S> rwork(lwork);
    rocblas_int minn = std::min(m, n);

    // input data initialization
    std::vector<T> A(lda * n * bc);
    gesvdx_initData<true, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // execute computations:
    // complementary execution to compute all singular vectors if needed
    if(mT * nT > 0)
    {
        CHECK_ROCBLAS_ERROR(rocsolver_gesvdx(STRIDED, handle, left_svectT, right_svectT, srange, mT,
                                             nT, dA.data(), lda, stA, vl, vu, il, iu, dNsv.data(),
                                             dS.data(), stS, dUT.data(), lduT, stUT, dVT.data(),
                                             ldvT, stVT, difail.data(), stF, dinfo.data(), bc));

        if(left_svect == rocblas_svect_none && right_svect != rocblas_svect_none)
            CHECK_HIP_ERROR(Ures.transfer_from(dUT));
        if(right_svect == rocblas_svect_none && left_svect != rocblas_svect_none)
            CHECK_HIP_ERROR(Vres.transfer_from(dVT));
    }

    gesvdx_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        //cblas_gesvdx<T>(rocblas_svect_none, rocblas_svect_none, srange, m, n, hA[b], lda, vl, vu, il, iu, hNsv[b], hS[b], hU[b], ldu, hV[b], ldv,
        //               work.data(), lwork, rwork.data(), hifail[b], hinfo[b]);
        // WORKAROUND:
        cblas_gesvd<T>(rocblas_svect_none, rocblas_svect_none, m, n, hA[b], lda, hS[b], hU[b], ldu,
                       hV[b], ldv, work.data(), lwork, rwork.data(), hinfo[b]);
        hNsv[b][0] = 0;
        offset[b] = -1;
        if(srange == rocblas_srange_index)
        {
            offset[b] = minn - iu;
            hNsv[b][0] = iu - il + 1;
        }
        else if(srange == rocblas_srange_value)
        {
            for(int j = 0; j < minn; ++j)
            {
                if(hS[b][j] <= vu && hS[b][j] > vl)
                {
                    if(offset[b] == -1)
                        offset[b] = j;
                    hNsv[b][0]++;
                }
            }
        }
        else
        {
            offset[b] = 0;
            hNsv[b][0] = minn;
        }
    }

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                         dA.data(), lda, stA, vl, vu, il, iu, dNsv.data(),
                                         dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                         difail.data(), stF, dinfo.data(), bc));

    CHECK_HIP_ERROR(Sres.transfer_from(dS));
    CHECK_HIP_ERROR(NsvRes.transfer_from(dNsv));
    //CHECK_HIP_ERROR(ifailRes.transfer_from(difail));
    CHECK_HIP_ERROR(infoRes.transfer_from(dinfo));

    if(left_svect == rocblas_svect_singular)
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
    if(right_svect == rocblas_svect_singular)
        CHECK_HIP_ERROR(Vres.transfer_from(dV));

    // Check info and ifail for non-convergence
    // (We expect the used input matrices to always converge)
    *max_err = 0;
    *max_errv = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hinfo[b][0] != infoRes[b][0])
            *max_err += 1;
        //for(int j = 0; j < hNsv[b][0]; ++j)
        //{
        //    if(hifail[b][j] != ifailRes[b][j])
        //        *max_err += 1;
        //}
    }

    // Check number of returned singular values
    double err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hNsv[b][0] != NsvRes[b][0])
            err++;
    *max_err = err > *max_err ? err : *max_err;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        // error is ||hS - Sres||
        err = norm_error('F', 1, hNsv[b][0], 1, hS[b] + offset[b], Sres[b]); //WORKAROUND
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        if(hinfo[b][0] == 0 && (left_svect != rocblas_svect_none || right_svect != rocblas_svect_none))
        {
            err = 0;
            // check singular vectors implicitly (A*v_k = s_k*u_k)
            for(rocblas_int k = 0; k < hNsv[b][0]; ++k)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    T tmp = 0;
                    for(rocblas_int j = 0; j < n; ++j)
                        tmp += A[b * lda * n + i + j * lda] * sconj(Vres[b][k + j * ldvres]);
                    tmp -= Sres[b][k] * Ures[b][i + k * ldures];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err = std::sqrt(err) / double(snorm('F', m, n, A.data() + b * lda * n, lda));
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <bool STRIDED,
          typename T,
          typename S,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvdx_getPerfData(const rocblas_handle handle,
                        const rocblas_svect left_svect,
                        const rocblas_svect right_svect,
                        const rocblas_srange srange,
                        const rocblas_int m,
                        const rocblas_int n,
                        Wd& dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        const S vl,
                        const S vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        Id& dNsv,
                        Ud& dS,
                        const rocblas_stride stS,
                        Td& dU,
                        const rocblas_int ldu,
                        const rocblas_stride stU,
                        Td& dV,
                        const rocblas_int ldv,
                        const rocblas_stride stV,
                        Id& difail,
                        const rocblas_stride stF,
                        Id& dinfo,
                        const rocblas_int bc,
                        Wh& hA,
                        Ih& hNsv,
                        Uh& hS,
                        Th& hU,
                        Th& hV,
                        Ih& hifail,
                        Ih& hinfo,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    /** As per lapack's documentation, the following workspace size should work:
    rocblas_int minn = std::min(m,n);
    rocblas_int maxn = std::max(m,n);
    rocblas_int lwork = minn * minn + 6 * minn + maxn;
    rocblas_int lrwork = 17 * minn * minn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    ** but gesvdx_ gives ilegal value for lwork.

    ** Making the memory query to get the correct workspace dimension:
    std::vector<T> query(1);
    cblas_gesvdx<T>(left_svect, right_svect, srange, m, n, hA[0], lda, vl, vu, il, iu, hNsv[0], hS[0], hU[0], ldu, hV[0], ldv,
                       query.data(), -1, rwork.data(), hifail[0], hinfo[0]);
    rocblas_int lwork = int(std::real(query[0]));
    std::vector<T> work(lwork);
    ** and now gesvdx_ fails with seg fault. **/

    // (TODO: Need to confirm problem with gesvdx_ and report it.
    //  For now we cannot report cpu time)

    std::vector<T> A(lda * n * bc);

    if(!perf)
    {
        //gesvdx_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        //for(rocblas_int b = 0; b < bc; ++b)
        //    cblas_gesvdx<T>(left_svect, right_svect, srange, m, n, hA[b], lda, vl, vu, il, iu, hNsv[b], hS[b], hU[b], ldu, hV[b], ldv,
        //                   work.data(), lwork, rwork.data(), hifail[b], hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesvdx_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesvdx_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n,
                                             dA.data(), lda, stA, vl, vu, il, iu, dNsv.data(),
                                             dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv,
                                             stV, difail.data(), stF, dinfo.data(), bc));
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
        gesvdx_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_gesvdx(STRIDED, handle, left_svect, right_svect, srange, m, n, dA.data(), lda,
                         stA, vl, vu, il, iu, dNsv.data(), dS.data(), stS, dU.data(), ldu, stU,
                         dV.data(), ldv, stV, difail.data(), stF, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvdx(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char leftvC = argus.get<char>("left_svect");
    char rightvC = argus.get<char>("right_svect");
    char srangeC = argus.get<char>("srange");
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int nn = std::min(m, n);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldu = argus.get<rocblas_int>("ldu", m);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", nn);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stS = argus.get<rocblas_stride>("strideS", nn);
    rocblas_stride stF = argus.get<rocblas_stride>("strideF", nn);
    rocblas_stride stU = argus.get<rocblas_stride>("strideU", ldu * nn);
    rocblas_stride stV = argus.get<rocblas_stride>("strideV", ldv * n);
    S vl = S(argus.get<double>("vl", 0));
    S vu = S(argus.get<double>("vu", srangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", srangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", srangeC == 'I' ? 1 : 0);

    rocblas_svect leftv = char2rocblas_svect(leftvC);
    rocblas_svect rightv = char2rocblas_svect(rightvC);
    rocblas_srange srange = char2rocblas_srange(srangeC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(rightv == rocblas_svect_overwrite || leftv == rocblas_svect_overwrite
       || rightv == rocblas_svect_all || leftv == rocblas_svect_all)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n, (T* const*)nullptr,
                                 lda, stA, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS,
                                 (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                 (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n, (T*)nullptr, lda,
                                 stA, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS,
                                 (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                 (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    /** TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY, NOT EXPLICITLY COMPARING
        WITH LAPACK. SO, WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF
        THE RIGHT AND LEFT VECTORS. WHILE DOING THIS, IF MORE VECTORS THAN THE
        SPECIFIED IN THE MAIN CALL NEED TO BE COMPUTED, WE DO SO WITH AN EXTRA CALL **/

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
            leftvT = rocblas_svect_singular;
            lduT = m;
            mT = m;
            nT = n;
        }
        if(rightv == rocblas_svect_none)
        {
            rightvT = rocblas_svect_singular;
            ldvT = nn;
            mT = m;
            nT = n;
        }
    }

    // determine sizes
    rocblas_int ldures = 1;
    rocblas_int ldvres = 1;
    size_t size_Sres = 0;
    size_t size_Ures = 0;
    size_t size_Vres = 0;
    size_t size_ifailRes = 0;
    size_t size_UT = 0;
    size_t size_VT = 0;
    size_t size_A = size_t(lda) * n;
    size_t size_S = size_t(nn);
    size_t size_V = size_t(ldv) * n;
    size_t size_U = size_t(ldu) * nn;
    size_t size_ifail = nn;
    if(argus.unit_check || argus.norm_check)
    {
        size_ifailRes = nn;
        size_VT = size_t(ldvT) * n;
        size_UT = size_t(lduT) * nn;
        size_Sres = nn;
        if(svects)
        {
            if(leftv == rocblas_svect_none)
            {
                size_Ures = size_UT;
                ldures = lduT;
            }
            else
            {
                size_Ures = size_U;
                ldures = ldu;
            }

            if(rightv == rocblas_svect_none)
            {
                size_Vres = size_VT;
                ldvres = ldvT;
            }
            else
            {
                size_Vres = size_V;
                ldvres = ldv;
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
        || (leftv == rocblas_svect_singular && ldu < m)
        || (rightv == rocblas_svect_singular && ldv < nn)
        || (srange == rocblas_srange_value && vl >= vu)
        || (srange == rocblas_srange_index && (il < 1 || iu < 0))
        || (srange == rocblas_srange_index && (iu > n || (n > 0 && il > iu)));

    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n, (T* const*)nullptr,
                                 lda, stA, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS,
                                 (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                 (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n, (T*)nullptr, lda,
                                 stA, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS,
                                 (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                 (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
        {
            CHECK_ALLOC_QUERY(rocsolver_gesvdx(
                STRIDED, handle, leftv, rightv, srange, m, n, (T* const*)nullptr, lda, stA, vl, vu,
                il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu, stU, (T*)nullptr,
                ldv, stV, (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesvdx(
                STRIDED, handle, leftvT, rightvT, srange, mT, nT, (T* const*)nullptr, lda, stA, vl,
                vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, lduT, stUT,
                (T*)nullptr, ldvT, stVT, (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));
        }
        else
        {
            CHECK_ALLOC_QUERY(rocsolver_gesvdx(
                STRIDED, handle, leftv, rightv, srange, m, n, (T*)nullptr, lda, stA, vl, vu, il, iu,
                (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu, stU, (T*)nullptr, ldv,
                stV, (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesvdx(
                STRIDED, handle, leftvT, rightvT, srange, mT, nT, (T*)nullptr, lda, stA, vl, vu, il,
                iu, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, lduT, stUT, (T*)nullptr,
                ldvT, stVT, (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));
        }

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T> hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T> hU(size_U, 1, stU, bc);
    host_strided_batch_vector<rocblas_int> hNsv(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> NsvRes(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hifail(12 * nn, 1, stF, bc);
    host_strided_batch_vector<rocblas_int> ifailRes(size_ifailRes, 1, stF, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> infoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> Sres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<T> Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T> Ures(size_Ures, 1, stUres, bc);
    // device
    device_strided_batch_vector<S> dS(size_S, 1, stS, bc);
    device_strided_batch_vector<T> dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T> dU(size_U, 1, stU, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dNsv(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> difail(size_ifail, 1, stF, bc);
    device_strided_batch_vector<T> dVT(size_VT, 1, stVT, bc);
    device_strided_batch_vector<T> dUT(size_UT, 1, stUT, bc);
    if(size_VT)
        CHECK_HIP_ERROR(dVT.memcheck());
    if(size_UT)
        CHECK_HIP_ERROR(dUT.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_U)
        CHECK_HIP_ERROR(dU.memcheck());
    if(size_ifail)
        CHECK_HIP_ERROR(difail.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());
    CHECK_HIP_ERROR(dNsv.memcheck());

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
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n,
                                                   dA.data(), lda, stA, vl, vu, il, iu, dNsv.data(),
                                                   dS.data(), stS, dU.data(), ldu, stU, dV.data(),
                                                   ldv, stV, difail.data(), stF, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvdx_getError<STRIDED, T>(
                handle, leftv, rightv, srange, m, n, dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS,
                dU, ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc, leftvT, rightvT, mT, nT, dUT,
                lduT, stUT, dVT, ldvT, stVT, hA, hNsv, NsvRes, hS, Sres, hU, Ures, ldures, hV, Vres,
                ldvres, hifail, ifailRes, hinfo, infoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvdx_getPerfData<STRIDED, T>(handle, leftv, rightv, srange, m, n, dA, lda, stA, vl,
                                           vu, il, iu, dNsv, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                           difail, stF, dinfo, bc, hA, hNsv, hS, hU, hV, hifail,
                                           hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                                           argus.profile, argus.profile_kernels, argus.perf);
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
            EXPECT_ROCBLAS_STATUS(rocsolver_gesvdx(STRIDED, handle, leftv, rightv, srange, m, n,
                                                   dA.data(), lda, stA, vl, vu, il, iu, dNsv.data(),
                                                   dS.data(), stS, dU.data(), ldu, stU, dV.data(),
                                                   ldv, stV, difail.data(), stF, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvdx_getError<STRIDED, T>(
                handle, leftv, rightv, srange, m, n, dA, lda, stA, vl, vu, il, iu, dNsv, dS, stS,
                dU, ldu, stU, dV, ldv, stV, difail, stF, dinfo, bc, leftvT, rightvT, mT, nT, dUT,
                lduT, stUT, dVT, ldvT, stVT, hA, hNsv, NsvRes, hS, Sres, hU, Ures, ldures, hV, Vres,
                ldvres, hifail, ifailRes, hinfo, infoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvdx_getPerfData<STRIDED, T>(handle, leftv, rightv, srange, m, n, dA, lda, stA, vl,
                                           vu, il, iu, dNsv, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                           difail, stF, dinfo, bc, hA, hNsv, hS, hU, hV, hifail,
                                           hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                                           argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 2 * min(m, n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * min(m, n));
        if(svects)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 2 * min(m, n));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(svects)
            max_error = (max_error >= max_errorv) ? max_error : max_errorv;

        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "srange", "m", "n", "lda", "vl",
                                       "vu", "il", "iu", "strideS", "ldu", "strideU", "ldv",
                                       "strideV", "strideF", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, srangeC, m, n, lda, vl, vu, il, iu, stS,
                                       ldu, stU, ldv, stV, stF, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "srange", "m", "n", "lda",
                                       "strideA", "vl", "vu", "il", "iu", "strideS", "ldu",
                                       "strideU", "ldv", "strideV", "strideF", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, srangeC, m, n, lda, stA, vl, vu, il, iu,
                                       stS, ldu, stU, ldv, stV, stF, bc);
            }
            else
            {
                rocsolver_bench_output("left_svect", "right_svect", "srange", "m", "n", "lda", "vl",
                                       "vu", "il", "iu", "ldu", "ldv");
                rocsolver_bench_output(leftvC, rightvC, srangeC, m, n, lda, vl, vu, il, iu, ldu, ldv);
            }
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

#define EXTERN_TESTING_GESVDX(...) extern template void testing_gesvdx<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GESVDX, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)