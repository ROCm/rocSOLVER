/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename SS, typename U>
void sygvj_hegvj_checkBadArgs(const rocblas_handle handle,
                              const rocblas_eform itype,
                              const rocblas_evect evect,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              T dB,
                              const rocblas_int ldb,
                              const rocblas_stride stB,
                              const SS abstol,
                              S dResidual,
                              const rocblas_int max_sweeps,
                              U dSweeps,
                              S dW,
                              const rocblas_stride stW,
                              U dInfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, nullptr, itype, evect, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, abstol, dResidual, max_sweeps,
                                                dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, rocblas_eform(0), evect, uplo, n,
                                                dA, lda, stA, dB, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, rocblas_evect(0), uplo, n,
                                                dA, lda, stA, dB, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, rocblas_evect_tridiagonal,
                                                uplo, n, dA, lda, stA, dB, ldb, stB, abstol,
                                                dResidual, max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, rocblas_fill_full, n,
                                                dA, lda, stA, dB, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                    stA, dB, ldb, stB, abstol, dResidual,
                                                    max_sweeps, dSweeps, dW, stW, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, (T) nullptr,
                                                lda, stA, dB, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                stA, (T) nullptr, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, abstol, (S) nullptr, max_sweeps,
                                                dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, abstol, dResidual, max_sweeps,
                                                (U) nullptr, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, abstol, dResidual, max_sweeps,
                                                dSweeps, (S) nullptr, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                stA, dB, ldb, stB, abstol, dResidual, max_sweeps,
                                                dSweeps, dW, stW, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, 0, (T) nullptr,
                                                lda, stA, (T) nullptr, ldb, stB, abstol, dResidual,
                                                max_sweeps, dSweeps, (S) nullptr, stW, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA, lda,
                                                    stA, dB, ldb, stB, abstol, (S) nullptr,
                                                    max_sweeps, (U) nullptr, dW, stW, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvj_hegvj_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_stride stW = 1;
    rocblas_stride stE = 1;
    rocblas_int bc = 1;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_evect evect = rocblas_evect_none;
    rocblas_fill uplo = rocblas_fill_upper;

    S abstol = 0;
    rocblas_int max_sweeps = 100;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvj_hegvj_checkBadArgs<STRIDED>(handle, itype, evect, uplo, n, dA.data(), lda, stA,
                                          dB.data(), ldb, stB, abstol, dResidual.data(), max_sweeps,
                                          dSweeps.data(), dW.data(), stW, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvj_hegvj_checkBadArgs<STRIDED>(handle, itype, evect, uplo, n, dA.data(), lda, stA,
                                          dB.data(), ldb, stB, abstol, dResidual.data(), max_sweeps,
                                          dSweeps.data(), dW.data(), stW, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygvj_hegvj_initData(const rocblas_handle handle,
                          const rocblas_eform itype,
                          const rocblas_evect evect,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hB,
                          host_strided_batch_vector<T>& A,
                          host_strided_batch_vector<T>& B,
                          const bool test,
                          const bool singular)
{
    if(CPU)
    {
        rocblas_int info;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                    {
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 400;
                        hB[b][i + j * ldb] = std::real(hB[b][i + j * ldb]) + 400;
                    }
                    else
                    {
                        hA[b][i + j * lda] -= 4;
                    }
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // make some matrices B not positive definite
                // always the same elements for debugging purposes
                // the algorithm must detect the lower order of the principal minors <= 0
                // in those matrices in the batch that are non positive definite
                rocblas_int i = n / 4 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
                i = n / 2 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
                i = n - 1 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
            }

            // store A and B for testing purposes
            if(test && evect != rocblas_evect_none)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        if(itype != rocblas_eform_bax)
                        {
                            A[b][i + j * lda] = hA[b][i + j * lda];
                            B[b][i + j * ldb] = hB[b][i + j * ldb];
                        }
                        else
                        {
                            A[b][i + j * lda] = hB[b][i + j * ldb];
                            B[b][i + j * ldb] = hA[b][i + j * lda];
                        }
                    }
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void sygvj_hegvj_getError(const rocblas_handle handle,
                          const rocblas_eform itype,
                          const rocblas_evect evect,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          const S abstol,
                          Sd& dResidual,
                          const rocblas_int max_sweeps,
                          Id& dSweeps,
                          Sd& dW,
                          const rocblas_stride stW,
                          Id& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Th& hB,
                          Sh& hResidualRes,
                          Ih& hSweepsRes,
                          Sh& hW,
                          Sh& hWRes,
                          Ih& hInfo,
                          Ih& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    S atol = (abstol <= 0) ? get_epsilon<S>() : abstol;

    rocblas_int lwork = (COMPLEX ? 2 * n - 1 : 3 * n - 1);
    rocblas_int lrwork = (COMPLEX ? 3 * n - 2 : 0);
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);

    // input data initialization
    sygvj_hegvj_initData<true, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA,
                                        hB, A, B, true, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygvj_hegvj(
        STRIDED, handle, itype, evect, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB, abstol,
        dResidual.data(), max_sweeps, dSweeps.data(), dW.data(), stW, dInfo.data(), bc));

    CHECK_HIP_ERROR(hResidualRes.transfer_from(dResidual));
    CHECK_HIP_ERROR(hSweepsRes.transfer_from(dSweeps));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != rocblas_evect_none)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cblas_sygv_hegv<T>(itype, evect, uplo, n, hA[b], lda, hB[b], ldb, hW[b], work.data(), lwork,
                           rwork.data(), hInfo[b]);
    }

    // (We expect the used input matrices to always converge)
    // check info for non-convergence and/or positive-definiteness
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hInfo[b][0] != hInfoRes[b][0])
            *max_err += 1;

    // Also check validity of residual
    for(rocblas_int b = 0; b < bc; ++b)
        if(hInfoRes[b][0] == 0 && (hResidualRes[b][0] < 0 || hResidualRes[b][0] > atol))
            *max_err += 1;

    // Also check validity of sweeps
    for(rocblas_int b = 0; b < bc; ++b)
        if(hInfoRes[b][0] == 0 && (hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps))
            *max_err += 1;

    double err;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect == rocblas_evect_none)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hInfoRes[b][0] == 0)
            {
                err = norm_error('F', 1, n, 1, hW[b], hWRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hInfoRes[b][0] == 0)
            {
                T alpha = 1;
                T beta = 0;

                // hARes contains eigenvectors x
                // compute B*x (or A*x) and store in hB
                cblas_symm_hemm<T>(rocblas_side_left, uplo, n, n, alpha, B[b], ldb, hARes[b], lda,
                                   beta, hB[b], ldb);

                if(itype == rocblas_eform_ax)
                {
                    // problem is A*x = (lambda)*B*x

                    // compute (1/lambda)*A*x and store in hA
                    for(int j = 0; j < n; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cblas_symv_hemv(uplo, n, alpha, A[b], lda, hARes[b] + j * lda, 1, beta,
                                        hA[b] + j * lda, 1);
                    }

                    // move B*x into hARes
                    for(rocblas_int i = 0; i < n; i++)
                        for(rocblas_int j = 0; j < n; j++)
                            hARes[b][i + j * lda] = hB[b][i + j * ldb];
                }
                else
                {
                    // problem is A*B*x = (lambda)*x or B*A*x = (lambda)*x

                    // compute (1/lambda)*A*B*x or (1/lambda)*B*A*x and store in hA
                    for(int j = 0; j < n; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cblas_symv_hemv(uplo, n, alpha, A[b], lda, hB[b] + j * ldb, 1, beta,
                                        hA[b] + j * lda, 1);
                    }
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err = norm_error('F', n, n, lda, hA[b], hARes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void sygvj_hegvj_getPerfData(const rocblas_handle handle,
                             const rocblas_eform itype,
                             const rocblas_evect evect,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Td& dB,
                             const rocblas_int ldb,
                             const rocblas_stride stB,
                             const S abstol,
                             Sd& dResidual,
                             const rocblas_int max_sweeps,
                             Id& dSweeps,
                             Sd& dW,
                             const rocblas_stride stW,
                             Id& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Th& hB,
                             Sh& hW,
                             Ih& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const bool singular)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    rocblas_int lwork = (COMPLEX ? 2 * n - 1 : 3 * n - 1);
    rocblas_int lrwork = (COMPLEX ? 3 * n - 2 : 0);
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    host_strided_batch_vector<T> A(1, 1, 1, 1);
    host_strided_batch_vector<T> B(1, 1, 1, 1);

    if(!perf)
    {
        sygvj_hegvj_initData<true, false, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB,
                                             bc, hA, hB, A, B, false, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_sygv_hegv<T>(itype, evect, uplo, n, hA[b], lda, hB[b], ldb, hW[b], work.data(),
                               lwork, rwork.data(), hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygvj_hegvj_initData<true, false, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc,
                                         hA, hB, A, B, false, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygvj_hegvj_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB,
                                             bc, hA, hB, A, B, false, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sygvj_hegvj(
            STRIDED, handle, itype, evect, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB,
            abstol, dResidual.data(), max_sweeps, dSweeps.data(), dW.data(), stW, dInfo.data(), bc));
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
        sygvj_hegvj_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB,
                                             bc, hA, hB, A, B, false, singular);

        start = get_time_us_sync(stream);
        rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA.data(), lda, stA,
                              dB.data(), ldb, stB, abstol, dResidual.data(), max_sweeps,
                              dSweeps.data(), dW.data(), stW, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvj_hegvj(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char itypeC = argus.get<char>("itype");
    char evectC = argus.get<char>("evect");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideD", n);

    S abstol = S(argus.get<double>("abstol", 0));
    rocblas_int max_sweeps = argus.get<rocblas_int>("max_sweeps", 100);

    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stWRes = (argus.unit_check || argus.norm_check) ? stW : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, (T* const*)nullptr,
                                      lda, stA, (T* const*)nullptr, ldb, stB, abstol, (S*)nullptr,
                                      max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stW,
                                      (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                        stB, abstol, (S*)nullptr, max_sweeps,
                                                        (rocblas_int*)nullptr, (S*)nullptr, stW,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    size_t size_W = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, (T* const*)nullptr,
                                      lda, stA, (T* const*)nullptr, ldb, stB, abstol, (S*)nullptr,
                                      max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stW,
                                      (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                        stB, abstol, (S*)nullptr, max_sweeps,
                                                        (rocblas_int*)nullptr, (S*)nullptr, stW,
                                                        (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sygvj_hegvj(
                STRIDED, handle, itype, evect, uplo, n, (T* const*)nullptr, lda, stA,
                (T* const*)nullptr, ldb, stB, abstol, (S*)nullptr, max_sweeps,
                (rocblas_int*)nullptr, (S*)nullptr, stW, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygvj_hegvj(
                STRIDED, handle, itype, evect, uplo, n, (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                stB, abstol, (S*)nullptr, max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stW,
                (rocblas_int*)nullptr, bc));

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
    host_strided_batch_vector<S> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
    host_strided_batch_vector<S> hWRes(size_WRes, 1, stWRes, bc);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<S> dResidual(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, bc);
    device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    CHECK_HIP_ERROR(dResidual.memcheck());
    CHECK_HIP_ERROR(dSweeps.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA.data(), lda, stA,
                                      dB.data(), ldb, stB, abstol, dResidual.data(), max_sweeps,
                                      dSweeps.data(), dW.data(), stW, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvj_hegvj_getError<STRIDED, T>(handle, itype, evect, uplo, n, dA, lda, stA, dB, ldb,
                                             stB, abstol, dResidual, max_sweeps, dSweeps, dW, stW,
                                             dInfo, bc, hA, hARes, hB, hResidualRes, hSweepsRes, hW,
                                             hWRes, hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygvj_hegvj_getPerfData<STRIDED, T>(handle, itype, evect, uplo, n, dA, lda, stA, dB,
                                                ldb, stB, abstol, dResidual, max_sweeps, dSweeps, dW,
                                                stW, dInfo, bc, hA, hB, hW, hInfo, &gpu_time_used,
                                                &cpu_time_used, hot_calls, argus.profile,
                                                argus.profile_kernels, argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvj_hegvj(STRIDED, handle, itype, evect, uplo, n, dA.data(), lda, stA,
                                      dB.data(), ldb, stB, abstol, dResidual.data(), max_sweeps,
                                      dSweeps.data(), dW.data(), stW, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvj_hegvj_getError<STRIDED, T>(handle, itype, evect, uplo, n, dA, lda, stA, dB, ldb,
                                             stB, abstol, dResidual, max_sweeps, dSweeps, dW, stW,
                                             dInfo, bc, hA, hARes, hB, hResidualRes, hSweepsRes, hW,
                                             hWRes, hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygvj_hegvj_getPerfData<STRIDED, T>(handle, itype, evect, uplo, n, dA, lda, stA, dB,
                                                ldb, stB, abstol, dResidual, max_sweeps, dSweeps, dW,
                                                stW, dInfo, bc, hA, hB, hW, hInfo, &gpu_time_used,
                                                &cpu_time_used, hot_calls, argus.profile,
                                                argus.profile_kernels, argus.perf, argus.singular);
    }

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("itype", "evect", "uplo", "n", "lda", "ldb", "abstol",
                                       "max_sweeps", "strideW", "batch_c");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb, abstol, max_sweeps, stW,
                                       bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "evect", "uplo", "n", "lda", "ldb", "strideA",
                                       "strideB", "abstol", "max_sweeps", "strideW", "batch_c");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb, stA, stB, abstol,
                                       max_sweeps, stW, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "evect", "uplo", "n", "lda", "ldb", "abstol",
                                       "max_sweeps");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb, abstol, max_sweeps);
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

#define EXTERN_TESTING_SYGVJ_HEGVJ(...) \
    extern template void testing_sygvj_hegvj<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYGVJ_HEGVJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
