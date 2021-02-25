/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename U>
void sygv_hegv_checkBadArgs(const rocblas_handle handle,
                            const rocblas_eform itype,
                            const rocblas_evect jobz,
                            const rocblas_fill uplo,
                            const rocblas_int n,
                            T dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            T dB,
                            const rocblas_int ldb,
                            const rocblas_stride stB,
                            U dD,
                            const rocblas_stride stD,
                            U dE,
                            const rocblas_stride stE,
                            rocblas_int* dInfo,
                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, nullptr, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, rocblas_eform(-1), jobz, uplo, n, dA,
                                              lda, stA, dB, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, rocblas_evect(-1), uplo, n, dA,
                                              lda, stA, dB, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, rocblas_evect_tridiagonal,
                                              uplo, n, dA, lda, stA, dB, ldb, stB, dD, stD, dE, stE,
                                              dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, rocblas_fill(-1), n, dA,
                                              lda, stA, dB, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda,
                                                  stA, dB, ldb, stB, dD, stD, dE, stE, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, (T) nullptr,
                                              lda, stA, dB, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              (T) nullptr, ldb, stB, dD, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, (U) nullptr, stD, dE, stE, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, dD, stD, (U) nullptr, stE, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA, dB,
                                              ldb, stB, dD, stD, dE, stE, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, 0, (T) nullptr,
                                              lda, stA, (T) nullptr, ldb, stB, (U) nullptr, stD,
                                              (U) nullptr, stE, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda,
                                                  stA, dB, ldb, stB, dD, stD, dE, stE,
                                                  (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygv_hegv_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_int bc = 1;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_evect jobz = rocblas_evect_none;
    rocblas_fill uplo = rocblas_fill_upper;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygv_hegv_checkBadArgs<STRIDED>(handle, itype, jobz, uplo, n, dA.data(), lda, stA, dB.data(),
                                        ldb, stB, dD.data(), stD, dE.data(), stE, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygv_hegv_checkBadArgs<STRIDED>(handle, itype, jobz, uplo, n, dA.data(), lda, stA, dB.data(),
                                        ldb, stB, dD.data(), stD, dE.data(), stE, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygv_hegv_initData(const rocblas_handle handle,
                        const rocblas_eform itype,
                        const rocblas_evect jobz,
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
            if(test && jobz != rocblas_evect_none)
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

template <bool STRIDED, typename T, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygv_hegv_getError(const rocblas_handle handle,
                        const rocblas_eform itype,
                        const rocblas_evect jobz,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        Td& dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        Td& dB,
                        const rocblas_int ldb,
                        const rocblas_stride stB,
                        Ud& dD,
                        const rocblas_stride stD,
                        Ud& dE,
                        const rocblas_stride stE,
                        Vd& dInfo,
                        const rocblas_int bc,
                        Th& hA,
                        Th& hARes,
                        Th& hB,
                        Uh& hD,
                        Uh& hDRes,
                        Vh& hInfo,
                        Vh& hInfoRes,
                        double* max_err,
                        const bool singular)
{
    using S = decltype(std::real(T{}));
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);

    rocblas_int lwork = (is_complex<T> ? 2 * n - 1 : 3 * n - 1);
    rocblas_int lrwork = (is_complex<T> ? 3 * n - 2 : 0);
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);

    // input data initialization
    sygv_hegv_initData<true, true, T>(handle, itype, jobz, n, dA, lda, stA, dB, ldb, stB, bc, hA,
                                      hB, A, B, true, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(), lda,
                                            stA, dB.data(), ldb, stB, dD.data(), stD, dE.data(),
                                            stE, dInfo.data(), bc));

    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(jobz != rocblas_evect_none)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cblas_sygv_hegv(itype, jobz, uplo, n, hA[b], lda, hB[b], ldb, hD[b], work.data(), lwork,
                        rwork.data(), hInfo[b]);
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved.
    // We do test with indefinite matrices B).

    // check info for non-convergence and/or positive-definiteness
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hInfo[b][0] != hInfoRes[b][0])
            *max_err += 1;

    double err;

    if(jobz == rocblas_evect_none)
    {
        // only eigenvalues needed; can compare with LAPACK

        // error is ||hD - hDRes|| / ||hD||
        // using frobenius norm
        for(rocblas_int b = 0; b < bc; ++b)
        {
            if(hInfoRes[b][0] == 0)
            {
                err = norm_error('F', 1, n, 1, hD[b], hDRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
    else
    {
        // both eigenvalues and eigenvectors needed; need to implicitly test
        // eigenvectors due to non-uniqueness of eigenvectors under scaling

        for(rocblas_int b = 0; b < bc; ++b)
        {
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
                        alpha = T(1) / hDRes[b][j];
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
                        alpha = T(1) / hDRes[b][j];
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

template <bool STRIDED, typename T, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygv_hegv_getPerfData(const rocblas_handle handle,
                           const rocblas_eform itype,
                           const rocblas_evect jobz,
                           const rocblas_fill uplo,
                           const rocblas_int n,
                           Td& dA,
                           const rocblas_int lda,
                           const rocblas_stride stA,
                           Td& dB,
                           const rocblas_int ldb,
                           const rocblas_stride stB,
                           Ud& dD,
                           const rocblas_stride stD,
                           Ud& dE,
                           const rocblas_stride stE,
                           Vd& dInfo,
                           const rocblas_int bc,
                           Th& hA,
                           Th& hB,
                           Uh& hD,
                           Vh& hInfo,
                           double* gpu_time_used,
                           double* cpu_time_used,
                           const rocblas_int hot_calls,
                           const bool perf,
                           const bool singular)
{
    using S = decltype(std::real(T{}));
    host_strided_batch_vector<T> A(1, 1, 1, 1);
    host_strided_batch_vector<T> B(1, 1, 1, 1);

    rocblas_int lwork = (is_complex<T> ? 2 * n - 1 : 3 * n - 1);
    rocblas_int lrwork = (is_complex<T> ? 3 * n - 2 : 0);
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);

    if(!perf)
    {
        sygv_hegv_initData<true, false, T>(handle, itype, jobz, n, dA, lda, stA, dB, ldb, stB, bc,
                                           hA, hB, A, B, false, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_sygv_hegv<S, T>(itype, jobz, uplo, n, hA[b], lda, hB[b], ldb, hD[b], work.data(),
                                  lwork, rwork.data(), hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygv_hegv_initData<true, false, T>(handle, itype, jobz, n, dA, lda, stA, dB, ldb, stB, bc, hA,
                                       hB, A, B, false, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygv_hegv_initData<false, true, T>(handle, itype, jobz, n, dA, lda, stA, dB, ldb, stB, bc,
                                           hA, hB, A, B, false, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(),
                                                lda, stA, dB.data(), ldb, stB, dD.data(), stD,
                                                dE.data(), stE, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sygv_hegv_initData<false, true, T>(handle, itype, jobz, n, dA, lda, stA, dB, ldb, stB, bc,
                                           hA, hB, A, B, false, singular);

        start = get_time_us_sync(stream);
        rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(), lda, stA, dB.data(),
                            ldb, stB, dD.data(), stD, dE.data(), stE, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygv_hegv(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stB = argus.bsb;
    rocblas_stride stD = argus.bsc;
    rocblas_stride stE = argus.bsp;
    rocblas_int bc = argus.batch_count;
    char itypeC = argus.itype;
    char jobzC = argus.evect;
    char uploC = argus.uplo_option;
    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_evect jobz = char2rocblas_evect(jobzC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stDRes = (argus.unit_check || argus.norm_check) ? stD : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full || jobz == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T* const*)nullptr, lda, stA,
                                                      (T* const*)nullptr, ldb, stB, (S*)nullptr, stD,
                                                      (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                      (S*)nullptr, stD, (S*)nullptr, stE,
                                                      (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    size_t size_D = size_t(n);
    size_t size_E = size_D;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T* const*)nullptr, lda, stA,
                                                      (T* const*)nullptr, ldb, stB, (S*)nullptr, stD,
                                                      (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                      (S*)nullptr, stD, (S*)nullptr, stE,
                                                      (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory size query is necessary
    if(!USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                  (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                                                  ldb, stB, (S*)nullptr, stD, (S*)nullptr, stE,
                                                  (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, (T*)nullptr,
                                                  lda, stA, (T*)nullptr, ldb, stB, (S*)nullptr, stD,
                                                  (S*)nullptr, stE, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hDRes(size_DRes, 1, stDRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_D)
            CHECK_HIP_ERROR(dD.memcheck());
        if(size_E)
            CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(),
                                                      lda, stA, dB.data(), ldb, stB, dD.data(), stD,
                                                      dE.data(), stE, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygv_hegv_getError<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dD, stD, dE, stE, dInfo, bc, hA, hARes, hB, hD, hDRes,
                                           hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygv_hegv_getPerfData<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb,
                                              stB, dD, stD, dE, stE, dInfo, bc, hA, hB, hD, hInfo,
                                              &gpu_time_used, &cpu_time_used, hot_calls, argus.perf,
                                              argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hDRes(size_DRes, 1, stDRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_D)
            CHECK_HIP_ERROR(dD.memcheck());
        if(size_E)
            CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(),
                                                      lda, stA, dB.data(), ldb, stB, dD.data(), stD,
                                                      dE.data(), stE, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygv_hegv_getError<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dD, stD, dE, stE, dInfo, bc, hA, hARes, hB, hD, hDRes,
                                           hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygv_hegv_getPerfData<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb,
                                              stB, dD, stD, dE, stE, dInfo, bc, hA, hB, hD, hInfo,
                                              &gpu_time_used, &cpu_time_used, hot_calls, argus.perf,
                                              argus.singular);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Arguments:\n";
            rocsolver_cout << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("itype", "jobz", "uplo", "n", "lda", "ldb", "strideD",
                                       "strideE", "batch_c");
                rocsolver_bench_output(itypeC, jobzC, uploC, n, lda, ldb, stD, stE, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "jobz", "uplo", "n", "lda", "ldb", "strideA",
                                       "strideB", "strideD", "strideE", "batch_c");
                rocsolver_bench_output(itypeC, jobzC, uploC, n, lda, ldb, stA, stB, stD, stE, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "jobz", "uplo", "n", "lda", "ldb");
                rocsolver_bench_output(itypeC, jobzC, uploC, n, lda, ldb);
            }
            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Results:\n";
            rocsolver_cout << "============================================\n";
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
            rocsolver_cout << std::endl;
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
