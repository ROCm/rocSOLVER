/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.h"
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
                            U dW,
                            const rocblas_stride stW,
                            rocblas_int* dInfo,
                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, nullptr, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, rocblas_eform(-1), jobz, uplo, n, dA,
                                              lda, stA, dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, rocblas_evect(-1), uplo, n,
                                              dA, lda, stA, dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, rocblas_evect_tridiagonal, uplo,
                                              n, dA, lda, stA, dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, rocblas_fill(-1), n, dA,
                                              lda, stA, dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda,
                                                  stA, dB, ldb, stB, dW, stW, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, (T) nullptr,
                                              lda, stA, dB, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              (T) nullptr, ldb, stB, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, (U) nullptr, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda, stA,
                                              dB, ldb, stB, dW, stW, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, 0, (T) nullptr,
                                              lda, stA, (T) nullptr, ldb, stB, (U) nullptr, stW,
                                              dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA, lda,
                                                  stA, dB, ldb, stB, dW, stW, dInfo, 0),
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
    rocblas_stride stW = 1;
    rocblas_int bc = 1;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_evect jobz = rocblas_evect_none;
    rocblas_fill uplo = rocblas_fill_upper;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygv_hegv_checkBadArgs<STRIDED>(handle, itype, jobz, uplo, n, dA.data(), lda, stA,
                                        dB.data(), ldb, stB, dW.data(), stW, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygv_hegv_checkBadArgs<STRIDED>(handle, itype, jobz, uplo, n, dA.data(), lda, stA,
                                        dB.data(), ldb, stB, dW.data(), stW, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygv_hegv_initData(const rocblas_handle handle,
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
                        Ud& dW,
                        const rocblas_stride stW,
                        Vd& dInfo,
                        const rocblas_int bc,
                        Th& hA,
                        Th& hATmp,
                        Th& hBTmp,
                        Th& hB,
                        Uh& hW,
                        Vh& hInfo)
{
    if(CPU)
    {
        rocblas_int info;
        rocblas_init<T>(hATmp, true);
        rocblas_init<T>(hBTmp, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // make A hermitian and scale to ensure positive definiteness
            cblas_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, n, n, n,
                       (T)1.0, hATmp[b], lda, hATmp[b], lda, (T)0.0, hA[b], lda);

            for(rocblas_int i = 0; i < n; i++)
                hA[b][i + i * lda] += 400;

            // make B hermitian and scale to ensure positive definiteness
            cblas_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, n, n, n,
                       (T)1.0, hBTmp[b], ldb, hBTmp[b], ldb, (T)0.0, hB[b], ldb);

            for(rocblas_int i = 0; i < n; i++)
                hB[b][i + i * lda] += 400;

            // store B in hBTmp
            for(rocblas_int i = 0; i < n; i++)
                for(rocblas_int j = 0; j < n; j++)
                    hBTmp[b][i + j * ldb] = hB[b][i + j * ldb];

            // apply Cholesky factorization to B
            cblas_potrf(uplo, n, hB[b], ldb, &info);
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
                        Ud& dW,
                        const rocblas_stride stW,
                        Vd& dInfo,
                        const rocblas_int bc,
                        Th& hA,
                        Th& hARes,
                        Th& hB,
                        Th& hBRes,
                        Uh& hW,
                        Uh& hWRes,
                        Vh& hInfo,
                        Vh& hInfoRes,
                        double* max_err)
{
    using S = decltype(std::real(T{}));

    // input data initialization
    sygv_hegv_initData<true, true, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB, dW,
                                      stW, dInfo, bc, hA, hARes, hB, hBRes, hW, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(), lda, stA,
                                            dB.data(), ldb, stB, dW.data(), stW, dInfo.data(), bc));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(jobz != rocblas_evect_none)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    if(jobz == rocblas_evect_none)
    {
        // only eigenvalues needed; can compare with LAPACK

        rocblas_int lwork = 2 * n - 1;
        rocblas_int lrwork = 3 * n - 2;
        std::vector<T> work(lwork);
        std::vector<S> rwork(lrwork);

        // CPU lapack
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_sygv_hegv<S, T>(itype, jobz, uplo, n, hA[b], lda, hB[b], ldb, hW[b], work.data(),
                                  lwork, rwork.data(), hInfo[b]);
        }

        // error is ||hW - hWRes|| / ||hW||
        // using frobenius norm
        double err;
        *max_err = 0;
        for(rocblas_int b = 0; b < bc; ++b)
        {
            err = norm_error('F', 1, n, 1, hW[b], hWRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }

        // also check info for non positive definite cases
        err = 0;
        for(rocblas_int b = 0; b < bc; ++b)
            if(hInfo[b][0] != hInfoRes[b][0])
                err++;
        *max_err += err;
    }
    else
    {
        // both eigenvalues and eigenvectors needed; need to implicitly test
        // eigenvectors due to non-uniqueness of eigenvectors under scaling
        T alpha;
        T beta = 0;

        // hA contains A and hBRes contains B
        // hARes contains eigenvectors x
        if(itype == rocblas_eform_ax)
        {
            // problem is A*x = (lambda)*B*x
            for(rocblas_int b = 0; b < bc; ++b)
            {
                // compute B*x and store in hB
                alpha = 1;
                cblas_symm_hemm<T>(rocblas_side_left, uplo, n, n, alpha, hA[b], lda, hARes[b], lda,
                                   beta, hB[b], ldb);

                // compute (1/lambda)*A*x and store in hBRes
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cblas_symv_hemv(uplo, n, alpha, hA[b], lda, hARes[b] + j * lda, 1, beta,
                                    hBRes[b] + j * ldb, 1);
                }
            }
        }
        else if(itype == rocblas_eform_abx)
        {
            // problem is A*B*x = (lambda)*x
            for(rocblas_int b = 0; b < bc; ++b)
            {
                // compute B*x and store in hB
                alpha = 1;
                cblas_symm_hemm<T>(rocblas_side_left, uplo, n, n, alpha, hBRes[b], ldb, hARes[b],
                                   lda, beta, hB[b], ldb);

                // compute (1/lambda)*A*B*x and store in hBRes
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cblas_symv_hemv(uplo, n, alpha, hA[b], lda, hB[b] + j * ldb, 1, beta,
                                    hBRes[b] + j * ldb, 1);
                }

                // move x into hB
                for(rocblas_int i = 0; i < n; i++)
                    for(rocblas_int j = 0; j < n; j++)
                        hB[b][i + j * ldb] = hARes[b][i + j * lda];
            }
        }
        else if(itype == rocblas_eform_bax)
        {
            // problem is B*A*x = (lambda)*x
            for(rocblas_int b = 0; b < bc; ++b)
            {
                // compute A*x and store in hB
                alpha = 1;
                cblas_symm_hemm<T>(rocblas_side_left, uplo, n, n, alpha, hA[b], lda, hARes[b], lda,
                                   beta, hB[b], ldb);

                // compute (1/lambda)*B*A*x and store in hA
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cblas_symv_hemv(uplo, n, alpha, hBRes[b], ldb, hB[b] + j * ldb, 1, beta,
                                    hA[b] + j * lda, 1);
                }

                // move x into hB and (1/lambda)*B*A*x into hBRes
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        hB[b][i + j * ldb] = hARes[b][i + j * lda];
                        hBRes[b][i + j * ldb] = hA[b][i + j * lda];
                    }
                }
            }
        }

        // error is ||hB - hBRes|| / ||hB||
        // using frobenius norm
        double err;
        *max_err = 0;
        for(rocblas_int b = 0; b < bc; ++b)
        {
            err = norm_error('F', n, n, ldb, hB[b], hBRes[b]);
            *max_err = err > *max_err ? err : *max_err;
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
                           Ud& dW,
                           const rocblas_stride stW,
                           Vd& dInfo,
                           const rocblas_int bc,
                           Th& hA,
                           Th& hATmp,
                           Th& hB,
                           Th& hBTmp,
                           Uh& hW,
                           Vh& hInfo,
                           double* gpu_time_used,
                           double* cpu_time_used,
                           const rocblas_int hot_calls,
                           const bool perf)
{
    using S = decltype(std::real(T{}));

    rocblas_int lwork = 2 * n - 1;
    rocblas_int lrwork = 3 * n - 2;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);

    if(!perf)
    {
        sygv_hegv_initData<true, false, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dW, stW, dInfo, bc, hA, hATmp, hB, hBTmp, hW, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cblas_sygv_hegv<S, T>(itype, jobz, uplo, n, hA[b], lda, hB[b], ldb, hW[b], work.data(),
                                  lwork, rwork.data(), hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygv_hegv_initData<true, false, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB, dW,
                                       stW, dInfo, bc, hA, hATmp, hB, hBTmp, hW, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygv_hegv_initData<false, true, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dW, stW, dInfo, bc, hA, hATmp, hB, hBTmp, hW, hInfo);

        CHECK_ROCBLAS_ERROR(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(),
                                                lda, stA, dB.data(), ldb, stB, dW.data(), stW,
                                                dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sygv_hegv_initData<false, true, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dW, stW, dInfo, bc, hA, hATmp, hB, hBTmp, hW, hInfo);

        start = get_time_us_sync(stream);
        rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n, dA.data(), lda, stA, dB.data(),
                            ldb, stB, dW.data(), stW, dInfo.data(), bc);
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
    rocblas_stride stW = argus.bsp;
    rocblas_int bc = argus.batch_count;
    char itypeC = argus.itype;
    char jobzC = argus.evect;
    char uploC = argus.uplo_option;
    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_evect jobz = char2rocblas_evect(jobzC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // hARes and hBRes should always be allocated (used in initData)
    rocblas_stride stARes = stA;
    rocblas_stride stBRes = stB;
    rocblas_stride stWRes = (argus.unit_check || argus.norm_check) ? stW : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    size_t size_W = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // hARes and hBRes should always be allocated (used in initData)
    size_t size_ARes = size_A;
    size_t size_BRes = size_B;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T* const*)nullptr, lda, stA,
                                                      (T* const*)nullptr, ldb, stB, (S*)nullptr,
                                                      stW, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                      (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sygv_hegv(
                STRIDED, handle, itype, jobz, uplo, n, (T* const*)nullptr, lda, stA,
                (T* const*)nullptr, ldb, stB, (S*)nullptr, stW, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                  (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                  (S*)nullptr, stW, (rocblas_int*)nullptr, bc));

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
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
        host_strided_batch_vector<S> hWRes(size_WRes, 1, stWRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      dA.data(), lda, stA, dB.data(), ldb, stB,
                                                      dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygv_hegv_getError<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dW, stW, dInfo, bc, hA, hARes, hB, hBRes, hW, hWRes,
                                           hInfo, hInfoRes, &max_error);

        // collect performance data
        if(argus.timing)
            sygv_hegv_getPerfData<STRIDED, T>(
                handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB, dW, stW, dInfo, bc, hA,
                hARes, hB, hBRes, hW, hInfo, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
        host_strided_batch_vector<S> hWRes(size_WRes, 1, stWRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygv_hegv(STRIDED, handle, itype, jobz, uplo, n,
                                                      dA.data(), lda, stA, dB.data(), ldb, stB,
                                                      dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygv_hegv_getError<STRIDED, T>(handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB,
                                           dW, stW, dInfo, bc, hA, hARes, hB, hBRes, hW, hWRes,
                                           hInfo, hInfoRes, &max_error);

        // collect performance data
        if(argus.timing)
            sygv_hegv_getPerfData<STRIDED, T>(
                handle, itype, jobz, uplo, n, dA, lda, stA, dB, ldb, stB, dW, stW, dInfo, bc, hA,
                hARes, hB, hBRes, hW, hInfo, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
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
                rocsolver_bench_output("itype", "jobz", "uplo", "n", "lda", "ldb", "strideW",
                                       "batch_c");
                rocsolver_bench_output(itypeC, jobzC, uploC, n, lda, ldb, stW, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "jobz", "uplo", "n", "lda", "ldb", "strideA",
                                       "strideB", "strideW", "batch_c");
                rocsolver_bench_output(itypeC, jobzC, uploC, n, lda, ldb, stA, stB, stW, bc);
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
