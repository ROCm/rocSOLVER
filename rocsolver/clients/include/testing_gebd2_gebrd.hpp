/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, bool GEBRD, typename S, typename T, typename U>
void gebd2_gebrd_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              S dD,
                              const rocblas_stride stD,
                              S dE,
                              const rocblas_stride stE,
                              U dTauq,
                              const rocblas_stride stQ,
                              U dTaup,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, nullptr, m, n, dA, lda, stA, dD,
                                                stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTauq, stQ, dTaup, stP, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T) nullptr, lda, stA,
                                                dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA,
                                                (S) nullptr, stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                (S) nullptr, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                dE, stE, (U) nullptr, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                dE, stE, dTauq, stQ, (U) nullptr, stP, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, 0, n, (T) nullptr, lda, stA,
                                                (S) nullptr, stD, (S) nullptr, stE, (U) nullptr,
                                                stQ, (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, 0, (T) nullptr, lda, stA,
                                                (S) nullptr, stD, (S) nullptr, stE, (U) nullptr,
                                                stQ, (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTauq, stQ, dTaup, stP, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GEBRD, typename T>
void testing_gebd2_gebrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_stride stQ = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
        device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTauq.memcheck());
        CHECK_HIP_ERROR(dTaup.memcheck());

        // check bad arguments
        gebd2_gebrd_checkBadArgs<STRIDED, GEBRD>(handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTauq.data(), stQ, dTaup.data(),
                                                 stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
        device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTauq.memcheck());
        CHECK_HIP_ERROR(dTaup.memcheck());

        // check bad arguments
        gebd2_gebrd_checkBadArgs<STRIDED, GEBRD>(handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTauq.data(), stQ, dTaup.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Ud& dTauq,
                          const rocblas_stride stQ,
                          Ud& dTaup,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Sh& hD,
                          Sh& hE,
                          Uh& hTauq,
                          Uh& hTaup)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
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

template <bool STRIDED, bool GEBRD, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_getError(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Ud& dTauq,
                          const rocblas_stride stQ,
                          Ud& dTaup,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Sh& hD,
                          Sh& hE,
                          Uh& hTauq,
                          Uh& hTaup,
                          double* max_err)
{
    constexpr bool COMPLEX = is_complex<T>;
    constexpr bool VERIFY_IMPLICIT_TEST = false;

    std::vector<T> hW(max(m, n));

    // input data initialization
    gebd2_gebrd_initData<true, true, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ,
                                           dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

    // execute computations
    // use verify_implicit_test to check correctness of the implicit test using
    // CPU lapack
    if(!VERIFY_IMPLICIT_TEST)
    {
        // GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, dTauq.data(), stQ,
                                                  dTaup.data(), stP, bc));
        CHECK_HIP_ERROR(hARes.transfer_from(dA));
        CHECK_HIP_ERROR(hTauq.transfer_from(dTauq));
        CHECK_HIP_ERROR(hTaup.transfer_from(dTaup));
    }
    else
    {
        // CPU lapack
        for(rocblas_int b = 0; b < bc; ++b)
        {
            memcpy(hARes[b], hA[b], lda * n * sizeof(T));
            GEBRD
            ? cblas_gebrd<S, T>(m, n, hARes[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data(),
                                max(m, n))
            : cblas_gebd2<S, T>(m, n, hARes[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data());
        }
    }

    // reconstruct A from the factorization for implicit testing
    std::vector<T> vec(max(m, n));
    vec[0] = 1;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        T* a = hARes[b];
        T* tauq = hTauq[b];
        T* taup = hTaup[b];

        if(m >= n)
        {
            for(int j = n - 1; j >= 0; j--)
            {
                if(j < n - 1)
                {
                    if(COMPLEX)
                    {
                        cblas_lacgv(1, taup + j, 1);
                        cblas_lacgv(n - j - 1, a + j + (j + 1) * lda, lda);
                    }
                    for(int i = 1; i < n - j - 1; i++)
                    {
                        vec[i] = a[j + (j + i + 1) * lda];
                        a[j + (j + i + 1) * lda] = 0;
                    }
                    cblas_larf(rocblas_side_right, m - j, n - j - 1, vec.data(), 1, taup + j,
                               a + j + (j + 1) * lda, lda, hW.data());
                    if(COMPLEX)
                        cblas_lacgv(1, taup + j, 1);
                }

                for(int i = 1; i < m - j; i++)
                {
                    vec[i] = a[(j + i) + j * lda];
                    a[(j + i) + j * lda] = 0;
                }
                cblas_larf(rocblas_side_left, m - j, n - j, vec.data(), 1, tauq + j,
                           a + j + j * lda, lda, hW.data());
            }
        }
        else
        {
            for(int j = m - 1; j >= 0; j--)
            {
                if(j < m - 1)
                {
                    for(int i = 1; i < m - j - 1; i++)
                    {
                        vec[i] = a[(j + i + 1) + j * lda];
                        a[(j + i + 1) + j * lda] = 0;
                    }
                    cblas_larf(rocblas_side_left, m - j - 1, n - j, vec.data(), 1, tauq + j,
                               a + (j + 1) + j * lda, lda, hW.data());
                }

                if(COMPLEX)
                {
                    cblas_lacgv(1, taup + j, 1);
                    cblas_lacgv(n - j, a + j + j * lda, lda);
                }
                for(int i = 1; i < n - j; i++)
                {
                    vec[i] = a[j + (j + i) * lda];
                    a[j + (j + i) * lda] = 0;
                }
                cblas_larf(rocblas_side_right, m - j, n - j, vec.data(), 1, taup + j,
                           a + j + j * lda, lda, hW.data());
                if(COMPLEX)
                    cblas_lacgv(1, taup + j, 1);
            }
        }
    }

    // error is ||hA - hARes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool GEBRD, typename S, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Sd& dD,
                             const rocblas_stride stD,
                             Sd& dE,
                             const rocblas_stride stE,
                             Ud& dTauq,
                             const rocblas_stride stQ,
                             Ud& dTaup,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Sh& hD,
                             Sh& hE,
                             Uh& hTauq,
                             Uh& hTaup,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const bool perf)
{
    std::vector<T> hW(max(m, n));

    if(!perf)
    {
        gebd2_gebrd_initData<true, false, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                                stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GEBRD ? cblas_gebrd<S, T>(m, n, hA[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data(),
                                      max(m, n))
                  : cblas_gebd2<S, T>(m, n, hA[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data());
        }
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    gebd2_gebrd_initData<true, false, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                            stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gebd2_gebrd_initData<false, true, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                                stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        CHECK_ROCBLAS_ERROR(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, dTauq.data(), stQ,
                                                  dTaup.data(), stP, bc));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        gebd2_gebrd_initData<false, true, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                                stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        start = get_time_us();
        rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                              dE.data(), stE, dTauq.data(), stQ, dTaup.data(), stP, bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GEBRD, typename T>
void testing_gebd2_gebrd(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stD = argus.bsp;
    rocblas_stride stE = argus.bsp;
    rocblas_stride stQ = argus.bsp;
    rocblas_stride stP = argus.bsp;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = min(m, n);
    size_t size_E = min(m, n);
    size_t size_Q = min(m, n);
    size_t size_P = min(m, n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (S*)nullptr,
                                                        stD, (S*)nullptr, stE, (T*)nullptr, stQ,
                                                        (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T*)nullptr,
                                                        lda, stA, (S*)nullptr, stD, (S*)nullptr,
                                                        stE, (T*)nullptr, stQ, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hE(size_E, 1, stE, bc);
        host_strided_batch_vector<T> hTaup(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hTauq(size_Q, 1, stQ, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<T> dTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dTaup(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_D)
            CHECK_HIP_ERROR(dD.memcheck());
        if(size_E)
            CHECK_HIP_ERROR(dE.memcheck());
        if(size_Q)
            CHECK_HIP_ERROR(dTauq.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dTaup.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTauq.data(), stQ, dTaup.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebd2_gebrd_getError<STRIDED, GEBRD, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE,
                                                       dTauq, stQ, dTaup, stP, bc, hA, hARes, hD,
                                                       hE, hTauq, hTaup, &max_error);

        // collect performance data
        if(argus.timing)
            gebd2_gebrd_getPerfData<STRIDED, GEBRD, S, T>(
                handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc, hA, hD,
                hE, hTauq, hTaup, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hE(size_E, 1, stE, bc);
        host_strided_batch_vector<T> hTaup(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<T> dTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dTaup(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_D)
            CHECK_HIP_ERROR(dD.memcheck());
        if(size_E)
            CHECK_HIP_ERROR(dE.memcheck());
        if(size_Q)
            CHECK_HIP_ERROR(dTauq.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dTaup.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTauq.data(), stQ, dTaup.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebd2_gebrd_getError<STRIDED, GEBRD, S, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE,
                                                       dTauq, stQ, dTaup, stP, bc, hA, hARes, hD,
                                                       hE, hTauq, hTaup, &max_error);

        // collect performance data
        if(argus.timing)
            gebd2_gebrd_getPerfData<STRIDED, GEBRD, S, T>(
                handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc, hA, hD,
                hE, hTauq, hTaup, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
    }

    // validate results for rocsolver-test
    // using m*n * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, m * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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
