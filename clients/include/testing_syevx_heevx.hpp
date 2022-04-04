/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename SS, typename U>
void syevx_heevx_checkBadArgs(const rocblas_handle handle,
                              const rocblas_evect evect,
                              const rocblas_erange erange,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              const SS vl,
                              const SS vu,
                              const rocblas_int il,
                              const rocblas_int iu,
                              const SS abstol,
                              U dNev,
                              S dW,
                              const rocblas_stride stW,
                              T dZ,
                              const rocblas_int ldz,
                              const rocblas_stride stZ,
                              U dIfail,
                              const rocblas_stride stF,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, nullptr, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, dNev, dW, stW, dZ, ldz,
                                                stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, rocblas_evect(-1), erange, uplo, n,
                                                dA, lda, stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                                                dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, rocblas_erange(-1), uplo, n,
                                                dA, lda, stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                                                dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, rocblas_fill_full,
                                                n, dA, lda, stA, vl, vu, il, iu, abstol, dNev, dW,
                                                stW, dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA,
                                                    lda, stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                                                    dZ, ldz, stZ, dIfail, stF, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n,
                                                (T) nullptr, lda, stA, vl, vu, il, iu, abstol, dNev,
                                                dW, stW, dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, (U) nullptr, dW, stW,
                                                dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, dNev, (S) nullptr, stW,
                                                dZ, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                                                (T) nullptr, ldz, stZ, dIfail, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, dNev, dW, stW, dZ, ldz,
                                                stZ, (U) nullptr, stF, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                stA, vl, vu, il, iu, abstol, dNev, dW, stW, dZ, ldz,
                                                stZ, dIfail, stF, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, 0,
                                                (T) nullptr, lda, stA, vl, vu, il, iu, abstol, dNev,
                                                (S) nullptr, stW, (T) nullptr, ldz, stZ,
                                                (U) nullptr, stF, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA, lda,
                                                    stA, vl, vu, il, iu, abstol, (U) nullptr, dW,
                                                    stW, dZ, ldz, stZ, dIfail, stF, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevx_heevx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_original;
    rocblas_erange erange = rocblas_erange_value;
    rocblas_fill uplo = rocblas_fill_lower;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldz = 1;
    rocblas_stride stA = 1;
    rocblas_stride stW = 1;
    rocblas_stride stZ = 1;
    rocblas_stride stF = 1;
    rocblas_int bc = 1;

    S vl = 0.0;
    S vu = 1.0;
    rocblas_int il = 0;
    rocblas_int iu = 0;
    S abstol = 0;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dZ(1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIfail(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dNev.memcheck());
        CHECK_HIP_ERROR(dIfail.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syevx_heevx_checkBadArgs<STRIDED>(handle, evect, erange, uplo, n, dA.data(), lda, stA, vl,
                                          vu, il, iu, abstol, dNev.data(), dW.data(), stW,
                                          dZ.data(), ldz, stZ, dIfail.data(), stF, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dZ(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIfail(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dNev.memcheck());
        CHECK_HIP_ERROR(dIfail.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syevx_heevx_checkBadArgs<STRIDED>(handle, evect, erange, uplo, n, dA.data(), lda, stA, vl,
                                          vu, il, iu, abstol, dNev.data(), dW.data(), stW,
                                          dZ.data(), ldz, stZ, dIfail.data(), stF, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevx_heevx_initData(const rocblas_handle handle,
                          const rocblas_evect evect,
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

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 5;
                }
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
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

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevx_heevx_getError(const rocblas_handle handle,
                          const rocblas_evect evect,
                          const rocblas_erange erange,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          const S vl,
                          const S vu,
                          const rocblas_int il,
                          const rocblas_int iu,
                          const S abstol,
                          Id& dNev,
                          Sd& dW,
                          const rocblas_stride stW,
                          Td& dZ,
                          const rocblas_int ldz,
                          const rocblas_stride stZ,
                          Id& dIfail,
                          const rocblas_stride stF,
                          Id& dinfo,
                          const rocblas_int bc,
                          Th& hA,
                          Ih& hNev,
                          Ih& hNevRes,
                          Sh& hW,
                          Sh& hWRes,
                          Th& hZ,
                          Th& hZRes,
                          Ih& hIfail,
                          Ih& hIfailRes,
                          Ih& hinfo,
                          Ih& hinfoRes,
                          double* max_err)
{
    constexpr bool COMPLEX = is_complex<T>;

    int lwork = !COMPLEX ? 8 * n : 2 * n;
    int lrwork = !COMPLEX ? 0 : 7 * n;
    int liwork = 5 * n;

    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    syevx_heevx_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_syevx_heevx(
        STRIDED, handle, evect, erange, uplo, n, dA.data(), lda, stA, vl, vu, il, iu, abstol,
        dNev.data(), dW.data(), stW, dZ.data(), ldz, stZ, dIfail.data(), stF, dinfo.data(), bc));

    CHECK_HIP_ERROR(hNevRes.transfer_from(dNev));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == rocblas_evect_original)
    {
        CHECK_HIP_ERROR(hZRes.transfer_from(dZ));
        CHECK_HIP_ERROR(hIfailRes.transfer_from(dIfail));
    }

    // CPU lapack
    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    S atol = (abstol == 0) ? 2 * get_safemin<S>() : abstol;
    for(rocblas_int b = 0; b < bc; ++b)
        cblas_syevx_heevx<T>(evect, erange, uplo, n, hA[b], lda, vl, vu, il, iu, atol, hNev[b],
                             hW[b], hZ[b], ldz, work.data(), lwork, rwork.data(), iwork.data(),
                             hIfail[b], hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;

    // Check number of returned eigenvalues
    double err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hNev[b][0] != hNevRes[b][0])
            err++;
    *max_err = err > *max_err ? err : *max_err;

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect != rocblas_evect_original)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hinfo[b][0] == 0)
                err = norm_error('F', 1, hNev[b][0], 1, hW[b], hWRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hinfo[b][0] == 0)
            {
                // check ifail
                err = 0;
                for(int j = 0; j < hNev[b][0]; j++)
                {
                    if(hIfailRes[b][j] != 0)
                        err++;
                }
                *max_err = err > *max_err ? err : *max_err;

                // multiply A with each of the nev eigenvectors and divide by corresponding
                // eigenvalues
                T alpha;
                T beta = 0;
                for(int j = 0; j < hNev[b][0]; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cblas_symv_hemv(uplo, n, alpha, A.data() + b * lda * n, lda, hZRes[b] + j * ldz,
                                    1, beta, hZ[b] + j * ldz, 1);
                }

                // error is ||hZ - hZRes|| / ||hZ||
                // using frobenius norm
                err = norm_error('F', n, hNev[b][0], ldz, hZ[b], hZRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
            else
            {
                // check ifail
                err = 0;
                for(int j = 0; j < hinfo[b][0]; j++)
                {
                    if(hIfailRes[b][j] == 0)
                        err++;
                }
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevx_heevx_getPerfData(const rocblas_handle handle,
                             const rocblas_evect evect,
                             const rocblas_erange erange,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             const S vl,
                             const S vu,
                             const rocblas_int il,
                             const rocblas_int iu,
                             const S abstol,
                             Id& dNev,
                             Sd& dW,
                             const rocblas_stride stW,
                             Td& dZ,
                             const rocblas_int ldz,
                             const rocblas_stride stZ,
                             Id& dIfail,
                             const rocblas_stride stF,
                             Id& dinfo,
                             const rocblas_int bc,
                             Th& hA,
                             Ih& hNev,
                             Sh& hW,
                             Th& hZ,
                             Ih& hIfail,
                             Ih& hinfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool perf)
{
    constexpr bool COMPLEX = is_complex<T>;

    int lwork = !COMPLEX ? 8 * n : 2 * n;
    int lrwork = !COMPLEX ? 0 : 7 * n;
    int liwork = 5 * n;

    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);
    std::vector<T> A;

    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    S atol = (abstol == 0) ? 2 * get_safemin<S>() : abstol;

    if(!perf)
    {
        syevx_heevx_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cblas_syevx_heevx<T>(evect, erange, uplo, n, hA[b], lda, vl, vu, il, iu, atol, hNev[b],
                                 hW[b], hZ[b], ldz, work.data(), lwork, rwork.data(), iwork.data(),
                                 hIfail[b], hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syevx_heevx_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syevx_heevx_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_syevx_heevx(
            STRIDED, handle, evect, erange, uplo, n, dA.data(), lda, stA, vl, vu, il, iu, abstol,
            dNev.data(), dW.data(), stW, dZ.data(), ldz, stZ, dIfail.data(), stF, dinfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        syevx_heevx_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, dA.data(), lda, stA, vl, vu,
                              il, iu, abstol, dNev.data(), dW.data(), stW, dZ.data(), ldz, stZ,
                              dIfail.data(), stF, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevx_heevx(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    char erangeC = argus.get<char>("erange");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldz = argus.get<rocblas_int>("ldz", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);
    rocblas_stride stZ = argus.get<rocblas_stride>("strideZ", ldz * n);
    rocblas_stride stF = argus.get<rocblas_stride>("strideF", n);

    S vl = argus.get<S>("vl", 0);
    S vu = argus.get<S>("vu", erangeC == 'V' ? 1 : 0);
    rocblas_int il = argus.get<rocblas_int>("il", erangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", erangeC == 'I' ? 1 : 0);
    S abstol = argus.get<S>("abstol", 0);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_erange erange = char2rocblas_erange(erangeC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, (T* const*)nullptr,
                                      lda, stA, vl, vu, il, iu, abstol, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (T* const*)nullptr, ldz, stZ,
                                      (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, (T*)nullptr, lda,
                                      stA, vl, vu, il, iu, abstol, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (T*)nullptr, ldz, stZ,
                                      (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_W = n;
    size_t size_Z = size_t(ldz) * n;
    size_t size_ifail = n;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;
    size_t size_ZRes = (argus.unit_check || argus.norm_check) ? size_Z : 0;
    size_t size_ifailRes = (argus.unit_check || argus.norm_check) ? size_ifail : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || (evect != rocblas_evect_none && ldz < n) || bc < 0
                         || (erange == rocblas_erange_value && vl >= vu)
                         || (erange == rocblas_erange_index && (il < 1 || iu < 0))
                         || (erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu))));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, (T* const*)nullptr,
                                      lda, stA, vl, vu, il, iu, abstol, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (T* const*)nullptr, ldz, stZ,
                                      (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n, (T*)nullptr, lda,
                                      stA, vl, vu, il, iu, abstol, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (T*)nullptr, ldz, stZ,
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
            CHECK_ALLOC_QUERY(rocsolver_syevx_heevx(
                STRIDED, handle, evect, erange, uplo, n, (T* const*)nullptr, lda, stA, vl, vu, il,
                iu, abstol, (rocblas_int*)nullptr, (S*)nullptr, stW, (T* const*)nullptr, ldz, stZ,
                (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_syevx_heevx(
                STRIDED, handle, evect, erange, uplo, n, (T*)nullptr, lda, stA, vl, vu, il, iu,
                abstol, (rocblas_int*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldz, stZ,
                (rocblas_int*)nullptr, stF, (rocblas_int*)nullptr, bc));

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
    host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hNevRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
    host_strided_batch_vector<S> hWres(size_WRes, 1, stW, bc);
    host_strided_batch_vector<rocblas_int> hIfail(size_ifail, 1, stF, bc);
    host_strided_batch_vector<rocblas_int> hIfailRes(size_ifailRes, 1, stF, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, bc);
    device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dIfail(size_ifail, 1, stF, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    CHECK_HIP_ERROR(dNev.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_ifail)
        CHECK_HIP_ERROR(dIfail.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hZ(size_Z, 1, bc);
        host_batch_vector<T> hZRes(size_ZRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dZ(size_Z, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_Z)
            CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n,
                                                        dA.data(), lda, stA, vl, vu, il, iu, abstol,
                                                        dNev.data(), dW.data(), stW, dZ.data(), ldz,
                                                        stZ, dIfail.data(), stF, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevx_heevx_getError<STRIDED, T>(handle, evect, erange, uplo, n, dA, lda, stA, vl, vu,
                                             il, iu, abstol, dNev, dW, stW, dZ, ldz, stZ, dIfail,
                                             stF, dinfo, bc, hA, hNev, hNevRes, hW, hWres, hZ,
                                             hZRes, hIfail, hIfailRes, hinfo, hinfoRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevx_heevx_getPerfData<STRIDED, T>(
                handle, evect, erange, uplo, n, dA, lda, stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                dZ, ldz, stZ, dIfail, stF, dinfo, bc, hA, hNev, hW, hZ, hIfail, hinfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hZ(size_Z, 1, stZ, bc);
        host_strided_batch_vector<T> hZRes(size_ZRes, 1, stZ, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dZ(size_Z, 1, stZ, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_Z)
            CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevx_heevx(STRIDED, handle, evect, erange, uplo, n,
                                                        dA.data(), lda, stA, vl, vu, il, iu, abstol,
                                                        dNev.data(), dW.data(), stW, dZ.data(), ldz,
                                                        stZ, dIfail.data(), stF, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevx_heevx_getError<STRIDED, T>(handle, evect, erange, uplo, n, dA, lda, stA, vl, vu,
                                             il, iu, abstol, dNev, dW, stW, dZ, ldz, stZ, dIfail,
                                             stF, dinfo, bc, hA, hNev, hNevRes, hW, hWres, hZ,
                                             hZRes, hIfail, hIfailRes, hinfo, hinfoRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevx_heevx_getPerfData<STRIDED, T>(
                handle, evect, erange, uplo, n, dA, lda, stA, vl, vu, il, iu, abstol, dNev, dW, stW,
                dZ, ldz, stZ, dIfail, stF, dinfo, bc, hA, hNev, hW, hZ, hIfail, hinfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.perf);
        }
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
                rocsolver_bench_output("evect", "erange", "uplo", "n", "lda", "vl", "vu", "il",
                                       "iu", "abstol", "strideW", "strideF", "ldz", "batch_c");
                rocsolver_bench_output(evectC, erangeC, uploC, n, lda, vl, vu, il, iu, abstol, stW,
                                       stF, ldz, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("evect", "erange", "uplo", "n", "lda", "strideA", "vl", "vu",
                                       "il", "iu", "abstol", "strideW", "ldz", "strideZ", "strideF",
                                       "batch_c");
                rocsolver_bench_output(evectC, erangeC, uploC, n, lda, stA, vl, vu, il, iu, abstol,
                                       stW, ldz, stZ, stF, bc);
            }
            else
            {
                rocsolver_bench_output("evect", "erange", "uplo", "n", "lda", "vl", "vu", "il",
                                       "iu", "abstol", "ldz");
                rocsolver_bench_output(evectC, erangeC, uploC, n, lda, vl, vu, il, iu, abstol, ldz);
            }
            rocsolver_bench_header("Results:");
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
