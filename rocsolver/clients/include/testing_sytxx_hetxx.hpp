/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.h"
#include "clientcommon.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, bool SYTRD, typename S, typename T, typename U>
void sytxx_hetxx_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              S dD,
                              const rocblas_stride stD,
                              S dE,
                              const rocblas_stride stE,
                              U dTau,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, nullptr, uplo, n, dA, lda, stA, dD,
                                                stD, dE, stE, dTau, stP, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, rocblas_fill_full, n, dA, lda, stA, dD,
                                                stD, dE, stE, dTau, stP, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTau, stP, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, (T) nullptr, lda, stA,
                                                dD, stD, dE, stE, dTau, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA, lda, stA,
                                                (S) nullptr, stD, dE, stE, dTau, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA, lda, stA, dD, stD,
                                                (S) nullptr, stE, dTau, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA, lda, stA, dD, stD,
                                                dE, stE, (U) nullptr, stP, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, 0, (T) nullptr, lda, stA,
                                                (S) nullptr, stD, (S) nullptr, stE, (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTau, stP, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool SYTRD, typename T>
void testing_sytxx_hetxx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTau(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTau.memcheck());

        // check bad arguments
        sytxx_hetxx_checkBadArgs<STRIDED, SYTRD>(handle, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTau.data(), stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTau(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTau.memcheck());

        // check bad arguments
        sytxx_hetxx_checkBadArgs<STRIDED, SYTRD>(handle, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTau.data(), stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<!is_complex<T>, int> = 0>
void sytxx_hetxx_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_int bc,
                          Th& hA)
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
                    if(i == j || i == j+1 || i == j-1)
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

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<is_complex<T>, int> = 0>
void sytxx_hetxx_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_int bc,
                          Th& hA)
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
                        hA[b][i + j * lda] = hA[b][i + j * lda].real()+400;
                    else if(i == j+1 || i == j-1)
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

template <bool STRIDED, bool SYTRD, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void sytxx_hetxx_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Ud& dTau,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Sh& hD,
                          Sh& hE,
                          Uh& hTau,
                          double* max_err)
{
    using S = decltype(std::real(T{}));
    constexpr bool COMPLEX = is_complex<T>;

    std::vector<T> hW(32*n);

    // input data initialization
    sytxx_hetxx_initData<true, true, T>(handle, n, dA, lda, bc, hA);

/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) {
    for (int j=0;j<n;++j)
        rocblas_cout<<hA[0][i+j*lda]<<" ";
    rocblas_cout<<"\n";
}*/
/*for(rocblas_int b = 0; b < bc; ++b)
{
    memcpy(hARes[b], hA[b], lda * n * sizeof(T));
    SYTRD
    ? cblas_sytrd_hetrd<S,T>(uplo, n, hARes[b], lda, hD[b], hE[b], hTau[b], hW.data(), 32*n)
    : cblas_sytd2_hetd2<S,T>(uplo, n, hARes[b], lda, hD[b], hE[b], hTau[b]);
}*/
/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) {
    for (int j=0;j<n;++j)
        rocblas_cout<<hARes[0][i+j*lda]<<" ";
    rocblas_cout<<"\n";
}*/
    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA.data(), lda, stA,
                                              dD.data(), stD, dE.data(), stE, 
                                              dTau.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hTau.transfer_from(dTau));

    // Reconstruct matrix A from the factorization for implicit testing
    // A = H(n-1)...H(2)H(1)*T*H(1)'H(2)'...H(n-1)' if upper
    // A = H(1)H(2)...H(n-1)*T*H(n-1)'...H(2)'H(1)' if lower
    std::vector<T> v(n);
    for(rocblas_int b = 0; b < bc; ++b)
    {
        T* a = hARes[b];
        T* t = hTau[b];

        if(uplo == rocblas_fill_lower)
        {
            for(rocblas_int i=0; i<n-2; ++i)
                a[i + (n-1)*lda] = 0;
            a[n-2 + (n-1)*lda] = a[n-1 + (n-2)*lda];    

            // for each column
            for(rocblas_int j=n-2; j>=0; --j)
            {
                // prepare T and v
                for(rocblas_int i=0; i<j-1; ++i)
                    a[i + j*lda] = 0;
                if(j > 0)
                    a[j-1 + j*lda] = a[j + (j-1)*lda];
                for(rocblas_int i=j+2; i<n; ++i)
                {
                    v[i-j-1] = a[i + j*lda];
                    a[i + j*lda] = 0;
                }
                v[0] = 1;

/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) 
    rocblas_cout<<v[i]<<" ";
rocblas_cout<<"\n";
rocblas_cout<<t[j]<<"\n";
*/
                // apply householder reflector
                cblas_larf(rocblas_side_left, n - 1 - j, n - j, v.data(), 1, t + j,
                               a + j + 1 + j*lda, lda, hW.data());   
                if(COMPLEX)
                    cblas_lacgv(1, t + j, 1);             
                cblas_larf(rocblas_side_right, n - j, n - 1 - j, v.data(), 1, t + j,
                               a + j + (j + 1)*lda, lda, hW.data());

/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) {
    for (int j=0;j<n;++j)
        rocblas_cout<<hARes[0][i+j*lda]<<" ";
    rocblas_cout<<"\n";
}
*/


            }        
        }

        else
        {
            a[1] = a[lda];
            for(rocblas_int i=2; i<n; ++i)
                a[i] = 0;

            // for each column
            for(rocblas_int j=1; j<=n-1; ++j)
            {
                // prepare T and v
                for(rocblas_int i=0; i<j-1; ++i)
                {
                    v[i] = a[i + j*lda];
                    a[i + j*lda] = 0;
                }
                v[j-1] = 1;
                if(j < n-1)
                    a[j+1 +j*lda] = a[j + (j+1)*lda];
                for(rocblas_int i=j+2; i<n; ++i)
                    a[i + j*lda] = 0;    

                // apply householder reflector
                cblas_larf(rocblas_side_left, j, j + 1, v.data(), 1, t + j - 1,
                               a, lda, hW.data());                
                if(COMPLEX)
                    cblas_lacgv(1, t + j - 1, 1);
                cblas_larf(rocblas_side_right, j + 1, j, v.data(), 1, t + j - 1,
                               a, lda, hW.data());                
            } 
        }
    } 


/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) {
    for (int j=0;j<n;++j)
        rocblas_cout<<hARes[0][i+j*lda]<<" ";
    rocblas_cout<<"\n";
}*/
/*rocblas_cout<<"\n";
for(int i=0;i<n;++i) 
    rocblas_cout<<hD[0][i]<<" ";
rocblas_cout<<"\n";
rocblas_cout<<"\n";
for(int i=0;i<n-1;++i) 
    rocblas_cout<<hE[0][i]<<" ";
rocblas_cout<<"\n";
rocblas_cout<<"\n";
for(int i=0;i<n-1;++i) 
    rocblas_cout<<hTau[0][i]<<" ";
rocblas_cout<<"\n";
*/




    // CPU lapack

/*
rocblas_cout<<"\n";
for(int i=0;i<n;++i) {
    for (int j=0;j<n;++j)
        rocblas_cout<<hA[0][i+j*lda]<<" ";
    rocblas_cout<<"\n";
}
rocblas_cout<<"\n";
for(int i=0;i<n;++i) 
    rocblas_cout<<hD[0][i]<<" ";
rocblas_cout<<"\n";
rocblas_cout<<"\n";
for(int i=0;i<n-1;++i) 
    rocblas_cout<<hE[0][i]<<" ";
rocblas_cout<<"\n";
rocblas_cout<<"\n";
for(int i=0;i<n-1;++i) 
    rocblas_cout<<hTau[0][i]<<" ";
rocblas_cout<<"\n";
*/

    
    // error is ||hA - hARes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        *max_err = (uplo == rocblas_fill_lower) 
        ? norm_error_lowerTr('F', n, n, lda, hA[b], hARes[b])
        : norm_error_upperTr('F', n, n, lda, hA[b], hARes[b]);
    }
}

template <bool STRIDED, bool SYTRD, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void sytxx_hetxx_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Sd& dD,
                             const rocblas_stride stD,
                             Sd& dE,
                             const rocblas_stride stE,
                             Ud& dTau,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Sh& hD,
                             Sh& hE,
                             Uh& hTau,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const bool perf)
{
    using S = decltype(std::real(T{}));
    
    std::vector<T> hW(32*n);

    if(!perf)
    {
        sytxx_hetxx_initData<true, false, T>(handle, n, dA, lda, bc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            SYTRD ? cblas_sytrd_hetrd<S,T>(uplo, n, hA[b], lda, hD[b], hE[b], hTau[b], hW.data(), 32*n)
                  : cblas_sytd2_hetd2<S,T>(uplo, n, hA[b], lda, hD[b], hE[b], hTau[b]);
        }
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    sytxx_hetxx_initData<true, false, T>(handle, n, dA, lda, bc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytxx_hetxx_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, 
                                                  dTau.data(), stP, bc));
    }

    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sytxx_hetxx_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        start = get_time_us();
        rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                              dE.data(), stE, dTau.data(), stP, bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool SYTRD, typename T>
void testing_sytxx_hetxx(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stD = argus.bsp;
    rocblas_stride stE = argus.bsp;
    rocblas_stride stP = argus.bsp;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    char uploC = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(uploC);

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n,
                                                        (T* const*)nullptr, lda, stA, (S*)nullptr,
                                                        stD, (S*)nullptr, stE, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n,
                                                        (T*)nullptr, lda, stA, (S*)nullptr,
                                                        stD, (S*)nullptr, stE, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = n;
    size_t size_E = n;
    size_t size_tau = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n,
                                                        (T* const*)nullptr, lda, stA, (S*)nullptr,
                                                        stD, (S*)nullptr, stE, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (S*)nullptr, stD, (S*)nullptr,
                                                        stE, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
    host_strided_batch_vector<S> hE(size_E, 1, stE, bc);
    host_strided_batch_vector<T> hTau(size_tau, 1, stP, bc);
    // device
    device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
    device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
    device_strided_batch_vector<T> dTau(size_tau, 1, stP, bc);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTau.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytxx_hetxx_getError<STRIDED, SYTRD, T>(handle, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                                       dTau, stP, bc, hA, hARes, hD,
                                                       hE, hTau, &max_error);

        // collect performance data
        if(argus.timing)
            sytxx_hetxx_getPerfData<STRIDED, SYTRD, T>(
                handle, uplo, n, dA, lda, stA, dD, stD, dE, stE, dTau, stP, bc, hA, hD,
                hE, hTau, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytxx_hetxx(STRIDED, SYTRD, handle, uplo, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTau.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytxx_hetxx_getError<STRIDED, SYTRD, T>(handle, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                                       dTau, stP, bc, hA, hARes, hD,
                                                       hE, hTau, &max_error);

        // collect performance data
        if(argus.timing)
            sytxx_hetxx_getPerfData<STRIDED, SYTRD, T>(
                handle, uplo, n, dA, lda, stA, dD, stD, dE, stE, dTau, stP, bc, hA, hD,
                hE, hTau, &gpu_time_used, &cpu_time_used, hot_calls, argus.perf);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        rocsolver_test_check<T>(max_error, n);

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
                rocsolver_bench_output("uplo", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
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
