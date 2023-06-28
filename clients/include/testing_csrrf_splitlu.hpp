/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T>
void csrrf_splitlu_checkBadArgs(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nnzT,
                                rocblas_int* ptrT,
                                rocblas_int* indT,
                                T valT,
                                rocblas_int* ptrL,
                                rocblas_int* indL,
                                T valL,
                                rocblas_int* ptrU,
                                rocblas_int* indU,
                                T valU)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(nullptr, n, nnzT, ptrT, indT, valT, ptrL, indL,
                                                  valL, ptrU, indU, valU),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, (rocblas_int*)nullptr, indT,
                                                  valT, ptrL, indL, valL, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, (rocblas_int*)nullptr,
                                                  valT, ptrL, indL, valL, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, (T) nullptr, ptrL,
                                                  indL, valL, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT,
                                                  (rocblas_int*)nullptr, indL, valL, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL,
                                                  (rocblas_int*)nullptr, valL, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL,
                                                  (T) nullptr, ptrU, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL,
                                                  valL, (rocblas_int*)nullptr, indU, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL,
                                                  valL, ptrU, (rocblas_int*)nullptr, valU),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL,
                                                  valL, ptrU, indU, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, 0, ptrT, (rocblas_int*)nullptr,
                                                  (T) nullptr, ptrL, indL, valL, ptrU,
                                                  (rocblas_int*)nullptr, (T) nullptr),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, 0, 0, ptrT, (rocblas_int*)nullptr,
                                                  (T) nullptr, ptrL, (rocblas_int*)nullptr,
                                                  (T) nullptr, ptrU, (rocblas_int*)nullptr,
                                                  (T) nullptr),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    // N/A
}

template <typename T>
void testing_csrrf_splitlu_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int nnzT = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T> valT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrL(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indL(1, 1, 1, 1);
    device_strided_batch_vector<T> valL(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrU(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indU(1, 1, 1, 1);
    device_strided_batch_vector<T> valU(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(ptrL.memcheck());
    CHECK_HIP_ERROR(indL.memcheck());
    CHECK_HIP_ERROR(valL.memcheck());
    CHECK_HIP_ERROR(ptrU.memcheck());
    CHECK_HIP_ERROR(indU.memcheck());
    CHECK_HIP_ERROR(valU.memcheck());

    // check bad arguments
    csrrf_splitlu_checkBadArgs(handle, n, nnzT, ptrT.data(), indT.data(), valT.data(), ptrL.data(),
                               indL.data(), valL.data(), ptrU.data(), indU.data(), valU.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_splitlu_initData(rocblas_handle handle,
                            const rocblas_int n,
                            const rocblas_int nnzT,
                            Ud& dptrT,
                            Ud& dindT,
                            Td& dvalT,
                            Uh& hptrT,
                            Uh& hindT,
                            Th& hvalT,
                            Uh& hptrL,
                            Uh& hindL,
                            Th& hvalL,
                            Uh& hptrU,
                            Uh& hindU,
                            Th& hvalU,
                            const fs::path testcase,
                            bool test = true)
{
    bool mat_zero = (nnzT == 0);

    // if not matrix zero, read data from files
    if(!mat_zero)
    {
        if(CPU)
        {
            fs::path file;

            // read-in T
            file = testcase / "ptrT";
            read_matrix(file.string(), 1, n + 1, hptrT.data(), 1);
            file = testcase / "indT";
            read_matrix(file.string(), 1, nnzT, hindT.data(), 1);
            file = testcase / "valT";
            read_matrix(file.string(), 1, nnzT, hvalT.data(), 1);

            // get results (matrices L and U) if validation is required
            if(test)
            {
                rocblas_int nnzL, nnzU;

                // read-in L
                file = testcase / "ptrL";
                read_matrix(file.string(), 1, n + 1, hptrL.data(), 1);
                nnzL = hptrL[0][n];
                file = testcase / "indL";
                read_matrix(file.string(), 1, nnzL, hindL.data(), 1);
                file = testcase / "valL";
                read_matrix(file.string(), 1, nnzL, hvalL.data(), 1);

                // read-in U
                file = testcase / "ptrU";
                read_matrix(file.string(), 1, n + 1, hptrU.data(), 1);
                nnzU = hptrU[0][n];
                file = testcase / "indU";
                read_matrix(file.string(), 1, nnzU, hindU.data(), 1);
                file = testcase / "valU";
                read_matrix(file.string(), 1, nnzU, hvalU.data(), 1);
            }
        }

        if(GPU)
        {
            CHECK_HIP_ERROR(dptrT.transfer_from(hptrT));
            CHECK_HIP_ERROR(dindT.transfer_from(hindT));
            CHECK_HIP_ERROR(dvalT.transfer_from(hvalT));
        }
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_splitlu_getError(rocblas_handle handle,
                            const rocblas_int n,
                            const rocblas_int nnzT,
                            Ud& dptrT,
                            Ud& dindT,
                            Td& dvalT,
                            Ud& dptrL,
                            Ud& dindL,
                            Td& dvalL,
                            Ud& dptrU,
                            Ud& dindU,
                            Td& dvalU,
                            Uh& hptrT,
                            Uh& hindT,
                            Th& hvalT,
                            Uh& hptrL,
                            Uh& hindL,
                            Th& hvalL,
                            Uh& hptrU,
                            Uh& hindU,
                            Th& hvalU,
                            Uh& hptrLres,
                            Uh& hindLres,
                            Th& hvalLres,
                            Uh& hptrUres,
                            Uh& hindUres,
                            Th& hvalUres,
                            double* max_err,
                            const fs::path testcase)
{
    // input data initialization
    csrrf_splitlu_initData<true, true, T>(handle, n, nnzT, dptrT, dindT, dvalT, hptrT, hindT, hvalT,
                                          hptrL, hindL, hvalL, hptrU, hindU, hvalU, testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_splitlu(
        handle, n, nnzT, dptrT.data(), dindT.data(), dvalT.data(), dptrL.data(), dindL.data(),
        dvalL.data(), dptrU.data(), dindU.data(), dvalU.data()));

    CHECK_HIP_ERROR(hptrLres.transfer_from(dptrL));
    CHECK_HIP_ERROR(hindLres.transfer_from(dindL));
    CHECK_HIP_ERROR(hvalLres.transfer_from(dvalL));
    CHECK_HIP_ERROR(hptrUres.transfer_from(dptrU));
    CHECK_HIP_ERROR(hindUres.transfer_from(dindU));
    CHECK_HIP_ERROR(hvalUres.transfer_from(dvalU));

    double err = 0;
    bool mat_zero = (nnzT == 0);

    // if not mat zero, compare computed results with original result
    if(!mat_zero)
    {
        for(rocblas_int i = 0; i <= n; ++i)
        {
            err += (hptrL[0][i] - hptrLres[0][i]);
            err += (hptrU[0][i] - hptrUres[0][i]);
        }

        rocblas_int nnzL = hptrL[0][n];
        rocblas_int nnzU = hptrU[0][n];

        err += nnzT - (nnzL + nnzU - n);

        for(rocblas_int i = 0; i < nnzL; ++i)
        {
            err += (hindL[0][i] - hindLres[0][i]);
            err += (hvalL[0][i] - hvalLres[0][i]);
        }

        for(rocblas_int i = 0; i < nnzU; ++i)
        {
            err += (hindU[0][i] - hindUres[0][i]);
            err += (hvalU[0][i] - hvalUres[0][i]);
        }
    }
    // otherwise simply check that L = identity and ptrU = 0
    else
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            err += i - hptrLres[0][i];
            err += i - hindLres[0][i];
            err += 1 - hvalLres[0][i];
            err += hptrUres[0][i];
        }
        err += n - hptrLres[0][n];
        err += hptrUres[0][n];
    }

    *max_err = err;
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_splitlu_getPerfData(rocblas_handle handle,
                               const rocblas_int n,
                               const rocblas_int nnzT,
                               Ud& dptrT,
                               Ud& dindT,
                               Td& dvalT,
                               Ud& dptrL,
                               Ud& dindL,
                               Td& dvalL,
                               Ud& dptrU,
                               Ud& dindU,
                               Td& dvalU,
                               Uh& hptrT,
                               Uh& hindT,
                               Th& hvalT,
                               Uh& hptrL,
                               Uh& hindL,
                               Th& hvalL,
                               Uh& hptrU,
                               Uh& hindU,
                               Th& hvalU,
                               double* gpu_time_used,
                               double* cpu_time_used,
                               const rocblas_int hot_calls,
                               const int profile,
                               const bool profile_kernels,
                               const bool perf,
                               const fs::path testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrrf_splitlu_initData<true, false, T>(handle, n, nnzT, dptrT, dindT, dvalT, hptrT, hindT, hvalT,
                                           hptrL, hindL, hvalL, hptrU, hindU, hvalU, testcase, false);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrrf_splitlu_initData<false, true, T>(handle, n, nnzT, dptrT, dindT, dvalT, hptrT, hindT,
                                               hvalT, hptrL, hindL, hvalL, hptrU, hindU, hvalU,
                                               testcase, false);

        CHECK_ROCBLAS_ERROR(rocsolver_csrrf_splitlu(
            handle, n, nnzT, dptrT.data(), dindT.data(), dvalT.data(), dptrL.data(), dindL.data(),
            dvalL.data(), dptrU.data(), dindU.data(), dvalU.data()));
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
        csrrf_splitlu_initData<false, true, T>(handle, n, nnzT, dptrT, dindT, dvalT, hptrT, hindT,
                                               hvalT, hptrL, hindL, hvalL, hptrU, hindU, hvalU,
                                               testcase, false);

        start = get_time_us_sync(stream);
        rocsolver_csrrf_splitlu(handle, n, nnzT, dptrT.data(), dindT.data(), dvalT.data(),
                                dptrL.data(), dindL.data(), dvalL.data(), dptrU.data(),
                                dindU.data(), dvalU.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_csrrf_splitlu(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnzT = argus.get<rocblas_int>("nnzT");
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzT < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, (rocblas_int*)nullptr,
                                                      (rocblas_int*)nullptr, (T*)nullptr,
                                                      (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                      (T*)nullptr, (rocblas_int*)nullptr,
                                                      (rocblas_int*)nullptr, (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine existing test case
    rocblas_int nnzA = nnzT;
    bool mat_zero = (nnzA == 0);
    if(!mat_zero)
    {
        if(n > 0)
        {
            if(n <= 35)
                n = 20;
            else if(n <= 75)
                n = 50;
            else if(n <= 175)
                n = 100;
            else
                n = 250;
        }

        if(n <= 50) // small case
        {
            if(nnzA <= 80)
                nnzA = 60;
            else if(nnzA <= 120)
                nnzA = 100;
            else
                nnzA = 140;
        }
        else // large case
        {
            if(nnzA <= 400)
                nnzA = 300;
            else if(nnzA <= 600)
                nnzA = 500;
            else
                nnzA = 700;
        }
    }

    // read/set corresponding nnzL, nnzU and nnzT
    fs::path testcase;
    rocblas_int nnzU = 0;
    rocblas_int nnzL = n;
    if(!mat_zero && n > 0)
    {
        testcase = get_sparse_data_dir() / fs::path(fmt::format("mat_{}_{}", n, nnzA));
        fs::path file = testcase / "ptrL";
        read_last(file.string(), &nnzL);
        file = testcase / "ptrU";
        read_last(file.string(), &nnzU);
    }
    nnzT = nnzL + nnzU - n;

    // memory size query if necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_csrrf_splitlu(
            handle, n, nnzT, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
            (rocblas_int*)nullptr, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // determine sizes
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_ptrL = size_t(n) + 1;
    size_t size_indL = size_t(nnzL);
    size_t size_valL = size_t(nnzL);
    size_t size_ptrU = size_t(n) + 1;
    size_t size_indU = size_t(nnzU);
    size_t size_valU = size_t(nnzU);

    size_t size_ptrLres = 0;
    size_t size_indLres = 0;
    size_t size_valLres = 0;
    size_t size_ptrUres = 0;
    size_t size_indUres = 0;
    size_t size_valUres = 0;
    if(argus.unit_check || argus.norm_check)
    {
        size_ptrLres = size_ptrL;
        size_indLres = size_indL;
        size_valLres = size_valL;
        size_ptrUres = size_ptrU;
        size_indUres = size_indU;
        size_valUres = size_valU;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<rocblas_int> hptrL(size_ptrL, 1, size_ptrL, 1);
    host_strided_batch_vector<rocblas_int> hindL(size_indL, 1, size_indL, 1);
    host_strided_batch_vector<T> hvalL(size_valL, 1, size_valL, 1);
    host_strided_batch_vector<rocblas_int> hptrU(size_ptrU, 1, size_ptrU, 1);
    host_strided_batch_vector<rocblas_int> hindU(size_indU, 1, size_indU, 1);
    host_strided_batch_vector<T> hvalU(size_valU, 1, size_valU, 1);
    host_strided_batch_vector<rocblas_int> hptrUres(size_ptrUres, 1, size_ptrUres, 1);
    host_strided_batch_vector<rocblas_int> hindUres(size_indUres, 1, size_indUres, 1);
    host_strided_batch_vector<T> hvalUres(size_valUres, 1, size_valUres, 1);
    host_strided_batch_vector<rocblas_int> hptrLres(size_ptrLres, 1, size_ptrLres, 1);
    host_strided_batch_vector<rocblas_int> hindLres(size_indLres, 1, size_indLres, 1);
    host_strided_batch_vector<T> hvalLres(size_valLres, 1, size_valLres, 1);

    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    device_strided_batch_vector<rocblas_int> dptrL(size_ptrL, 1, size_ptrL, 1);
    device_strided_batch_vector<rocblas_int> dindL(size_indL, 1, size_indL, 1);
    device_strided_batch_vector<T> dvalL(size_valL, 1, size_valL, 1);
    device_strided_batch_vector<rocblas_int> dptrU(size_ptrU, 1, size_ptrU, 1);
    device_strided_batch_vector<rocblas_int> dindU(size_indU, 1, size_indU, 1);
    device_strided_batch_vector<T> dvalU(size_valU, 1, size_valU, 1);
    CHECK_HIP_ERROR(dptrT.memcheck());
    CHECK_HIP_ERROR(dptrL.memcheck());
    CHECK_HIP_ERROR(dptrU.memcheck());
    if(size_indT)
        CHECK_HIP_ERROR(dindT.memcheck());
    if(size_valT)
        CHECK_HIP_ERROR(dvalT.memcheck());
    if(size_indL)
        CHECK_HIP_ERROR(dindL.memcheck());
    if(size_valL)
        CHECK_HIP_ERROR(dvalL.memcheck());
    if(size_valU)
        CHECK_HIP_ERROR(dvalU.memcheck());
    if(size_indU)
        CHECK_HIP_ERROR(dindU.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_splitlu(handle, n, nnzT, dptrT.data(), dindT.data(),
                                                      dvalT.data(), dptrL.data(), dindL.data(),
                                                      dvalL.data(), dptrU.data(), dindU.data(),
                                                      dvalU.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_splitlu_getError<T>(handle, n, nnzT, dptrT, dindT, dvalT, dptrL, dindL, dvalL, dptrU,
                                  dindU, dvalU, hptrT, hindT, hvalT, hptrL, hindL, hvalL, hptrU,
                                  hindU, hvalU, hptrLres, hindLres, hvalLres, hptrUres, hindUres,
                                  hvalUres, &max_error, testcase);

    // collect performance data
    if(argus.timing)
        csrrf_splitlu_getPerfData<T>(handle, n, nnzT, dptrT, dindT, dvalT, dptrL, dindL, dvalL,
                                     dptrU, dindU, dvalU, hptrT, hindT, hvalT, hptrL, hindL, hvalL,
                                     hptrU, hindU, hvalU, &gpu_time_used, &cpu_time_used, hot_calls,
                                     argus.profile, argus.profile_kernels, argus.perf, testcase);

    // validate results for rocsolver-test
    // using machine precision for tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 1);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nnzT");
            rocsolver_bench_output(n, nnzT);

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

#define EXTERN_TESTING_CSRRF_SPLITLU(...) \
    extern template void testing_csrrf_splitlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CSRRF_SPLITLU, FOREACH_REAL_TYPE, APPLY_STAMP)
