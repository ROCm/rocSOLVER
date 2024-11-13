/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void lsvqr_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int nnz,
                    Ud& dptrA,
                    Ud& dindA,
                    Td& dvalA,
                    Td& dB,
                    Td& dX,
                    Uh& hptrA,
                    Uh& hindA,
                    Th& hvalA,
                    Th& hB,
                    Th& hX,
                    const fs::path testcase)
{
    if(CPU)
    {
        fs::path file;

        // read-in A
        file = testcase / "ptrA";
        read_matrix(file.string(), 1, n + 1, hptrA.data(), 1);
        file = testcase / "indA";
        read_matrix(file.string(), 1, nnz, hindA.data(), 1);
        file = testcase / "valA";
        read_matrix(file.string(), 1, nnz, hvalA.data(), 1);

        // read-in B
        file = testcase / "B";
        read_matrix(file.string(), 1, n, hB.data(), 1);

        // read-in X
        file = testcase / "X";
        read_matrix(file.string(), 1, n, hX.data(), 1);
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrA.transfer_from(hptrA));
        CHECK_HIP_ERROR(dindA.transfer_from(hindA));
        CHECK_HIP_ERROR(dvalA.transfer_from(hvalA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dX.transfer_from(hX));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void lsvqr_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int nnz,
                    Ud& dptrA,
                    Ud& dindA,
                    Td& dvalA,
                    Td& dB,
                    Td& dX,
                    Uh& hptrA,
                    Uh& hindA,
                    Th& hvalA,
                    Th& hB,
                    Th& hX,
                    Th& hXres,
                    rocsolver_spinfo spinfo,
                    double* max_error,
                    const fs::path testcase)
{
    lsvqr_initData<true, true, T>(handle, n, nnz, dptrA, dindA, dvalA, dB, dX, hptrA, hindA, hvalA,
                                  hB, hX, testcase);

    CHECK_ROCBLAS_ERROR(
        rocsolver_csrlsvqr(handle, n, nnz, dvalA, dptrA, dindA, dB, 0, 0, dX, nullptr, spinfo));

    CHECK_HIP_ERROR(hXres.transfer_from(dX));

    *max_error = norm_error('I', n, 1, n, hX[0], hXres[0]);
}

template <typename T>
void testing_lsvqr(Arguments& argus)
{
    rocblas_local_handle handle;
    rocsolver_local_spinfo spinfo(handle);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnz = argus.get<rocblas_int>("nnz");
    T tol = 0;
    rocblas_int reorder = 0;
    rocblas_int singularity = 0;

    fs::path testcase;
    if(n > 0)
    {
        std::string matname;
        matname = fmt::format("sqmat_{}_{}", n, nnz);
        testcase = get_sparse_data_dir() / fs::path(matname);
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ptrA = n + 1;
    size_t size_indA = nnz;
    size_t size_valA = nnz;
    size_t size_B = n;
    size_t size_X = n;

    host_strided_batch_vector<rocblas_int> hptrA(size_ptrA, 1, size_ptrA, 1);
    host_strided_batch_vector<rocblas_int> hindA(size_indA, 1, size_indA, 1);
    host_strided_batch_vector<T> hvalA(size_valA, 1, size_valA, 1);
    host_strided_batch_vector<T> hB(size_B, 1, size_B, 1);
    host_strided_batch_vector<T> hX(size_X, 1, size_X, 1);
    host_strided_batch_vector<T> hXres(size_X, 1, size_X, 1);

    device_strided_batch_vector<rocblas_int> dptrA(size_ptrA, 1, size_ptrA, 1);
    device_strided_batch_vector<rocblas_int> dindA(size_indA, 1, size_indA, 1);
    device_strided_batch_vector<T> dvalA(size_valA, 1, size_valA, 1);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, 1);
    device_strided_batch_vector<T> dX(size_X, 1, size_X, 1);
    CHECK_HIP_ERROR(dptrA.memcheck());
    if(size_indA)
        CHECK_HIP_ERROR(dindA.memcheck());
    if(size_valA)
        CHECK_HIP_ERROR(dvalA.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dB.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dB.memcheck());

    lsvqr_getError<T>(handle, n, nnz, dptrA, dindA, dvalA, dB, dX, hptrA, hindA, hvalA, hB, hX,
                      hXres, spinfo, &max_error, testcase);

    // validate results for rocsolver-test
    // using 20 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);
}

#define EXTERN_TESTING_LSVQR(...) extern template void testing_lsvqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LSVQR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
