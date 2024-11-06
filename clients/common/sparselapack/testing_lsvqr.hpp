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

template <bool CPU, bool GPU, typename T>
void lsvqr_initData(const rocblas_handle handle, const rocblas_int n, const rocblas_int nnz)
{
    if(CPU)
    {
        std::vector<T> denseA(n * n);

        const rocblas_int max_index = n * n;
        std::uniform_int_distribution<int> sample_index(0, max_index);

        // scale A to avoid singularities

        for(rocblas_int i = 0; i < m; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j)
                    denseA[i + j * n] += 400;
                else
                    denseA[i + j * n] -= 4;
            }
        }

        const rocblas_int n_zeroes = (n * n) - nnz;
        rocblas_int target_i = sample_index(rocblas_rng);
        rocblas_int target_j = sample_index(rocblas_rng);
        for(rocblas_int i = 0; i < n_zeroes; n++)
        {
            while((denseA[target_i + target_j * n] == 0) && (target_i != target_j))
            {
                target_i = sample_index(rocblas_rng);
                target_j = sample_index(rocblas_rng);
            }
            denseA[target_i + target_j * n] = 0;
        }

        // gather into sparse matrices
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <typename T>
void testing_lsvqr(Arguments& argus)
{
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnz = argus.get<rocblas_int>("nnz");
}

#define EXTERN_TESTING_LSVQR(...) extern template void testing_lsvqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LSVQR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
