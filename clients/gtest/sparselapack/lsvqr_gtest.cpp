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

#include "common/sparselapack/testing_lsvqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

using lsvqr_tuple = tuple<int, int>;

// for checkin_lapack tests
const vector<int> n_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    20,
    50,
};

const vector<int> nnz_range = {
    // invalid
    -1,
    // normal (valid) samples
    60,
    100,
    140,
};

Arguments lsvqr_setup_arguments(lsvqr_tuple tup)
{
    int n = std::get<0>(tup);
    int nnz = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("nnz", nnz);

    arg.timing = 0;

    return arg;
}

class LSVQR : public ::TestWithParam<lsvqr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lsvqr_setup_arguments(GetParam());

        // TODO!: bad arg check

        testing_lsvqr<T>(arg);
    }
};

TEST_P(LSVQR, __float)
{
    run_tests<float>();
}

TEST_P(LSVQR, __double)
{
    run_tests<double>();
}

TEST_P(LSVQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LSVQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LSVQR, Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
