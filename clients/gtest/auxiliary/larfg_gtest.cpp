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

#include "common/auxiliary/testing_larfg.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int> larfg_tuple;

// case when n = 0 and incx = 0 also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<int> incx_range = {
    // invalid
    -1,
    0,
    // normal (valid) samples
    1,
    5,
    8,
    10,
};

// for checkin_lapack tests
const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    1,
    12,
    20,
    35,
};

// for daily_lapack tests
const vector<int> large_n_size_range = {
    192,
    640,
    1024,
    2547,
};

Arguments larfg_setup_arguments(larfg_tuple tup)
{
    int n_size = std::get<0>(tup);
    int inc = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("incx", inc);

    arg.timing = 0;

    return arg;
}

class LARFG : public ::TestWithParam<larfg_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = larfg_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("incx") == 0)
            testing_larfg_bad_arg<T>();

        testing_larfg<T>(arg);
    }
};

// non-batch tests

TEST_P(LARFG, __float)
{
    run_tests<float>();
}

TEST_P(LARFG, __double)
{
    run_tests<double>();
}

TEST_P(LARFG, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARFG, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFG,
                         Combine(ValuesIn(large_n_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LARFG, Combine(ValuesIn(n_size_range), ValuesIn(incx_range)));
