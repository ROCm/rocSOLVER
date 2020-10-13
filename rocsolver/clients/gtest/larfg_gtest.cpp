/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larfg.hpp"

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

    arg.N = n_size;
    arg.incx = inc;

    arg.timing = 0;

    return arg;
}

class LARFG : public ::TestWithParam<larfg_tuple>
{
protected:
    LARFG() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LARFG, __float)
{
    Arguments arg = larfg_setup_arguments(GetParam());

    if(arg.N == 0 && arg.incx == 0)
        testing_larfg_bad_arg<float>();

    testing_larfg<float>(arg);
}

TEST_P(LARFG, __double)
{
    Arguments arg = larfg_setup_arguments(GetParam());

    if(arg.N == 0 && arg.incx == 0)
        testing_larfg_bad_arg<double>();

    testing_larfg<double>(arg);
}

TEST_P(LARFG, __float_complex)
{
    Arguments arg = larfg_setup_arguments(GetParam());

    if(arg.N == 0 && arg.incx == 0)
        testing_larfg_bad_arg<rocblas_float_complex>();

    testing_larfg<rocblas_float_complex>(arg);
}

TEST_P(LARFG, __double_complex)
{
    Arguments arg = larfg_setup_arguments(GetParam());

    if(arg.N == 0 && arg.incx == 0)
        testing_larfg_bad_arg<rocblas_double_complex>();

    testing_larfg<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFG,
                         Combine(ValuesIn(large_n_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LARFG, Combine(ValuesIn(n_size_range), ValuesIn(incx_range)));
