/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larft.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> larft_tuple;

// each order_size_range vector is {N,ldv,s}
// if s = 0, then storev = 'C'
// if s = 1, then storev = 'R'

// each reflector_size_range is {K,ldt,d}
// if d = 0, then direct = 'F'
// if d = 1, then direct = 'B'

// case when n == 0 and k == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> order_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {10, 5, 0},
    {10, 3, 1},
    // normal (valid) samples
    {15, 15, 0},
    {20, 20, 1},
    {35, 50, 0}};

const vector<vector<int>> reflector_size_range = {
    // invalid
    {0, 1, 0},
    {5, 1, 0},
    // normal (valid) samples
    {5, 5, 0},
    {10, 20, 1},
    {15, 15, 0}};

// for daily_lapack tests
const vector<vector<int>> large_order_size_range
    = {{192, 192, 0}, {640, 75, 1}, {1024, 1200, 0}, {2048, 100, 1}};

const vector<vector<int>> large_reflector_size_range
    = {{15, 15, 0}, {25, 40, 1}, {45, 45, 0}, {60, 70, 1}, {75, 75, 0}};

Arguments larft_setup_arguments(larft_tuple tup)
{
    vector<int> order_size = std::get<0>(tup);
    vector<int> reflector_size = std::get<1>(tup);

    Arguments arg;

    arg.N = order_size[0];
    arg.ldv = order_size[1];
    arg.K = reflector_size[0];
    arg.ldt = reflector_size[1];

    arg.direct_option = reflector_size[2] == 1 ? 'B' : 'F';
    arg.storev = order_size[2] == 1 ? 'R' : 'C';

    arg.timing = 0;

    return arg;
}

class LARFT : public ::TestWithParam<larft_tuple>
{
protected:
    LARFT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LARFT, __float)
{
    Arguments arg = larft_setup_arguments(GetParam());

    if(arg.N == 0 && arg.K == 0)
        testing_larft_bad_arg<float>();

    testing_larft<float>(arg);
}

TEST_P(LARFT, __double)
{
    Arguments arg = larft_setup_arguments(GetParam());

    if(arg.N == 0 && arg.K == 0)
        testing_larft_bad_arg<double>();

    testing_larft<double>(arg);
}

TEST_P(LARFT, __float_complex)
{
    Arguments arg = larft_setup_arguments(GetParam());

    if(arg.N == 0 && arg.K == 0)
        testing_larft_bad_arg<rocblas_float_complex>();

    testing_larft<rocblas_float_complex>(arg);
}

TEST_P(LARFT, __double_complex)
{
    Arguments arg = larft_setup_arguments(GetParam());

    if(arg.N == 0 && arg.K == 0)
        testing_larft_bad_arg<rocblas_double_complex>();

    testing_larft<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFT,
                         Combine(ValuesIn(large_order_size_range),
                                 ValuesIn(large_reflector_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARFT,
                         Combine(ValuesIn(order_size_range), ValuesIn(reflector_size_range)));
