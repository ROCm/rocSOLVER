/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_lacgv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> lacgv_tuple;

// each range is a {n,inc}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {1, 0},
    // normal (valid) samples
    {10, 1},
    {10, -1},
    {20, 2},
    {30, 3},
    {30, -3}};

// for daily_lapack tests
const vector<vector<int>> large_range
    = {{192, 10}, {192, -10}, {250, 20}, {500, 30}, {1500, 40}, {1500, -40}};

Arguments lacgv_setup_arguments(lacgv_tuple tup)
{
    Arguments arg;

    arg.N = tup[0];
    arg.incx = tup[1];

    return arg;
}

class LACGV : public ::TestWithParam<lacgv_tuple>
{
protected:
    LACGV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LACGV, __float_complex)
{
    Arguments arg = lacgv_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_lacgv_bad_arg<rocblas_float_complex>();

    testing_lacgv<rocblas_float_complex>(arg);
}

TEST_P(LACGV, __double_complex)
{
    Arguments arg = lacgv_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_lacgv_bad_arg<rocblas_double_complex>();

    testing_lacgv<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LACGV, ValuesIn(large_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LACGV, ValuesIn(range));
