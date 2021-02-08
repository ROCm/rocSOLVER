/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_laswp.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> laswp_tuple;

// each range1 vector is a {n,lda}

// each range2 vector is a {k1,k2,inc}

// case when n = 0, k1 = 1 and k2 = 3  will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> range1 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 0},
    // normal (valid) samples
    {10, 100},
    {20, 100},
    {30, 100}};
const vector<vector<int>> range2 = {
    // invalid
    {0, 1, 1},
    {1, 0, 1},
    {1, 2, 0},
    {2, 1, 1},
    // normal (valid) samples
    {1, 3, 1},
    {3, 5, 2},
    {5, 10, -1},
    {3, 12, -2}};

// for daily_lapack tests
const vector<vector<int>> large_range1 = {{192, 100}, {250, 100}, {500, 100}, {1500, 100}};
const vector<vector<int>> large_range2 = {{1, 50, 1}, {5, 60, 2}, {3, 70, -1}, {20, 100, -2}};

Arguments laswp_setup_arguments(laswp_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> pivots = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];
    arg.k1 = pivots[0];
    arg.k2 = pivots[1];
    arg.incx = pivots[2];
    arg.timing = 0;

    return arg;
}

class LASWP : public ::TestWithParam<laswp_tuple>
{
protected:
    LASWP() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LASWP, __float)
{
    Arguments arg = laswp_setup_arguments(GetParam());

    if(arg.N == 0 && arg.k1 == 1 && arg.k2 == 3)
        testing_laswp_bad_arg<float>();

    testing_laswp<float>(arg);
}

TEST_P(LASWP, __double)
{
    Arguments arg = laswp_setup_arguments(GetParam());

    if(arg.N == 0 && arg.k1 == 1 && arg.k2 == 3)
        testing_laswp_bad_arg<double>();

    testing_laswp<double>(arg);
}

TEST_P(LASWP, __float_complex)
{
    Arguments arg = laswp_setup_arguments(GetParam());

    if(arg.N == 0 && arg.k1 == 1 && arg.k2 == 3)
        testing_laswp_bad_arg<rocblas_float_complex>();

    testing_laswp<rocblas_float_complex>(arg);
}

TEST_P(LASWP, __double_complex)
{
    Arguments arg = laswp_setup_arguments(GetParam());

    if(arg.N == 0 && arg.k1 == 1 && arg.k2 == 3)
        testing_laswp_bad_arg<rocblas_double_complex>();

    testing_laswp<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LASWP, Combine(ValuesIn(large_range1), ValuesIn(large_range2)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LASWP, Combine(ValuesIn(range1), ValuesIn(range2)));
