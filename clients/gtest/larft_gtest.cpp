/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

    arg.set<rocblas_int>("n", order_size[0]);
    arg.set<rocblas_int>("ldv", order_size[1]);
    arg.set<char>("storev", order_size[2] == 1 ? 'R' : 'C');

    arg.set<rocblas_int>("k", reflector_size[0]);
    arg.set<rocblas_int>("ldt", reflector_size[1]);
    arg.set<char>("direct", reflector_size[2] == 1 ? 'B' : 'F');

    arg.timing = 0;

    return arg;
}

class LARFT : public ::TestWithParam<larft_tuple>
{
protected:
    LARFT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = larft_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("k") == 0)
            testing_larft_bad_arg<T>();

        testing_larft<T>(arg);
    }
};

// non-batch tests

TEST_P(LARFT, __float)
{
    run_tests<float>();
}

TEST_P(LARFT, __double)
{
    run_tests<double>();
}

TEST_P(LARFT, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARFT, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFT,
                         Combine(ValuesIn(large_order_size_range),
                                 ValuesIn(large_reflector_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARFT,
                         Combine(ValuesIn(order_size_range), ValuesIn(reflector_size_range)));
