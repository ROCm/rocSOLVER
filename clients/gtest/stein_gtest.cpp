/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_stein.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> stein_tuple;

// each size_range vector is a {N, ldz}

// case when N == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {2, 1},
    // normal (valid) samples
    {12, 12},
    {20, 30},
    {35, 40}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments stein_setup_arguments(stein_tuple tup)
{
    vector<int> size = tup;

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldz", size[1]);

    arg.timing = 0;

    return arg;
}

class STEIN : public ::TestWithParam<stein_tuple>
{
protected:
    STEIN() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = stein_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_stein_bad_arg<T>();

        testing_stein<T>(arg);
    }
};

// non-batch tests

TEST_P(STEIN, __float)
{
    run_tests<float>();
}

TEST_P(STEIN, __double)
{
    run_tests<double>();
}

TEST_P(STEIN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEIN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, STEIN, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, STEIN, ValuesIn(matrix_size_range));
