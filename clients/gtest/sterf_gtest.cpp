/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sterf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> sterf_tuple;

// each size_range vector is a {N}

// case when N == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0},
    // invalid
    {-1},
    // normal (valid) samples
    {12},
    {20},
    {35}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{192}, {256}, {300}};

Arguments sterf_setup_arguments(sterf_tuple tup)
{
    Arguments arg;

    arg.set<rocblas_int>("n", tup[0]);

    arg.timing = 0;

    return arg;
}

class STERF : public ::TestWithParam<sterf_tuple>
{
protected:
    STERF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = sterf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_sterf_bad_arg<T>();

        testing_sterf<T>(arg);
    }
};

// non-batch tests

TEST_P(STERF, __float)
{
    run_tests<float>();
}

TEST_P(STERF, __double)
{
    run_tests<double>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, STERF, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, STERF, ValuesIn(matrix_size_range));
