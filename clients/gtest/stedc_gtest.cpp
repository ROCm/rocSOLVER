/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_stedc.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> stedc_tuple;

// each size_range vector is a {N, ldc}

// each op_range vector is a {e}
// if e = 0, then evect = 'N'
// if e = 1, then evect = 'I'
// if e = 2, then evect = 'V'

// case when N == 0 and evect == N will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> op_range = {{0}, {1}, {2}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    // invalid for case evect != N
    {2, 1},
    // normal (valid) samples
    {12, 12},
    {20, 30},
    {35, 40}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments stedc_setup_arguments(stedc_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldc", size[1]);

    arg.set<char>("evect", (op[0] == 0 ? 'N' : (op[0] == 1 ? 'I' : 'V')));

    arg.timing = 0;

    return arg;
}

class STEDC : public ::TestWithParam<stedc_tuple>
{
protected:
    STEDC() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = stedc_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N')
            testing_stedc_bad_arg<T>();

        testing_stedc<T>(arg);
    }
};

// non-batch tests

TEST_P(STEDC, __float)
{
    run_tests<float>();
}

TEST_P(STEDC, __double)
{
    run_tests<double>();
}

TEST_P(STEDC, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEDC, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEDC,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEDC,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));
