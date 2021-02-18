/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_steqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> steqr_tuple;

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

Arguments steqr_setup_arguments(steqr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    Arguments arg;

    arg.N = size[0];
    arg.ldc = size[1];

    arg.evect = (op[0] == 0 ? 'N' : (op[0] == 1 ? 'I' : 'V'));

    arg.timing = 0;

    return arg;
}

class STEQR : public ::TestWithParam<steqr_tuple>
{
protected:
    STEQR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(STEQR, __float)
{
    Arguments arg = steqr_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N')
        testing_steqr_bad_arg<float>();

    testing_steqr<float>(arg);
}

TEST_P(STEQR, __double)
{
    Arguments arg = steqr_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N')
        testing_steqr_bad_arg<double>();

    testing_steqr<double>(arg);
}

TEST_P(STEQR, __float_complex)
{
    Arguments arg = steqr_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N')
        testing_steqr_bad_arg<rocblas_float_complex>();

    testing_steqr<rocblas_float_complex>(arg);
}

TEST_P(STEQR, __double_complex)
{
    Arguments arg = steqr_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N')
        testing_steqr_bad_arg<rocblas_double_complex>();

    testing_steqr<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEQR,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEQR,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));
