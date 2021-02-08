/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> larf_tuple;

// each size_range vector is a {M,N,lda}

// each incx_range vector is a {incx,s}
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'

// case when M == 0 and incx == 0  also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> incx_range = {
    // invalid
    {0, 0},
    // normal (valid) samples
    {-10, 0},
    {-5, 1},
    {-1, 0},
    {1, 1},
    {5, 0},
    {10, 1}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5},
    // normal (valid) samples
    {12, 20, 12},
    {20, 15, 20},
    {35, 35, 50}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

Arguments larf_setup_arguments(larf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> inc = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.incx = inc[0];

    arg.side_option = inc[1] == 1 ? 'R' : 'L';

    arg.timing = 0;

    return arg;
}

class LARF : public ::TestWithParam<larf_tuple>
{
protected:
    LARF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LARF, __float)
{
    Arguments arg = larf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.incx == 0)
        testing_larf_bad_arg<float>();

    testing_larf<float>(arg);
}

TEST_P(LARF, __double)
{
    Arguments arg = larf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.incx == 0)
        testing_larf_bad_arg<double>();

    testing_larf<double>(arg);
}

TEST_P(LARF, __float_complex)
{
    Arguments arg = larf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.incx == 0)
        testing_larf_bad_arg<rocblas_float_complex>();

    testing_larf<rocblas_float_complex>(arg);
}

TEST_P(LARF, __double_complex)
{
    Arguments arg = larf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.incx == 0)
        testing_larf_bad_arg<rocblas_double_complex>();

    testing_larf<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(incx_range)));
