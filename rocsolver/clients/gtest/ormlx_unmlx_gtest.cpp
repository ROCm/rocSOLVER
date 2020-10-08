/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_ormlx_unmlx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormlq_tuple;

// each size_range vector is a {M, N, K};

// each op_range is a {lda, ldc, s, t}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if ldc = -1, then ldc < limit (invalid size)
// if ldc = 0, then ldc = limit
// if ldc = 1, then ldc > limit
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'
// if t = 0, then trans = 'N'
// if t = 1, then trans = 'T'
// if t = 2, then trans = 'C'

// case when m = 0, side = 'L' and trans = 'T' will also execute the bad
// arguments test (null handle, null pointers and invalid values)

const vector<vector<int>> op_range = {
    // invalid
    {-1, 0, 0, 0},
    {0, -1, 0, 0},
    // normal (valid) samples
    {0, 0, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 0, 2},
    {0, 0, 1, 0},
    {0, 0, 1, 1},
    {0, 0, 1, 2},
    {1, 1, 0, 0}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 0},
    {1, 0, 0},
    {30, 30, 0},
    // always invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // invalid for side = 'R'
    {20, 10, 20},
    // invalid for side = 'L'
    {15, 25, 25},
    // normal (valid) samples
    {40, 40, 40},
    {45, 40, 30},
    {50, 50, 20}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{100, 100, 100}, {150, 100, 80}, {300, 400, 300}, {1024, 1000, 950}, {1500, 1500, 1000}};

Arguments ormlq_setup_arguments(ormlq_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    Arguments arg;

    arg.M = size[0];
    arg.N = size[1];
    arg.K = size[2];
    arg.ldc = arg.M + op[1] * 10;
    arg.lda = arg.K + op[0] * 10;

    arg.transA_option = (op[3] == 0 ? 'N' : (op[3] == 1 ? 'T' : 'C'));
    arg.side_option = op[2] == 0 ? 'L' : 'R';

    arg.timing = 0;

    return arg;
}

class ORML2 : public ::TestWithParam<ormlq_tuple>
{
protected:
    ORML2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNML2 : public ::TestWithParam<ormlq_tuple>
{
protected:
    UNML2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class ORMLQ : public ::TestWithParam<ormlq_tuple>
{
protected:
    ORMLQ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNMLQ : public ::TestWithParam<ormlq_tuple>
{
protected:
    UNMLQ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORML2, __float)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<float, 0>();

    testing_ormlx_unmlx<float, 0>(arg);
}

TEST_P(ORML2, __double)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<double, 0>();

    testing_ormlx_unmlx<double, 0>(arg);
}

TEST_P(UNML2, __float_complex)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<rocblas_float_complex, 0>();

    testing_ormlx_unmlx<rocblas_float_complex, 0>(arg);
}

TEST_P(UNML2, __double_complex)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<rocblas_double_complex, 0>();

    testing_ormlx_unmlx<rocblas_double_complex, 0>(arg);
}

TEST_P(ORMLQ, __float)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<float, 1>();

    testing_ormlx_unmlx<float, 1>(arg);
}

TEST_P(ORMLQ, __double)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<double, 1>();

    testing_ormlx_unmlx<double, 1>(arg);
}

TEST_P(UNMLQ, __float_complex)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<rocblas_float_complex, 1>();

    testing_ormlx_unmlx<rocblas_float_complex, 1>(arg);
}

TEST_P(UNMLQ, __double_complex)
{
    Arguments arg = ormlq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.side_option == 'L' && arg.transA_option == 'T')
        testing_ormlx_unmlx_bad_arg<rocblas_double_complex, 1>();

    testing_ormlx_unmlx<rocblas_double_complex, 1>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORML2, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORML2, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNML2, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNML2, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORMLQ, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMLQ, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNMLQ, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMLQ, Combine(ValuesIn(size_range), ValuesIn(op_range)));
