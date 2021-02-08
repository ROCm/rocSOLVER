/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orglx_unglx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> orglq_tuple;

// each m_size_range vector is a {M, lda, K}

// case when m = 0 and n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> m_size_range = {
    // quick return
    {0, 1, 0},
    // always invalid
    {-1, 1, 1},
    {20, 5, 1},
    {10, 10, 20},
    // invalid for case *
    {30, 30, 25},
    // normal (valid) samples
    {10, 10, 10},
    {20, 50, 20},
};

const vector<int> n_size_range = {
    // quick return
    0,
    // always invalid
    -1,
    // invalid for case *
    25,
    // normal (valid) samples
    50, 70, 130};

// for daily_lapack tests
const vector<vector<int>> large_m_size_range
    = {{164, 164, 130}, {198, 640, 198}, {130, 130, 130}, {220, 220, 140}, {400, 400, 200}};

const vector<int> large_n_size_range = {400, 640, 1000, 2000};

Arguments orglq_setup_arguments(orglq_tuple tup)
{
    vector<int> m_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = m_size[0];
    arg.N = n_size;
    arg.K = m_size[2];
    arg.lda = m_size[1];

    arg.timing = 0;

    return arg;
}

class ORGL2 : public ::TestWithParam<orglq_tuple>
{
protected:
    ORGL2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNGL2 : public ::TestWithParam<orglq_tuple>
{
protected:
    UNGL2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class ORGLQ : public ::TestWithParam<orglq_tuple>
{
protected:
    ORGLQ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNGLQ : public ::TestWithParam<orglq_tuple>
{
protected:
    UNGLQ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORGL2, __float)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<float, 0>();

    testing_orglx_unglx<float, 0>(arg);
}

TEST_P(ORGL2, __double)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<double, 0>();

    testing_orglx_unglx<double, 0>(arg);
}

TEST_P(UNGL2, __float_complex)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<rocblas_float_complex, 0>();

    testing_orglx_unglx<rocblas_float_complex, 0>(arg);
}

TEST_P(UNGL2, __double_complex)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<rocblas_double_complex, 0>();

    testing_orglx_unglx<rocblas_double_complex, 0>(arg);
}

TEST_P(ORGLQ, __float)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<float, 1>();

    testing_orglx_unglx<float, 1>(arg);
}

TEST_P(ORGLQ, __double)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<double, 1>();

    testing_orglx_unglx<double, 1>(arg);
}

TEST_P(UNGLQ, __float_complex)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<rocblas_float_complex, 1>();

    testing_orglx_unglx<rocblas_float_complex, 1>(arg);
}

TEST_P(UNGLQ, __double_complex)
{
    Arguments arg = orglq_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orglx_unglx_bad_arg<rocblas_double_complex, 1>();

    testing_orglx_unglx<rocblas_double_complex, 1>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGL2,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGL2,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGL2,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGL2,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGLQ,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGLQ,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGLQ,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGLQ,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));
