/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgxl_ungxl.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgql_tuple;

// each n_size_range vector is a {N, K}

// each m_size_range vector is a {M, lda}

// case when m = 0 and n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> n_size_range = {
    // quick return
    {0, 0},
    // always invalid
    {-1, 1},
    {1, -1},
    {10, 20},
    // invalid for case *
    {30, 25},
    // normal (valid) samples
    {10, 10},
    {20, 20},
};

const vector<vector<int>> m_size_range = {
    // quick return
    {0, 1},
    // always invalid
    {-1, 1},
    {20, 5},
    // invalid for case *
    {25, 25},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130}};

// for daily_lapack tests
const vector<vector<int>> large_n_size_range
    = {{164, 130}, {198, 198}, {130, 130}, {220, 140}, {400, 200}};

const vector<vector<int>> large_m_size_range = {{400, 640}, {640, 640}, {1000, 1000}, {2000, 2000}};

Arguments orgql_setup_arguments(orgql_tuple tup)
{
    vector<int> m_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = m_size[0];
    arg.N = n_size[0];
    arg.K = n_size[1];
    arg.lda = m_size[1];

    arg.timing = 0;

    return arg;
}

class ORG2L : public ::TestWithParam<orgql_tuple>
{
protected:
    ORG2L() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNG2L : public ::TestWithParam<orgql_tuple>
{
protected:
    UNG2L() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class ORGQL : public ::TestWithParam<orgql_tuple>
{
protected:
    ORGQL() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNGQL : public ::TestWithParam<orgql_tuple>
{
protected:
    UNGQL() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORG2L, __float)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<float, 0>();

    testing_orgxl_ungxl<float, 0>(arg);
}

TEST_P(ORG2L, __double)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<double, 0>();

    testing_orgxl_ungxl<double, 0>(arg);
}

TEST_P(UNG2L, __float_complex)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<rocblas_float_complex, 0>();

    testing_orgxl_ungxl<rocblas_float_complex, 0>(arg);
}

TEST_P(UNG2L, __double_complex)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<rocblas_double_complex, 0>();

    testing_orgxl_ungxl<rocblas_double_complex, 0>(arg);
}

TEST_P(ORGQL, __float)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<float, 1>();

    testing_orgxl_ungxl<float, 1>(arg);
}

TEST_P(ORGQL, __double)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<double, 1>();

    testing_orgxl_ungxl<double, 1>(arg);
}

TEST_P(UNGQL, __float_complex)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<rocblas_float_complex, 1>();

    testing_orgxl_ungxl<rocblas_float_complex, 1>(arg);
}

TEST_P(UNGQL, __double_complex)
{
    Arguments arg = orgql_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxl_ungxl_bad_arg<rocblas_double_complex, 1>();

    testing_orgxl_ungxl<rocblas_double_complex, 1>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORG2L,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORG2L,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNG2L,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNG2L,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGQL,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGQL,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGQL,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGQL,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));
