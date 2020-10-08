/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgxr_ungxr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgqr_tuple;

// each m_size_range vector is a {M, lda}

// each n_size_range vector is a {N, K}

// case when m = 0 and n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> m_size_range = {
    // quick return
    {0, 1},
    // always invalid
    {-1, 1},
    {20, 5},
    // invalid for case *
    {50, 50},
    // normal (valid) samples
    {70, 100},
    {130, 130}};

const vector<vector<int>> n_size_range = {
    // quick return
    {0, 1},
    // always invalid
    {-1, 1},
    {1, -1},
    {10, 20},
    // invalid for case *
    {55, 55},
    // normal (valid) samples
    {10, 0},
    {20, 20},
    {35, 25}};

// for daily_lapack tests
const vector<vector<int>> large_m_size_range = {{400, 410}, {640, 640}, {1000, 1024}, {2000, 2000}};

const vector<vector<int>> large_n_size_range
    = {{164, 162}, {198, 140}, {130, 130}, {220, 220}, {400, 200}};

Arguments orgqr_setup_arguments(orgqr_tuple tup)
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

class ORG2R : public ::TestWithParam<orgqr_tuple>
{
protected:
    ORG2R() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNG2R : public ::TestWithParam<orgqr_tuple>
{
protected:
    UNG2R() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class ORGQR : public ::TestWithParam<orgqr_tuple>
{
protected:
    ORGQR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNGQR : public ::TestWithParam<orgqr_tuple>
{
protected:
    UNGQR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORG2R, __float)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<float, 0>();

    testing_orgxr_ungxr<float, 0>(arg);
}

TEST_P(ORG2R, __double)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<double, 0>();

    testing_orgxr_ungxr<double, 0>(arg);
}

TEST_P(UNG2R, __float_complex)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<rocblas_float_complex, 0>();

    testing_orgxr_ungxr<rocblas_float_complex, 0>(arg);
}

TEST_P(UNG2R, __double_complex)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<rocblas_double_complex, 0>();

    testing_orgxr_ungxr<rocblas_double_complex, 0>(arg);
}

TEST_P(ORGQR, __float)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<float, 1>();

    testing_orgxr_ungxr<float, 1>(arg);
}

TEST_P(ORGQR, __double)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<double, 1>();

    testing_orgxr_ungxr<double, 1>(arg);
}

TEST_P(UNGQR, __float_complex)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<rocblas_float_complex, 1>();

    testing_orgxr_ungxr<rocblas_float_complex, 1>(arg);
}

TEST_P(UNGQR, __double_complex)
{
    Arguments arg = orgqr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_orgxr_ungxr_bad_arg<rocblas_double_complex, 1>();

    testing_orgxr_ungxr<rocblas_double_complex, 1>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORG2R,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORG2R,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNG2R,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNG2R,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGQR,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGQR,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGQR,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGQR,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));
