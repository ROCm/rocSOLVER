/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

    arg.set<rocblas_int>("m", m_size[0]);
    arg.set<rocblas_int>("lda", m_size[1]);

    arg.set<rocblas_int>("n", n_size[0]);
    arg.set<rocblas_int>("k", n_size[1]);

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class ORGXR_UNGXR : public ::TestWithParam<orgqr_tuple>
{
protected:
    ORGXR_UNGXR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgqr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_orgxr_ungxr_bad_arg<T, BLOCKED>();

        testing_orgxr_ungxr<T, BLOCKED>(arg);
    }
};

class ORG2R : public ORGXR_UNGXR<false>
{
};

class UNG2R : public ORGXR_UNGXR<false>
{
};

class ORGQR : public ORGXR_UNGXR<true>
{
};

class UNGQR : public ORGXR_UNGXR<true>
{
};

// non-batch tests

TEST_P(ORG2R, __float)
{
    run_tests<float>();
}

TEST_P(ORG2R, __double)
{
    run_tests<double>();
}

TEST_P(UNG2R, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNG2R, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGQR, __float)
{
    run_tests<float>();
}

TEST_P(ORGQR, __double)
{
    run_tests<double>();
}

TEST_P(UNGQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
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
