/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

    arg.set<rocblas_int>("m", m_size[0]);
    arg.set<rocblas_int>("lda", m_size[1]);
    arg.set<rocblas_int>("k", m_size[2]);

    arg.set<rocblas_int>("n", n_size);

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class ORGLX_UNGLX : public ::TestWithParam<orglq_tuple>
{
protected:
    ORGLX_UNGLX() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = orglq_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_orglx_unglx_bad_arg<T, BLOCKED>();

        testing_orglx_unglx<T, BLOCKED>(arg);
    }
};

class ORGL2 : public ORGLX_UNGLX<false>
{
};

class UNGL2 : public ORGLX_UNGLX<false>
{
};

class ORGLQ : public ORGLX_UNGLX<true>
{
};

class UNGLQ : public ORGLX_UNGLX<true>
{
};

// non-batch tests

TEST_P(ORGL2, __float)
{
    run_tests<float>();
}

TEST_P(ORGL2, __double)
{
    run_tests<double>();
}

TEST_P(UNGL2, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGL2, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGLQ, __float)
{
    run_tests<float>();
}

TEST_P(ORGLQ, __double)
{
    run_tests<double>();
}

TEST_P(UNGLQ, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGLQ, __double_complex)
{
    run_tests<rocblas_double_complex>();
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
