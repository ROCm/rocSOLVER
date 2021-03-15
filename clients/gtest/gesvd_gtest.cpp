/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesvd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvd_tuple;

// each size_range vector is a {m, n, fa};
// if fa = 0 then no fast algorithm is allowed
// if fa = 1 fast algorithm is used when possible

// each opt_range vector is a {lda, ldu, ldv, leftsv, rightsv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit
// if leftsv (rightsv) = 0 then overwrite singular vectors
// if leftsv (rightsv) = 1 then compute singular vectors
// if leftsv (rightsv) = 2 then compute all orthogonal matrix
// if leftsv (rightsv) = 3 then no singular vectors are computed

// case when m = n = 0 and rightsv = leftsv = 3 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    // invalid
    {-1, 1, 0},
    {1, -1, 0},
    // normal (valid) samples
    {1, 1, 0},
    {20, 20, 0},
    {40, 30, 0},
    {60, 30, 0},
    {60, 30, 1},
    {30, 40, 0},
    {30, 60, 0},
    {30, 60, 1}};

const vector<vector<int>> opt_range = {
    // invalid
    {-1, 0, 0, 2, 2},
    {0, -1, 0, 1, 2},
    {0, 0, -1, 2, 1},
    {0, 0, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 1, 3, 3},
    {0, 0, 1, 3, 2},
    {0, 1, 0, 3, 1},
    {0, 1, 1, 3, 0},
    {1, 0, 0, 2, 3},
    {1, 0, 1, 2, 2},
    {1, 1, 0, 2, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 1, 3},
    {0, 0, 0, 1, 2},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 3},
    {0, 0, 0, 0, 2},
    {0, 0, 0, 0, 1}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{120, 100, 0}, {300, 120, 0}, {300, 120, 1}, {100, 120, 1}, {120, 300, 0}, {120, 300, 1}};

const vector<vector<int>> large_opt_range
    = {{0, 0, 0, 3, 3}, {1, 0, 0, 0, 1}, {0, 1, 0, 1, 0}, {0, 0, 1, 1, 1},
       {0, 0, 0, 3, 0}, {0, 0, 0, 1, 3}, {0, 0, 0, 3, 2}};

Arguments gesvd_setup_arguments(gesvd_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt = std::get<1>(tup);

    Arguments arg;

    // sizes
    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    // fast algorithm
    if(size[2] == 0)
        arg.set<char>("fast_alg", 'I');
    else
        arg.set<char>("fast_alg", 'O');

    // leading dimensions
    arg.set<rocblas_int>("lda", m + opt[0] * 10);
    arg.set<rocblas_int>("ldu", m + opt[1] * 10);
    if(opt[4] == 2)
        arg.set<rocblas_int>("ldv", n + opt[2] * 10);
    else
        arg.set<rocblas_int>("ldv", min(m, n) + opt[2] * 10);

    // vector options
    if(opt[3] == 0)
        arg.set<char>("left_svect", 'O');
    else if(opt[3] == 1)
        arg.set<char>("left_svect", 'S');
    else if(opt[3] == 2)
        arg.set<char>("left_svect", 'A');
    else
        arg.set<char>("left_svect", 'N');

    if(opt[4] == 0)
        arg.set<char>("right_svect", 'O');
    else if(opt[4] == 1)
        arg.set<char>("right_svect", 'S');
    else if(opt[4] == 2)
        arg.set<char>("right_svect", 'A');
    else
        arg.set<char>("right_svect", 'N');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GESVD : public ::TestWithParam<gesvd_tuple>
{
protected:
    GESVD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0
           && arg.peek<char>("left_svect") == 'N' && arg.peek<char>("right_svect") == 'N')
            testing_gesvd_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gesvd<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GESVD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GESVD, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GESVD, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GESVD, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GESVD, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GESVD, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVD, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVD, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVD, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESVD,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, GESVD, Combine(ValuesIn(size_range), ValuesIn(opt_range)));
