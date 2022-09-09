/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesvdx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvdx_tuple;

// each size_range vector is a {m, n, lda, ldu, ldv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit

// each opt_range vector is a {leftsv, rightsv, rng, vl, vu, il, iu};
// if leftsv (rightsv) = 1 then compute singular vectors
// if leftsv (rightsv) = 0 then no singular vectors are computed
// if rng = 0, then find all singular values
// if rng = 1, then find singular values in (vl, vu]
// if rng = 2, then find the il-th to the iu-th singular value

// case when m = n = 0, rightsv = leftsv = 0 and rng = 0 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0},
    {1, 0, 0, 0, 0},
    // invalid
    {-1, 1, 0, 0, 0},
    {1, -1, 0, 0, 0},
    {10, 10, -1, 0, 0},
    {10, 10, 0, -1, 0},
    {10, 10, 0, 0, -1},
    // normal (valid) samples
    {1, 1, 0, 0, 0},
    {20, 20, 0, 0, 0},
    {40, 30, 0, 0, 0},
    {30, 40, 0, 0, 0},
    {30, 30, 1, 0, 0},
    {60, 40, 0, 1, 0},
    {40, 60, 0, 0, 1},
    {50, 50, 1, 1, 1}};

const vector<vector<int>> opt_range = {
    // always invalid
    {0, 0, 1, 2, 1, 0, 0},
    {0, 0, 1, -1, 1, 0, 0},
    {0, 0, 2, 0, 0, 2, 1},
    {0, 0, 2, 0, 0, 10, 80},
    // valid only when n=0
    {0, 0, 2, 0, 0, 1, 0},
    // valid only when n>1
    {0, 0, 2, 0, 0, 1, 5},
    {0, 1, 2, 0, 0, 1, 15},
    {1, 0, 2, 0, 0, 7, 12},
    {1, 1, 2, 0, 0, 10, 15},
    // always valid samples
    {0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 10, 0, 0},
    {0, 1, 1, 5, 12, 0, 0},
    {1, 0, 1, 0, 12, 0, 0},
    {1, 1, 1, 12, 18, 0, 0}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{100, 100, 1, 0, 0}, {300, 120, 0, 0, 1}, {200, 300, 0, 0, 0}};

const vector<vector<int>> large_opt_range
    = {{0, 0, 1, 5, 10, 0, 0}, {1, 0, 1, 10, 20, 0, 0}, {0, 1, 2, 0, 0, 3, 10}, {1, 1, 0, 0, 0, 0, 0}};

Arguments gesvdx_setup_arguments(gesvdx_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt = std::get<1>(tup);

    Arguments arg;

    // sizes
    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    // leading dimensions
    arg.set<rocblas_int>("lda", m + size[2] * 10);
    arg.set<rocblas_int>("ldu", m + size[3] * 10);
    arg.set<rocblas_int>("ldv", min(m, n) + size[4] * 10);

    // vector options
    if(opt[0] == 0)
        arg.set<char>("left_svect", 'N');
    else
        arg.set<char>("left_svect", 'S');

    if(opt[1] == 0)
        arg.set<char>("right_svect", 'N');
    else
        arg.set<char>("right_svect", 'S');

    // only testing standard use case/defaults for strides

    // ranges
    arg.set<char>("srange", (opt[2] == 0 ? 'A' : (opt[2] == 1 ? 'V' : 'I')));
    arg.set<double>("vl", opt[3]);
    arg.set<double>("vu", opt[4]);
    arg.set<rocblas_int>("il", opt[5]);
    arg.set<rocblas_int>("iu", opt[6]);

    arg.timing = 0;

    return arg;
}

class GESVDX : public ::TestWithParam<gesvdx_tuple>
{
protected:
    GESVDX() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvdx_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0
           && arg.peek<char>("left_svect") == 'N' && arg.peek<char>("right_svect") == 'N'
           && arg.peek<char>("srange") == 'A')
            testing_gesvdx_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gesvdx<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GESVDX, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVDX, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVDX, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVDX, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GESVDX, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GESVDX, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GESVDX, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GESVDX, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GESVDX, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVDX, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVDX, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVDX, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESVDX,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, GESVDX, Combine(ValuesIn(size_range), ValuesIn(opt_range)));
