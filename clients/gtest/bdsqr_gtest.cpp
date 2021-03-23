/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_bdsqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> bdsqr_tuple;

// each size_range is a {n, nv, nu, nc}

// each opt_range is a {uplo, ldv, ldu, ldc}
// if uplo = 0, then is upper bidiagonal
// if uplo = 1, then is lower bidiagonal
// if ldx = -1, then ldx < limit (invalid size)
// if ldx = 0, then ldx = limit
// if ldx = 1, then ldx > limit

// case when n = 0 and uplo = 'L' will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 1},
    // invalid
    {-1, 1, 1, 1},
    {1, -1, 1, 1},
    {1, 1, -1, 1},
    {1, 1, 1, -1},
    // normal (valid) samples
    {15, 10, 10, 10},
    {20, 0, 0, 15},
    {30, 30, 50, 0},
    {50, 60, 20, 0},
    {70, 0, 0, 0}};

const vector<vector<int>> opt_range = {
    // invalid
    {0, -1, 0, 0},
    {0, 0, -1, 0},
    {0, 0, 0, -1},
    // normal (valid) samples
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{152, 152, 152, 152}, {640, 640, 656, 700}, {1000, 1024, 1000, 80}, {2000, 0, 0, 0}};

const vector<vector<int>> large_opt_range = {{0, 0, 0, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 0}};

Arguments bdsqr_setup_arguments(bdsqr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt = std::get<1>(tup);

    Arguments arg;

    rocblas_int n = size[0];
    rocblas_int nv = size[1];
    rocblas_int nu = size[2];
    rocblas_int nc = size[3];
    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("nv", nv);
    arg.set<rocblas_int>("nu", nu);
    arg.set<rocblas_int>("nc", nc);

    arg.set<char>("uplo", opt[0] ? 'L' : 'U');

    arg.set<rocblas_int>("ldv", (nv > 0 ? n : 1) + opt[1] * 10);
    arg.set<rocblas_int>("ldu", (nu > 0 ? nu : 1) + opt[2] * 10);
    arg.set<rocblas_int>("ldc", (nc > 0 ? n : 1) + opt[3] * 10);

    arg.timing = 0;

    return arg;
}

class BDSQR : public ::TestWithParam<bdsqr_tuple>
{
protected:
    BDSQR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = bdsqr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("uplo") == 'L')
            testing_bdsqr_bad_arg<T>();

        testing_bdsqr<T>(arg);
    }
};

// non-batch tests

TEST_P(BDSQR, __float)
{
    run_tests<float>();
}

TEST_P(BDSQR, __double)
{
    run_tests<double>();
}

TEST_P(BDSQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(BDSQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         BDSQR,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, BDSQR, Combine(ValuesIn(size_range), ValuesIn(opt_range)));
