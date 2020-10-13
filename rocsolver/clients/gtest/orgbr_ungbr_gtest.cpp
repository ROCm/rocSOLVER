/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgbr_ungbr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgbr_tuple;

// each size_range is a {M, N, K};

// each store vector is a {lda, st}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'

// case when m = 0, n = 0 and storev = 'C' will also execute the bad arguments
// test (null handle, null pointers and invalid values)

const vector<vector<int>> store = {
    // always invalid
    {-1, 0},
    {-1, 1},
    // normal (valid) samples
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // always quick return
    {0, 0, 0},
    // quick return for storev = 'R' invalid for 'C'
    {0, 1, 0},
    // quick return for storev = 'C' invalid for 'R'
    {1, 0, 0},
    // always invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // invalid for storev = 'C'
    {10, 30, 5},
    // invalid for storev = 'R'
    {30, 10, 5},
    // always invalid
    {30, 10, 20},
    {10, 30, 20},
    // normal (valid) samples
    {30, 30, 0},
    {20, 20, 20},
    {50, 50, 50},
    {100, 100, 50}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{150, 150, 100}, {270, 270, 270},    {400, 400, 400},
       {800, 800, 300}, {1000, 1000, 1000}, {1500, 1500, 800}};

Arguments orgbr_setup_arguments(orgbr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.storev = store[1] == 1 ? 'R' : 'C';
    arg.K = size[2];
    arg.M = size[0];
    arg.lda = size[0];
    arg.N = size[1];

    arg.lda += store[0] * 10;

    arg.timing = 0;

    return arg;
}

class ORGBR : public ::TestWithParam<orgbr_tuple>
{
protected:
    ORGBR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNGBR : public ::TestWithParam<orgbr_tuple>
{
protected:
    UNGBR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORGBR, __float)
{
    Arguments arg = orgbr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.storev == 'C')
        testing_orgbr_ungbr_bad_arg<float>();

    testing_orgbr_ungbr<float>(arg);
}

TEST_P(ORGBR, __double)
{
    Arguments arg = orgbr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.storev == 'C')
        testing_orgbr_ungbr_bad_arg<double>();

    testing_orgbr_ungbr<double>(arg);
}

TEST_P(UNGBR, __float_complex)
{
    Arguments arg = orgbr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.storev == 'C')
        testing_orgbr_ungbr_bad_arg<rocblas_float_complex>();

    testing_orgbr_ungbr<rocblas_float_complex>(arg);
}

TEST_P(UNGBR, __double_complex)
{
    Arguments arg = orgbr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.storev == 'C')
        testing_orgbr_ungbr_bad_arg<rocblas_double_complex>();

    testing_orgbr_ungbr<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORGBR, Combine(ValuesIn(large_size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGBR, Combine(ValuesIn(size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNGBR, Combine(ValuesIn(large_size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGBR, Combine(ValuesIn(size_range), ValuesIn(store)));
