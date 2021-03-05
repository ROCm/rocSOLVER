/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_ormbr_unmbr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormbr_tuple;

// each size_range vector is a {M, N, K}

// each store_range vector is a {lda, ldc, s, t, st}
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
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'

// case when m = 0, n = 1, side = 'L', trans = 'T' and storev = 'C'
// will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> store_range = {
    // invalid
    {-1, 0, 0, 0, 0},
    {0, -1, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 0, 0, 0},
    {1, 1, 0, 0, 1},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 2, 1},
    {0, 0, 1, 0, 0},
    {0, 0, 1, 0, 1},
    {0, 0, 1, 1, 0},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 2, 0},
    {0, 0, 1, 2, 1},
};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    // invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // normal (valid) samples
    {10, 30, 5},
    {20, 5, 10},
    {20, 20, 25},
    {50, 50, 30},
    {70, 40, 40},
};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {
    {200, 150, 100}, {270, 270, 270}, {400, 400, 405}, {800, 500, 300}, {1500, 1000, 300},
};

Arguments ormbr_setup_arguments(ormbr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.storev = store[4] == 0 ? 'C' : 'R';
    arg.transA_option = (store[3] == 0 ? 'N' : (store[3] == 1 ? 'T' : 'C'));
    arg.side_option = store[2] == 0 ? 'L' : 'R';

    arg.K = size[2];
    arg.N = size[1];
    arg.M = size[0];

    arg.ldc = arg.M + store[1] * 10;

    int nq = arg.side_option == 'L' ? arg.M : arg.N;
    if(arg.storev == 'C')
    {
        arg.lda = nq;
    }
    else
    {
        arg.lda = min(nq, arg.K);
    }

    arg.lda += store[0] * 10;

    arg.timing = 0;

    return arg;
}

class ORMBR_UNMBR : public ::TestWithParam<ormbr_tuple>
{
protected:
    ORMBR_UNMBR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = ormbr_setup_arguments(GetParam());

        if(arg.M == 0 && arg.N == 1 && arg.side_option == 'L' && arg.transA_option == 'T'
           && arg.storev == 'C')
            testing_ormbr_unmbr_bad_arg<T>();

        testing_ormbr_unmbr<T>(arg);
    }
};

class ORMBR : public ORMBR_UNMBR
{
};

class UNMBR : public ORMBR_UNMBR
{
};

// non-batch tests

TEST_P(ORMBR, __float)
{
    run_tests<float>();
}

TEST_P(ORMBR, __double)
{
    run_tests<double>();
}

TEST_P(UNMBR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMBR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORMBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNMBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));
