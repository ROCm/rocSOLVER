/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_ormtr_unmtr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormtr_tuple;

// each size_range vector is a {M, N}

// each store vector is a {lda, ldc, s, t, u}
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
// if u = 0, then uplo = 'U'
// if u = 1, then uplo = 'L'

// case when m = 0, n = 1, side = 'L', trans = 'T' and uplo = 'U'
// will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> store = {
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
    {0, 1},
    {1, 0},
    // invalid
    {-1, 1},
    {1, -1},
    // normal (valid) samples
    {10, 30},
    {20, 5},
    {20, 20},
    {50, 50},
    {70, 40},
};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {
    {200, 150}, {270, 270}, {400, 400}, {800, 500}, {1500, 1000},
};

Arguments ormtr_setup_arguments(ormtr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.transA_option = (store[3] == 0 ? 'N' : (store[3] == 1 ? 'T' : 'C'));
    arg.uplo_option = store[4] == 0 ? 'U' : 'L';
    arg.side_option = store[2] == 0 ? 'L' : 'R';

    arg.N = size[1];
    arg.M = size[0];

    arg.ldc = arg.M + store[1] * 10;

    int nq = arg.side_option == 'L' ? arg.M : arg.N;
    arg.lda = nq;

    arg.lda += store[0] * 10;

    arg.timing = 0;

    return arg;
}

class ORMTR : public ::TestWithParam<ormtr_tuple>
{
protected:
    ORMTR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class UNMTR : public ::TestWithParam<ormtr_tuple>
{
protected:
    UNMTR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ORMTR, __float)
{
    Arguments arg = ormtr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 1 && arg.side_option == 'L' && arg.transA_option == 'T'
       && arg.uplo_option == 'U')
        testing_ormtr_unmtr_bad_arg<float>();

    testing_ormtr_unmtr<float>(arg);
}

TEST_P(ORMTR, __double)
{
    Arguments arg = ormtr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 1 && arg.side_option == 'L' && arg.transA_option == 'T'
       && arg.uplo_option == 'U')
        testing_ormtr_unmtr_bad_arg<double>();

    testing_ormtr_unmtr<double>(arg);
}

TEST_P(UNMTR, __float_complex)
{
    Arguments arg = ormtr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 1 && arg.side_option == 'L' && arg.transA_option == 'T'
       && arg.uplo_option == 'U')
        testing_ormtr_unmtr_bad_arg<rocblas_float_complex>();

    testing_ormtr_unmtr<rocblas_float_complex>(arg);
}

TEST_P(UNMTR, __double_complex)
{
    Arguments arg = ormtr_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 1 && arg.side_option == 'L' && arg.transA_option == 'T'
       && arg.uplo_option == 'U')
        testing_ormtr_unmtr_bad_arg<rocblas_double_complex>();

    testing_ormtr_unmtr<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORMTR, Combine(ValuesIn(large_size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMTR, Combine(ValuesIn(size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNMTR, Combine(ValuesIn(large_size_range), ValuesIn(store)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMTR, Combine(ValuesIn(size_range), ValuesIn(store)));
