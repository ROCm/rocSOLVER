/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgtr_ungtr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> orgtr_tuple;

// each size_range vector is a {n, lda}

// case when n = 0 and uplo = 'U' will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<int> uplo = {0, 1};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100},
    {100, 150}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192}, {500, 600}, {640, 640}, {1000, 1024}};

Arguments orgtr_setup_arguments(orgtr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    int uplo = std::get<1>(tup);

    Arguments arg;

    arg.uplo_option = uplo == 1 ? 'U' : 'L';
    arg.N = size[0];
    arg.lda = size[1];

    arg.timing = 0;

    return arg;
}

class ORGTR_UNGTR : public ::TestWithParam<orgtr_tuple>
{
protected:
    ORGTR_UNGTR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgtr_setup_arguments(GetParam());

        if(arg.N == 0 && arg.uplo_option == 'U')
            testing_orgtr_ungtr_bad_arg<T>();

        testing_orgtr_ungtr<T>(arg);
    }
};

class ORGTR : public ORGTR_UNGTR
{
};

class UNGTR : public ORGTR_UNGTR
{
};

// non-batch tests

TEST_P(ORGTR, __float)
{
    run_tests<float>();
}

TEST_P(ORGTR, __double)
{
    run_tests<double>();
}

TEST_P(UNGTR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGTR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORGTR, Combine(ValuesIn(large_size_range), ValuesIn(uplo)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGTR, Combine(ValuesIn(size_range), ValuesIn(uplo)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNGTR, Combine(ValuesIn(large_size_range), ValuesIn(uplo)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGTR, Combine(ValuesIn(size_range), ValuesIn(uplo)));
