/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgtr_ungtr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, printable_char> orgtr_tuple;

// each size_range vector is a {n, lda}

// case when n = 0 and uplo = 'U' will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

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
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("uplo", uplo);

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

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("uplo") == 'U')
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

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGTR,
                         Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGTR, Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGTR,
                         Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGTR, Combine(ValuesIn(size_range), ValuesIn(uplo_range)));
