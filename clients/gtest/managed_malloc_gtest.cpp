/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_managed_malloc.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> managed_malloc_tuple;

// each matrix_size_range is a {m, lda, ldx}

// each n_size_range is a {n, ldy, nb}

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 70}};

const vector<vector<int>> n_size_range = {
    // normal (valid) samples
    {16, 16, 10},
    {20, 30, 10}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {130, 130, 150},
};

const vector<vector<int>> large_n_size_range = {{64, 64, 60}};

Arguments managed_malloc_setup_arguments(managed_malloc_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldx", matrix_size[2]);

    arg.set<rocblas_int>("n", n_size[0]);
    arg.set<rocblas_int>("ldy", n_size[1]);
    arg.set<rocblas_int>("k", n_size[2]);

    arg.timing = 0;

    return arg;
}

class MANAGED_MALLOC : public ::TestWithParam<managed_malloc_tuple>
{
protected:
    MANAGED_MALLOC() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = managed_malloc_setup_arguments(GetParam());

        testing_managed_malloc<T>(arg);
    }
};

// non-batch tests

TEST_P(MANAGED_MALLOC, __float)
{
    run_tests<float>();
}

TEST_P(MANAGED_MALLOC, __double)
{
    run_tests<double>();
}

TEST_P(MANAGED_MALLOC, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(MANAGED_MALLOC, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         MANAGED_MALLOC,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         MANAGED_MALLOC,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
