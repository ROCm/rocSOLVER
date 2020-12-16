/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

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

    arg.M = matrix_size[0];
    arg.N = n_size[0];
    arg.K = n_size[2];
    arg.lda = matrix_size[1];
    arg.ldb = matrix_size[2];
    arg.ldc = n_size[1];

    arg.timing = 0;

    return arg;
}

class MANAGED_MALLOC : public ::TestWithParam<managed_malloc_tuple>
{
protected:
    MANAGED_MALLOC() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(MANAGED_MALLOC, __float)
{
    Arguments arg = managed_malloc_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_managed_malloc_bad_arg<float>();

    arg.batch_count = 1;
    testing_managed_malloc<float>(arg);
}

TEST_P(MANAGED_MALLOC, __double)
{
    Arguments arg = managed_malloc_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_managed_malloc_bad_arg<double>();

    arg.batch_count = 1;
    testing_managed_malloc<double>(arg);
}

TEST_P(MANAGED_MALLOC, __float_complex)
{
    Arguments arg = managed_malloc_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_managed_malloc_bad_arg<rocblas_float_complex>();

    arg.batch_count = 1;
    testing_managed_malloc<rocblas_float_complex>(arg);
}

TEST_P(MANAGED_MALLOC, __double_complex)
{
    Arguments arg = managed_malloc_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_managed_malloc_bad_arg<rocblas_double_complex>();

    arg.batch_count = 1;
    testing_managed_malloc<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         MANAGED_MALLOC,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         MANAGED_MALLOC,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
