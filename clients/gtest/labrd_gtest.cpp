/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_labrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> labrd_tuple;

// each matrix_size_range is a {m, lda, ldx}

// each n_size_range is a {n, ldy, nb}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return (if nb = 0, else invalid)
    {0, 1, 1},
    // invalid
    {1, 1, 0},
    {-1, 1, 1},
    {1, 1, -1},
    {20, 5, 20},
    {20, 20, 5},
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 70},
    {130, 130, 150},
    {150, 200, 200}};

const vector<vector<int>> n_size_range = {
    // quick return
    {0, 1, 0},
    {1, 1, 0},
    // invalid
    {-1, 1, 1},
    {1, 1, -1},
    {20, 5, 20},
    {20, 20, 25},
    // normal (valid) samples
    {16, 16, 10},
    {20, 30, 10},
    {120, 120, 30},
    {150, 200, 30}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152, 152},
    {640, 640, 656},
    {1000, 1024, 1000},
};

const vector<vector<int>> large_n_size_range
    = {{64, 64, 60}, {98, 98, 60}, {130, 130, 100}, {220, 240, 100}, {400, 450, 100}};

Arguments labrd_setup_arguments(labrd_tuple tup)
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

class LABRD : public ::TestWithParam<labrd_tuple>
{
protected:
    LABRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LABRD, __float)
{
    Arguments arg = labrd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_labrd_bad_arg<float>();

    arg.batch_count = 1;
    testing_labrd<float>(arg);
}

TEST_P(LABRD, __double)
{
    Arguments arg = labrd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_labrd_bad_arg<double>();

    arg.batch_count = 1;
    testing_labrd<double>(arg);
}

TEST_P(LABRD, __float_complex)
{
    Arguments arg = labrd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_labrd_bad_arg<rocblas_float_complex>();

    arg.batch_count = 1;
    testing_labrd<rocblas_float_complex>(arg);
}

TEST_P(LABRD, __double_complex)
{
    Arguments arg = labrd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_labrd_bad_arg<rocblas_double_complex>();

    arg.batch_count = 1;
    testing_labrd<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LABRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LABRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
