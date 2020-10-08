/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> getrs_tuple;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, trans};
// if trans = 0 then no transpose
// if trans = 1 then transpose
// if trans = 2 then conjugate transpose

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};
const vector<vector<int>> matrix_sizeB_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 2},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range
    = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};
const vector<vector<int>> large_matrix_sizeB_range = {
    {100, 0}, {150, 0}, {200, 1}, {524, 2}, {1000, 2},
};

Arguments getrs_setup_arguments(getrs_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_sizeA[0];
    arg.N = matrix_sizeB[0];
    arg.lda = matrix_sizeA[1];
    arg.ldb = matrix_sizeA[2];

    if(matrix_sizeB[1] == 0)
        arg.transA_option = 'N';
    else if(matrix_sizeB[1] == 1)
        arg.transA_option = 'T';
    else
        arg.transA_option = 'C';

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = arg.M;
    arg.bsa = arg.lda * arg.M;
    arg.bsb = arg.ldb * arg.N;

    return arg;
}

class GETRS : public ::TestWithParam<getrs_tuple>
{
protected:
    GETRS() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(GETRS, __float)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, false, float>();

    arg.batch_count = 1;
    testing_getrs<false, false, float>(arg);
}

TEST_P(GETRS, __double)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, false, double>();

    arg.batch_count = 1;
    testing_getrs<false, false, double>(arg);
}

TEST_P(GETRS, __float_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_getrs<false, false, rocblas_float_complex>(arg);
}

TEST_P(GETRS, __double_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_getrs<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(GETRS, batched__float)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true, true, float>();

    arg.batch_count = 3;
    testing_getrs<true, true, float>(arg);
}

TEST_P(GETRS, batched__double)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true, true, double>();

    arg.batch_count = 3;
    testing_getrs<true, true, double>(arg);
}

TEST_P(GETRS, batched__float_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_getrs<true, true, rocblas_float_complex>(arg);
}

TEST_P(GETRS, batched__double_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_getrs<true, true, rocblas_double_complex>(arg);
}

// strided_batched tests

TEST_P(GETRS, strided_batched__float)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, true, float>();

    arg.batch_count = 3;
    testing_getrs<false, true, float>(arg);
}

TEST_P(GETRS, strided_batched__double)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, true, double>();

    arg.batch_count = 3;
    testing_getrs<false, true, double>(arg);
}

TEST_P(GETRS, strided_batched__float_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_getrs<false, true, rocblas_float_complex>(arg);
}

TEST_P(GETRS, strided_batched__double_complex)
{
    Arguments arg = getrs_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_getrs<false, true, rocblas_double_complex>(arg);
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
