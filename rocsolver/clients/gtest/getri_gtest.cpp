/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> getri_tuple;

// each matrix_size_range vector is a {n, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {32, 32, 0},
    {50, 50, 1},
    {70, 100, 0},
    {100, 150, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 1}, {500, 600, 1}, {640, 640, 0}, {1000, 1024, 0}, {1200, 1230, 0}};

Arguments getri_setup_arguments(getri_tuple tup)
{
    // vector<int> matrix_size = std::get<0>(tup);

    Arguments arg;

    arg.N = tup[0];
    arg.lda = tup[1];

    arg.timing = 0;
    arg.singular = tup[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = arg.N;
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GETRI : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(GETRI, __float)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, false, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getri<false, false, float>(arg);

    arg.singular = 0;
    testing_getri<false, false, float>(arg);
}

TEST_P(GETRI, __double)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, false, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getri<false, false, double>(arg);

    arg.singular = 0;
    testing_getri<false, false, double>(arg);
}

TEST_P(GETRI, __float_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getri<false, false, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getri<false, false, rocblas_float_complex>(arg);
}

TEST_P(GETRI, __double_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getri<false, false, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getri<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(GETRI, batched__float)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, true, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, true, float>(arg);

    arg.singular = 0;
    testing_getri<true, true, float>(arg);
}

TEST_P(GETRI, batched__double)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, true, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, true, double>(arg);

    arg.singular = 0;
    testing_getri<true, true, double>(arg);
}

TEST_P(GETRI, batched__float_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, true, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getri<true, true, rocblas_float_complex>(arg);
}

TEST_P(GETRI, batched__double_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, true, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getri<true, true, rocblas_double_complex>(arg);
}

// strided_batched tests

TEST_P(GETRI, strided_batched__float)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, true, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<false, true, float>(arg);

    arg.singular = 0;
    testing_getri<false, true, float>(arg);
}

TEST_P(GETRI, strided_batched__double)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, true, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<false, true, double>(arg);

    arg.singular = 0;
    testing_getri<false, true, double>(arg);
}

TEST_P(GETRI, strided_batched__float_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<false, true, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getri<false, true, rocblas_float_complex>(arg);
}

TEST_P(GETRI, strided_batched__double_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<false, true, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getri<false, true, rocblas_double_complex>(arg);
}

// outofplace_batched tests

TEST_P(GETRI, outofplace_batched__float)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, false, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, false, float>(arg);

    arg.singular = 0;
    testing_getri<true, false, float>(arg);
}

TEST_P(GETRI, outofplace_batched__double)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, false, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, false, double>(arg);

    arg.singular = 0;
    testing_getri<true, false, double>(arg);
}

TEST_P(GETRI, outofplace_batched__float_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, false, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, false, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getri<true, false, rocblas_float_complex>(arg);
}

TEST_P(GETRI, outofplace_batched__double_complex)
{
    Arguments arg = getri_setup_arguments(GetParam());

    if(arg.N == 0)
        testing_getri_bad_arg<true, false, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getri<true, false, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getri<true, false, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI, ValuesIn(matrix_size_range));
