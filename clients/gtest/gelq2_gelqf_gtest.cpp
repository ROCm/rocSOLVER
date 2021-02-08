/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gelq2_gelqf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gelqf_tuple;

// each matrix_size_range is a {m, lda}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {64, 98, 130, 220, 400};

Arguments gelqf_setup_arguments(gelqf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = n_size;
    arg.lda = matrix_size[1];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = min(arg.M, arg.N);
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GELQ2 : public ::TestWithParam<gelqf_tuple>
{
protected:
    GELQ2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GELQF : public ::TestWithParam<gelqf_tuple>
{
protected:
    GELQF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(GELQ2, __float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 0, float>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 0, float>(arg);
}

TEST_P(GELQ2, __double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 0, double>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 0, double>(arg);
}

TEST_P(GELQ2, __float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 0, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 0, rocblas_float_complex>(arg);
}

TEST_P(GELQ2, __double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 0, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 0, rocblas_double_complex>(arg);
}

TEST_P(GELQF, __float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 1, float>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 1, float>(arg);
}

TEST_P(GELQF, __double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 1, double>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 1, double>(arg);
}

TEST_P(GELQF, __float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 1, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 1, rocblas_float_complex>(arg);
}

TEST_P(GELQF, __double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, false, 1, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_gelq2_gelqf<false, false, 1, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(GELQ2, batched__float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 0, float>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 0, float>(arg);
}

TEST_P(GELQ2, batched__double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 0, double>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 0, double>(arg);
}

TEST_P(GELQ2, batched__float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GELQ2, batched__double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GELQF, batched__float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 1, float>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 1, float>(arg);
}

TEST_P(GELQF, batched__double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 1, double>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 1, double>(arg);
}

TEST_P(GELQF, batched__float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GELQF, batched__double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<true, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<true, true, 1, rocblas_double_complex>(arg);
}

// strided_batched cases

TEST_P(GELQ2, strided_batched__float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 0, float>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 0, float>(arg);
}

TEST_P(GELQ2, strided_batched__double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 0, double>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 0, double>(arg);
}

TEST_P(GELQ2, strided_batched__float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GELQ2, strided_batched__double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GELQF, strided_batched__float)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 1, float>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 1, float>(arg);
}

TEST_P(GELQF, strided_batched__double)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 1, double>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 1, double>(arg);
}

TEST_P(GELQF, strided_batched__float_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GELQF, strided_batched__double_complex)
{
    Arguments arg = gelqf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gelq2_gelqf_bad_arg<false, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gelq2_gelqf<false, true, 1, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELQ2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELQ2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELQF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELQF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
