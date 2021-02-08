/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potf2_potrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> potrf_tuple;

// each size_range vector is a {N, lda, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each uplo_range is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {10, 2, 0},
    // normal (valid) samples
    {10, 10, 1},
    {20, 30, 0},
    {50, 50, 1},
    {70, 80, 0}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0}, {640, 960, 1}, {1000, 1000, 0}, {1024, 1024, 1}, {2000, 2000, 0},
};

Arguments potrf_setup_arguments(potrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.uplo_option = uplo;

    arg.timing = 0;
    arg.singular = matrix_size[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class POTF2 : public ::TestWithParam<potrf_tuple>
{
protected:
    POTF2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class POTRF : public ::TestWithParam<potrf_tuple>
{
protected:
    POTRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(POTF2, __float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 0, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 0, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 0, float>(arg);
}

TEST_P(POTF2, __double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 0, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 0, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 0, double>(arg);
}

TEST_P(POTF2, __float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 0, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 0, rocblas_float_complex>(arg);
}

TEST_P(POTF2, __double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 0, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 0, rocblas_double_complex>(arg);
}

TEST_P(POTRF, __float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 1, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 1, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 1, float>(arg);
}

TEST_P(POTRF, __double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 1, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 1, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 1, double>(arg);
}

TEST_P(POTRF, __float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 1, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 1, rocblas_float_complex>(arg);
}

TEST_P(POTRF, __double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, false, 1, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_potf2_potrf<false, false, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, false, 1, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(POTF2, batched__float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 0, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 0, float>(arg);
}

TEST_P(POTF2, batched__double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 0, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 0, double>(arg);
}

TEST_P(POTF2, batched__float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 0, rocblas_float_complex>(arg);
}

TEST_P(POTF2, batched__double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 0, rocblas_double_complex>(arg);
}

TEST_P(POTRF, batched__float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 1, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 1, float>(arg);
}

TEST_P(POTRF, batched__double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 1, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 1, double>(arg);
}

TEST_P(POTRF, batched__float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 1, rocblas_float_complex>(arg);
}

TEST_P(POTRF, batched__double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<true, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<true, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<true, true, 1, rocblas_double_complex>(arg);
}

// strided_batched cases

TEST_P(POTF2, strided_batched__float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 0, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 0, float>(arg);
}

TEST_P(POTF2, strided_batched__double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 0, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 0, double>(arg);
}

TEST_P(POTF2, strided_batched__float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 0, rocblas_float_complex>(arg);
}

TEST_P(POTF2, strided_batched__double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 0, rocblas_double_complex>(arg);
}

TEST_P(POTRF, strided_batched__float)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 1, float>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 1, float>(arg);
}

TEST_P(POTRF, strided_batched__double)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 1, double>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 1, double>(arg);
}

TEST_P(POTRF, strided_batched__float_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 1, rocblas_float_complex>(arg);
}

TEST_P(POTRF, strided_batched__double_complex)
{
    Arguments arg = potrf_setup_arguments(GetParam());

    if(arg.uplo_option == 'L' && arg.N == 0)
        testing_potf2_potrf_bad_arg<false, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_potf2_potrf<false, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_potf2_potrf<false, true, 1, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
