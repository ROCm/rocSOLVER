/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syev_heev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> syev_heev_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {evect, uplo}

// case when n == 0, evect == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments syev_heev_setup_arguments(syev_heev_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<char> op = std::get<1>(tup);

    Arguments arg;

    arg.N = size[0];
    arg.lda = size[1];

    arg.evect = op[0];
    arg.uplo_option = op[1];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N; // strideA
    arg.bsb = arg.N; // strideD
    arg.bsc = arg.N; // strideE

    arg.timing = 0;

    return arg;
}

class SYEV : public ::TestWithParam<syev_heev_tuple>
{
protected:
    SYEV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class HEEV : public ::TestWithParam<syev_heev_tuple>
{
protected:
    HEEV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(SYEV, __float)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, false, float>();

    arg.batch_count = 1;
    testing_syev_heev<false, false, float>(arg);
}

TEST_P(SYEV, __double)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, false, double>();

    arg.batch_count = 1;
    testing_syev_heev<false, false, double>(arg);
}

TEST_P(HEEV, __float_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_syev_heev<false, false, rocblas_float_complex>(arg);
}

TEST_P(HEEV, __double_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_syev_heev<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(SYEV, batched__float)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<true, true, float>();

    arg.batch_count = 3;
    testing_syev_heev<true, true, float>(arg);
}

TEST_P(SYEV, batched__double)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<true, true, double>();

    arg.batch_count = 3;
    testing_syev_heev<true, true, double>(arg);
}

TEST_P(HEEV, batched__float_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_syev_heev<true, true, rocblas_float_complex>(arg);
}

TEST_P(HEEV, batched__double_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_syev_heev<true, true, rocblas_double_complex>(arg);
}

// strided_batched tests

TEST_P(SYEV, strided_batched__float)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, true, float>();

    arg.batch_count = 3;
    testing_syev_heev<false, true, float>(arg);
}

TEST_P(SYEV, strided_batched__double)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, true, double>();

    arg.batch_count = 3;
    testing_syev_heev<false, true, double>(arg);
}

TEST_P(HEEV, strided_batched__float_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_syev_heev<false, true, rocblas_float_complex>(arg);
}

TEST_P(HEEV, strided_batched__double_complex)
{
    Arguments arg = syev_heev_setup_arguments(GetParam());

    if(arg.N == 0 && arg.evect == 'N' && arg.uplo_option == 'L')
        testing_syev_heev_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_syev_heev<false, true, rocblas_double_complex>(arg);
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack, SYEV, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, HEEV, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEV, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEV, Combine(ValuesIn(size_range), ValuesIn(op_range)));
